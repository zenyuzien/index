"""
Bayesian population smoothing for the two count-over-population indicators
(primary_care_clinician_ratio_proxy, outpatient_facility_density_per_10k),
then re-runs the manual PCA pipeline (build_comparison_table.py) on the
smoothed values.

Why: both indicators are a small raw count (physicians, facilities) divided
by parish population. In low-population parishes a single physician or
facility swings the ratio by thousands of points (e.g. Grant parish: 1
physician on a population of 22,109 gives a raw ratio of 22,000:1 -- almost
pure sampling noise, not a real access gap).

Method: Marshall (1991) empirical-Bayes Poisson-Gamma rate smoothing. Each
parish's raw rate (count / population) is shrunk toward the statewide rate,
weighted by population -- small parishes shrink hard, large parishes barely
move.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import index_rotated_pca as pc

VARS = ["primary_care_clinician_ratio_proxy", "outpatient_facility_density_per_10k",
        "avg_drive_time_to_nearest_pcp_minutes", "pct_households_no_vehicle",
        "non_elderly_uninsured_rate", "low_income_below200fpl_pct",
        "fixed_broadband_subscription_rate"]

labels = ["Very Low", "Low", "Medium", "High", "Very High"]
def sdcat(s, t1=0.5, t2=1.5):
    m, sd = s.mean(), s.std()
    return pd.cut(s, [-np.inf, m - t2*sd, m - t1*sd, m + t1*sd, m + t2*sd, np.inf],
                  labels=labels, include_lowest=True).astype(str)

def eb_poisson_smooth(count, pop):
    """Marshall (1991): shrink each parish's raw rate toward the pooled rate."""
    y = count / pop
    m = count.sum() / pop.sum()
    n = len(count)
    s2 = max(np.sum(pop * (y - m) ** 2) / pop.sum() - m * n / pop.sum(), 0.0)
    w = s2 / (s2 + m / pop)
    return w * y + (1 - w) * m


def main():
    base = pd.read_csv('hai_dataset.csv')
    parish, fips = base['parish'], base['fips_code']
    df = pd.read_csv('hai_dataset.csv', usecols=range(9))

    pep = pd.read_csv("raw_cache/co-est2024-alldata.csv", encoding="latin1")
    pep = pep[(pep["STATE"] == 22) & (pep["COUNTY"] > 0)].copy()
    pep["fips_code"] = 22000 + pep["COUNTY"]
    pop = pep.set_index("fips_code")["POPESTIMATE2024"]
    pop_v = base[["parish", "fips_code"]].merge(pop.rename("pop"), on="fips_code", how="left")["pop"].to_numpy(float)

    phy = pd.read_csv("data/la-2025-primary_care_physicians.csv")
    phy_v = parish.map(phy.set_index("parish")["num_primary_care_physcian"]).fillna(0).to_numpy(float)

    fac = pd.read_csv("data/la_health_facilities_with_geom.csv")
    fac_v = parish.map(fac.groupby("parish").size()).fillna(0).to_numpy(float)

    phy_rate_smooth = eb_poisson_smooth(phy_v, pop_v)
    fac_rate_smooth = eb_poisson_smooth(fac_v, pop_v)

    df["primary_care_clinician_ratio_proxy"] = 1.0 / phy_rate_smooth          # people per physician, higher = worse
    df["outpatient_facility_density_per_10k"] = 1.0 / (fac_rate_smooth * 10000)  # 1/x, matches build_comparison_table.py
    df["fixed_broadband_subscription_rate"] = 100 - df["fixed_broadband_subscription_rate"]
    SIGNS = {v: 1 for v in VARS}

    pca_score = pc.compute(df.copy(), SIGNS)['unweighted_linear_aggregation_of_rotated_pcs']
    pca_category = sdcat(pca_score)

    rank_score = df[VARS].rank().mean(axis=1)
    rank_category = sdcat(rank_score)

    rurality = pd.read_csv('parish_indices_svi_dci_rurality_classified.csv')[['parish', 'rurality_class']]
    dci_raw = pd.read_excel('DCI-2019-2023-Scores-Only-11.xlsx', sheet_name='Counties')
    dci = dci_raw.loc[dci_raw['State Abbreviation'] == 'LA', ['County ID', '2019-2023 Distress Score']]
    dci = dci.rename(columns={'County ID': 'fips_code', '2019-2023 Distress Score': 'dci_index'})
    svi_raw = pd.read_csv('SVI_2022_US_county.csv')
    svi = svi_raw.loc[svi_raw['ST_ABBR'] == 'LA', ['FIPS', 'RPL_THEMES']]
    svi = svi.rename(columns={'FIPS': 'fips_code', 'RPL_THEMES': 'sovi_index'})

    out = pd.DataFrame({
        'parish': parish, 'fips_code': fips,
        'pca_category_manual_smoothed': pca_category,
        'rank_category_proper_ties_smoothed': rank_category,
    })
    out = out.merge(rurality, on='parish', how='left').merge(dci, on='fips_code', how='left').merge(svi, on='fips_code', how='left')
    out.to_csv('parish_comparison_table_smoothed.csv', index=False)

    print(f"agreement with rank: {(out['pca_category_manual_smoothed'] == out['rank_category_proper_ties_smoothed']).sum()}/64")
    hv = out[(out['pca_category_manual_smoothed'].isin(["High", "Very High"])) & (out['rurality_class'] != "Rural")]
    lv = out[(out['pca_category_manual_smoothed'].isin(["Very Low", "Low"])) & (out['rurality_class'] != "Urban")]
    print(f"H/VH barrier & not-Rural ({len(hv)}):", sorted(hv['parish']))
    print(f"L/VL barrier & not-Urban ({len(lv)}):", sorted(lv['parish']))
    print("\nsaved parish_comparison_table_smoothed.csv")


if __name__ == "__main__":
    main()
