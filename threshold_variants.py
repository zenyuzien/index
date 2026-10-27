"""
Re-runs the unsmoothed manual PCA pipeline (build_comparison_table.py's own
math) at different sigma-binning thresholds, to show how the Very Low..Very
High category counts -- and therefore the PCA/rank agreement and the
rurality-mismatch counts -- depend on where those bin edges are drawn.

build_comparison_table.py itself always uses sdcat(t1=0.5, t2=1.5); this
script reruns the same scores at that default plus a looser t1=0.75 and
reports the same three summary numbers for each.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import index_rotated_pca as pc

VARS = ["primary_care_clinician_ratio_proxy", "outpatient_facility_density_per_10k",
        "avg_drive_time_to_nearest_pcp_minutes", "pct_households_no_vehicle",
        "non_elderly_uninsured_rate", "low_income_below200fpl_pct",
        "fixed_broadband_subscription_rate"]

labels = ["Very Low", "Low", "Medium", "High", "Very High"]
def sdcat(s, t1, t2):
    m, sd = s.mean(), s.std()
    return pd.cut(s, [-np.inf, m - t2*sd, m - t1*sd, m + t1*sd, m + t2*sd, np.inf],
                  labels=labels, include_lowest=True).astype(str)


def run(t1, t2):
    base = pd.read_csv('hai_dataset.csv')
    parish = base['parish']
    df = pd.read_csv('hai_dataset.csv', usecols=range(9))
    df["outpatient_facility_density_per_10k"] = 1 / df["outpatient_facility_density_per_10k"]
    df["fixed_broadband_subscription_rate"] = 100 - df["fixed_broadband_subscription_rate"]
    SIGNS = {v: 1 for v in VARS}

    pca_score = pc.compute(df.copy(), SIGNS)['unweighted_linear_aggregation_of_rotated_pcs']
    pca_category = sdcat(pca_score, t1, t2)
    rank_score = df[VARS].rank().mean(axis=1)
    rank_category = sdcat(rank_score, t1, t2)

    rurality = pd.read_csv('parish_indices_svi_dci_rurality_classified.csv')[['parish', 'rurality_class']]
    out = pd.DataFrame({'parish': parish, 'pca': pca_category, 'rank': rank_category}).merge(rurality, on='parish')

    agree = (out['pca'] == out['rank']).sum()
    hv = sorted(out.loc[(out['pca'].isin(["High", "Very High"])) & (out['rurality_class'] != "Rural"), 'parish'])
    lv = sorted(out.loc[(out['pca'].isin(["Very Low", "Low"])) & (out['rurality_class'] != "Urban"), 'parish'])

    print(f"t1={t1} t2={t2}  std={pca_score.std():.4f}  agreement with rank: {agree}/64")
    print(f"  H/VH barrier & not-Rural ({len(hv)}):", hv)
    print(f"  L/VL barrier & not-Urban ({len(lv)}):", lv)
    return out


if __name__ == "__main__":
    run(0.5, 1.5)
    print()
    run(0.75, 1.5)
