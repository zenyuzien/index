import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import index_rotated_pca as pc

# ============================================================
# Load + prep data
# ============================================================
df = pd.read_csv('hai_dataset.csv', usecols=range(9))
VARS = ["primary_care_clinician_ratio_proxy", "outpatient_facility_density_per_10k",
        "avg_drive_time_to_nearest_pcp_minutes", "pct_households_no_vehicle",
        "non_elderly_uninsured_rate", "low_income_below200fpl_pct",
        "fixed_broadband_subscription_rate"]
df["outpatient_facility_density_per_10k"] = 1 / df["outpatient_facility_density_per_10k"]
df["fixed_broadband_subscription_rate"] = 100 - df["fixed_broadband_subscription_rate"]
SIGNS = {v: 1 for v in VARS}   # all +1: direction already baked into the two transforms above

base = pd.read_csv('hai_dataset.csv')
parish, fips = base['parish'], base['fips_code']

labels = ["Very Low", "Low", "Medium", "High", "Very High"]
def sdcat(s, t1=0.5, t2=1.5):
    """Bin a score into 5 categories by standard deviation from its own mean."""
    m, sd = s.mean(), s.std()
    return pd.cut(s, [-np.inf, m - t2*sd, m - t1*sd, m + t1*sd, m + t2*sd, np.inf],
                  labels=labels, include_lowest=True).astype(str)

# ============================================================
# 1. PCA category — manual (hand-rolled Givens varimax) rotation, 0.5sigma/1.5sigma bins
# ============================================================
pca_score = pc.compute(df.copy(), SIGNS)['unweighted_linear_aggregation_of_rotated_pcs']
pca_category = sdcat(pca_score)

# ============================================================
# 2. Rank category — proper tie handling (pandas .rank(), average method), 0.5sigma/1.5sigma bins
# ============================================================
rank_score = df[VARS].rank().mean(axis=1)
rank_category = sdcat(rank_score)

# ============================================================
# 3-5. Reused as-is from existing files, not recomputed
# ============================================================
rurality = pd.read_csv('parish_indices_svi_dci_rurality_classified.csv')[['parish', 'rurality_class']]

dci_raw = pd.read_excel('DCI-2019-2023-Scores-Only-11.xlsx', sheet_name='Counties')
dci = dci_raw.loc[dci_raw['State Abbreviation'] == 'LA', ['County ID', '2019-2023 Distress Score']]
dci = dci.rename(columns={'County ID': 'fips_code', '2019-2023 Distress Score': 'dci_index'})

svi_raw = pd.read_csv('SVI_2022_US_county.csv')
svi = svi_raw.loc[svi_raw['ST_ABBR'] == 'LA', ['FIPS', 'RPL_THEMES']]
svi = svi.rename(columns={'FIPS': 'fips_code', 'RPL_THEMES': 'sovi_index'})

# ============================================================
# Assemble + gap check
# ============================================================
out = pd.DataFrame({
    'parish': parish, 'fips_code': fips,
    'pca_category_manual': pca_category,
    'rank_category_proper_ties': rank_category,
})
out = out.merge(rurality, on='parish', how='left')
out = out.merge(dci, on='fips_code', how='left')
out = out.merge(svi, on='fips_code', how='left')

print("rows:", len(out))
print("missing values per column:\n", out.isna().sum())
gaps = out[out.isna().any(axis=1)]
print("GAPS FOUND:" if len(gaps) else "no gaps -- all parishes fully matched")
if len(gaps):
    print(gaps.to_string(index=False))

out.to_csv('parish_comparison_table.csv', index=False)
print("\nsaved parish_comparison_table.csv")
