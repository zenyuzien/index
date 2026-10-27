#!/usr/bin/env python3
"""
HAI driver: dataset generation -> rank-average index -> rotated-PCA index -> CSV.

Pipeline
    1. dataset.build_dataset()                 seven indicators for the 64 parishes; the ONLY imputation happens here:
                                               outpatient_facility_density_per_10k, zero rows only, filled by KNN
    2. index_rank_average.compute()            equal-weight rank average index
    3. index_rotated_pca.compute()             rotated-PCA index

    python3 run_hai.py                       # writes hai_dataset.csv next to this file
    python3 run_hai.py --compare OTHER.csv   # also diffs the result against another copy of the dataset
"""
import argparse
import os

import pandas as pd

import dataset
import index_rank_average
import index_rotated_pca

HERE = os.path.dirname(os.path.abspath(__file__))

PCA = index_rotated_pca.NAME
COLUMN_ORDER = [
    "parish", "fips_code",
    "primary_care_clinician_ratio_proxy", "avg_drive_time_to_nearest_pcp_minutes", "pct_households_no_vehicle",
    "non_elderly_uninsured_rate", "low_income_below200fpl_pct", "fixed_broadband_subscription_rate",
    "outpatient_facility_density_per_10k",
    PCA, PCA + "_z",
    index_rank_average.NAME, PCA + "_norm100",
    index_rank_average.CATEGORY, index_rotated_pca.CATEGORY,
]


def compare(new, ref_path):
    ref = pd.read_csv(ref_path)
    m = ref.merge(new, on="fips_code", suffixes=("_ref", "_new"))
    print(f"\nComparison vs {os.path.basename(ref_path)} ({len(m)} parishes)")
    for c in COLUMN_ORDER[2:]:
        a, b = m[c + "_ref"], m[c + "_new"]
        if pd.api.types.is_numeric_dtype(a):
            diff = (a - b).abs()
            print(f"  {c:75s} identical: {(diff < 1e-9).sum()}/{len(m)}  max|diff|={diff.max():.4f}")
        else:
            print(f"  {c:75s} identical: {(a == b).sum()}/{len(m)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", metavar="CSV", help="path of another dataset CSV to diff against")
    ap.add_argument("--out", default=os.path.join(HERE, "hai_dataset.csv"))
    args = ap.parse_args()

    print("1/3 building dataset (zero facility density filled by KNN - the only imputation)")
    df = dataset.build_dataset()
    others = [c for c in dataset.INDICATORS if c != dataset.FACILITY_COL]
    assert df[others].notna().all().all(), "another indicator has missing values - only facility density may be imputed"
    assert (df[dataset.FACILITY_COL] > 0).all(), "a zero facility density was left in"

    print("2/3 equal-weight rank average")
    df = index_rank_average.compute(df, dataset.SIGNS)
    print("3/3 rotated-PCA aggregation")
    df = index_rotated_pca.compute(df, dataset.SIGNS)

    df = df[COLUMN_ORDER]
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    if args.compare:
        compare(df, args.compare)


if __name__ == "__main__":
    main()
