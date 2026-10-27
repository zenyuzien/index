"""Equal-weight rank average index (Borda / DCI style)."""
import numpy as np
import pandas as pd

NAME = "equal_weight_rank_average"
CATEGORY = "equal_weight_rank_average_accessibility_category"
BINS = [0, 20, 40, 60, 80, 100]
LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]


def compute(df, signs):
    """
    1. direction-align every indicator (multiply by its sign so higher = more barrier)
    2. percentile-rank each indicator across all parishes: 100 * rank / (n - 1)
    3. average the ranks with equal weight
    4. re-percentile that average to obtain the final 0-100 index (higher = more barrier)
    Adds the index and its accessibility category (binned on 100 - index, so "Very Low" = least accessible).
    """
    cols = list(signs)
    n = len(df)
    X = df[cols].to_numpy(float) * np.array([signs[c] for c in cols])[None, :]
    PR = np.zeros_like(X)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        PR[:, j] = 100 * ranks / (n - 1)
    prelim = PR.mean(axis=1)
    order = np.argsort(prelim)
    fr = np.empty(n)
    fr[order] = np.arange(n)
    df[NAME] = np.round(100 * fr / (n - 1), 2)
    df[CATEGORY] = pd.cut(100 - df[NAME], bins=BINS, labels=LABELS, include_lowest=True).astype(str)
    return df
