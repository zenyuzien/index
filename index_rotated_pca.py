"""Unweighted linear aggregation of varimax-rotated principal components (SoVI style)."""
import numpy as np
import pandas as pd

NAME = "unweighted_linear_aggregation_of_rotated_pcs"
CATEGORY = NAME + "_norm100_accessibility_category"
BINS = [0, 20, 40, 60, 80, 100]
LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]


def varimax_rotate(L, p, max_sweeps=100, tol=1e-8):
    """Pairwise-rotation varimax. Returns rotated loadings A and the rotation matrix T."""
    K = L.shape[1]
    A, T = L.copy(), np.eye(K)

    def score(M):
        a2, a4 = M ** 2, M ** 4
        return float(np.sum(a4.mean(axis=0) - a2.mean(axis=0) ** 2))

    prev = score(A)
    for _ in range(max_sweeps):
        for k in range(K):
            for l in range(k + 1, K):
                ak, al = A[:, k], A[:, l]
                u, v = ak ** 2 - al ** 2, 2 * ak * al
                U, Vs = u.sum(), v.sum()
                num = 2 * (np.sum(u * v) - U * Vs / p)
                den = np.sum(u ** 2) - np.sum(v ** 2) - (U ** 2 - Vs ** 2) / p
                theta = 0.25 * np.arctan2(num, den)
                c, s = np.cos(theta), np.sin(theta)
                A[:, k], A[:, l] = c * ak + s * al, -s * ak + c * al
                tk, tl = T[:, k].copy(), T[:, l].copy()
                T[:, k], T[:, l] = c * tk + s * tl, -s * tk + c * tl
        cur = score(A)
        if abs(cur - prev) < tol:
            break
        prev = cur
    return A, T


def compute(df, signs):
    """
    1. z-score the indicators; correlation matrix R = Z'Z / (n-1); eigendecompose
    2. keep components with eigenvalue > 1 (Kaiser)
    3. loadings L = V_K sqrt(Lambda_K); varimax rotation -> A, T
    4. standardized rotated scores F = Z V_K Lambda_K^-1/2 T
    5. sign each component by sign(sum_j A_jk * d_j), sum the signed components
    6. min-max scale to 0-100 (higher = more barrier)
    Adds the raw score, z-score, 0-100 score and the accessibility category (binned on 100 - norm100).
    """
    cols = list(signs)
    n, p = len(df), len(cols)
    d = np.array([signs[c] for c in cols], float)
    X = df[cols].to_numpy(float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    R = (Z.T @ Z) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(R)
    o = np.argsort(eigvals)[::-1]
    eigvals, V = eigvals[o], eigvecs[:, o]
    K = int(np.sum(eigvals > 1.0))
    V_K = V[:, :K]
    L = V_K @ np.diag(np.sqrt(eigvals[:K]))
    A, T = varimax_rotate(L, p)
    F = (Z @ V_K) @ np.diag(1 / np.sqrt(eigvals[:K])) @ T
    sign_vec = np.array([1.0 if np.sum(A[:, k] * d) >= 0 else -1.0 for k in range(K)])
    raw = (F * sign_vec[None, :]).sum(axis=1)
    df[NAME] = np.round(raw, 4)
    df[NAME + "_z"] = np.round((raw - raw.mean()) / raw.std(ddof=1), 4)
    df[NAME + "_norm100"] = np.round((raw - raw.min()) / (raw.max() - raw.min()) * 100, 2)
    df[CATEGORY] = pd.cut(100 - df[NAME + "_norm100"], bins=BINS, labels=LABELS, include_lowest=True).astype(str)
    print(f"  PCA components retained (eigenvalue > 1): K = {K}")
    return df
