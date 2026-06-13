import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


# ----------------------------- Utilities -----------------------------

def infer_presence_mask(Xv: np.ndarray) -> np.ndarray:
    """True where sample exists in this view. Missing iff entire row is NaN."""
    Xv = np.asarray(Xv)
    return ~np.all(np.isnan(Xv), axis=1)

def fill_nans_in_observed_rows(X: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Fill NaNs inside observed rows using column means (if any)."""
    X = np.asarray(X, dtype=np.float64).copy()
    if not np.isnan(X[obs_mask]).any():
        return X
    col_means = np.nanmean(X[obs_mask], axis=0)
    ii, jj = np.where(np.isnan(X) & obs_mask[:, None])
    X[ii, jj] = col_means[jj]
    return X

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)

def proj_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Project vector v onto simplex {x>=0, sum x = z}."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("proj_simplex expects a 1D array.")
    if z <= 0:
        raise ValueError("z must be > 0")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(1, v.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(v, z / v.size, dtype=np.float64)

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)

def orthonormal_columns_random(n_rows: int, n_cols: int, rng: np.random.Generator) -> np.ndarray:
    """Random matrix with orthonormal columns (Q from QR)."""
    A = rng.standard_normal((n_rows, n_cols))
    Q, _ = np.linalg.qr(A)
    return Q[:, :n_cols].astype(np.float64)

def svd_procrustes(M: np.ndarray, out_cols: int) -> np.ndarray:
    """
    Return UV^T where M = U Σ V^T. If M is rectangular, UV^T has same shape as M.
    Keeps rank/out_cols up to min dims.
    """
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(out_cols, U.shape[1], Vt.shape[0])
    return (U[:, :r] @ Vt[:r, :]).astype(np.float64)

def safe_svd_uvt(M: np.ndarray) -> np.ndarray:
    """UV^T with full (thin) SVD."""
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return (U @ Vt).astype(np.float64)



# ----------------------------- Utilities -----------------------------


def masked_fro_norm_sq(X: np.ndarray, mask: np.ndarray) -> float:
    """Sum ||x_i||^2 over i where mask is True."""
    Xm = X[mask]
    return float(np.sum(Xm * Xm))


# ----------------------------- Config -----------------------------

@dataclass
class FIMVCVIAConfig:
    n_clusters: int
    n_anchors: int  # m
    mu: float = 1.0

    max_iters: int = 30
    tol: float = 1e-6
    normalize_rows: bool = True

    # embedding + kmeans
    svd_rank: Optional[int] = None  # default = n_clusters
    random_state: int = 0
    verbose: bool = False


# ----------------------------- Model -----------------------------

class FIMVC_VIA:
    """
    FIMVC-VIA (IEEE TNNLS 2024): Fast Incomplete Multi-View Clustering with View-Independent Anchors

    NaN-row convention:
      - X_list[v] is (n, d_v)
      - if an entire row is NaN => sample missing in that view

    Learns:
      - view-specific anchor matrices B_v ∈ R^{d_v × m} with B_v^T B_v = I
      - consensus anchor graph Z ∈ R^{m × n}, columns on simplex
      - view weights gamma ∈ R^V, gamma>=0, sum=1

    After fit():
      - embedding_ : (n, k) from SVD of Z^T
      - labels_    : (n,)
    """

    def __init__(self, cfg: FIMVCVIAConfig):
        self.cfg = cfg
        self.labels_: Optional[np.ndarray] = None
        self.embedding_: Optional[np.ndarray] = None

        self.B_list_: Optional[List[np.ndarray]] = None   # each (d_v, m)
        self.Z_: Optional[np.ndarray] = None              # (m, n)
        self.gamma_: Optional[np.ndarray] = None          # (V,)
        self.masks_: Optional[List[np.ndarray]] = None    # each (n,)
        self.history_: List[Dict] = []

    def _preprocess(self, X_list: List[np.ndarray]) -> (List[np.ndarray], List[np.ndarray]):
        X_proc, masks = [], []
        for Xv in X_list:
            Xv = np.asarray(Xv, dtype=np.float64)
            mv = infer_presence_mask(Xv)
            Xv = fill_nans_in_observed_rows(Xv, mv)
            if self.cfg.normalize_rows and mv.any():
                Xv[mv] = l2_normalize_rows(Xv[mv])
            Xv[~mv] = 0.0  # implement masking (missing rows contribute nothing)
            X_proc.append(Xv)
            masks.append(mv)
        return X_proc, masks

    def _update_B(self, Xv: np.ndarray, fv: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Paper step: Bp = U_m V_m^T where SVD of p = (Xp ⊙ Ap) Z^T.
        In our row-format Xv is (n,d). Mask fv in {0,1}^n.
        Equivalent p = Xv^T diag(fv) Z^T  -> (d,m).
        """
        # Xv_masked: (n,d)
        Xv_masked = Xv * fv[:, None]
        # p: (d,m)
        p = Xv_masked.T @ Z.T
        B = safe_svd_uvt(p)  # (d,m) with orthonormal columns (when m<=d)
        # ensure shape exactly (d,m)
        return B[:, :Z.shape[0]]

    def _update_Z(self, X_list: List[np.ndarray], f_list: List[np.ndarray],
                  B_list: List[np.ndarray], gamma: np.ndarray, mu: float) -> np.ndarray:
        """
        Paper step: optimize Z column-wise, each column projected onto simplex.
        We implement a stable derivation consistent with their per-column quadratic:
          For sample i:
            a_i = V*mu + Σ_v (gamma_v^2 * f_v[i])
            b_i (m-vector) = Σ_v (gamma_v^2 * f_v[i]) * (x_i^(v) · B_v)   (=> row of C_v)
            y_i = b_i / a_i
            z_i = proj_simplex(y_i)
        This ensures: if sample missing in a view, it contributes 0 for that view.
        """
        V = len(X_list)
        n = X_list[0].shape[0]
        m = self.cfg.n_anchors

        gamma2 = (gamma ** 2).astype(np.float64)

        # Precompute C_v = X_v B_v  (n×m), but only meaningful where present
        C_list = []
        for v in range(V):
            C = X_list[v] @ B_list[v]  # (n,m)
            C_list.append(C)

        Z = np.zeros((m, n), dtype=np.float64)
        Vmu = V * float(mu)

        for i in range(n):
            # scalar denominator
            denom = Vmu
            b = np.zeros((m,), dtype=np.float64)
            for v in range(V):
                if f_list[v][i] > 0.5:
                    w = gamma2[v]
                    denom += w
                    b += w * C_list[v][i]
            denom = max(denom, 1e-12)
            y = b / denom
            Z[:, i] = proj_simplex(y, 1.0)

        return Z

    def _update_gamma(self, X_list: List[np.ndarray], masks: List[np.ndarray],
                      B_list: List[np.ndarray], Z: np.ndarray) -> np.ndarray:
        """
        Paper step: min Σ gamma_p^2 * Δ_p  s.t. gamma>=0, sum=1
        => gamma_p ∝ 1/Δ_p where Δ_p = || XpHp - Bp Z Hp ||_F^2.

        Our Δ_v computed only over observed samples in that view:
          Δ_v = Σ_{i:mask} || x_i^(v) - B_v z_i ||^2
        """
        V = len(X_list)
        n = X_list[0].shape[0]
        m = Z.shape[0]

        deltas = np.zeros((V,), dtype=np.float64)
        for v in range(V):
            mv = masks[v]
            if mv.sum() == 0:
                deltas[v] = np.inf
                continue
            # pred for observed samples: (n_obs,d) = (Z_obs^T @ B_v^T)
            Z_obs = Z[:, mv].T  # (n_obs,m)
            pred = Z_obs @ B_list[v].T  # (n_obs,d_v)
            resid = X_list[v][mv] - pred
            deltas[v] = float(np.sum(resid * resid))

        inv = 1.0 / np.maximum(deltas, 1e-12)
        gamma = inv / np.maximum(inv.sum(), 1e-12)
        return gamma

    def _objective(self, X_list, masks, B_list, Z, gamma, mu) -> float:
        """Compute J = Σ gamma^2 ||XHp - BZHp||^2 + mu||Z||^2 (masked)."""
        V = len(X_list)
        J = 0.0
        for v in range(V):
            mv = masks[v]
            if mv.sum() == 0:
                continue
            Z_obs = Z[:, mv].T
            pred = Z_obs @ B_list[v].T
            resid = X_list[v][mv] - pred
            J += float((gamma[v] ** 2) * np.sum(resid * resid))
        J += float(mu * np.sum(Z * Z))
        return float(J)

    def _embed_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """Paper says: perform SVD on Z. We return n×k embedding from SVD(Z^T)."""
        k = self.cfg.svd_rank if self.cfg.svd_rank is not None else self.cfg.n_clusters
        # Z^T is (n,m)
        U, _, _ = randomized_svd(Z.T, n_components=k, n_iter=3, random_state=self.cfg.random_state)
        return U.astype(np.float64)

    def fit(self, X_list: List[np.ndarray]) -> "FIMVC_VIA":
        if len(X_list) < 2:
            raise ValueError("FIMVC-VIA requires at least 2 views.")

        X_list = [np.asarray(X, dtype=np.float64) for X in X_list]
        n = X_list[0].shape[0]
        for X in X_list:
            if X.shape[0] != n:
                raise ValueError("All views must have same number of samples (rows).")

        rng = np.random.default_rng(self.cfg.random_state)
        X_proc, masks = self._preprocess(X_list)
        f_list = [mv.astype(np.float64) for mv in masks]

        V = len(X_proc)
        m = self.cfg.n_anchors
        d_list = [X.shape[1] for X in X_proc]
        for d in d_list:
            if m > d:
                raise ValueError(f"Need n_anchors (m) <= d_v for each view. Got m={m}, d_v={d}.")

        # init
        B_list = [orthonormal_columns_random(d_list[v], m, rng) for v in range(V)]
        Z = rng.random((m, n))
        Z /= np.maximum(Z.sum(axis=0, keepdims=True), 1e-12)
        gamma = np.full((V,), 1.0 / V, dtype=np.float64)

        prev_J = None
        for it in range(1, self.cfg.max_iters + 1):
            # Step 1: update each B_v (SVD)
            for v in range(V):
                B_list[v] = self._update_B(X_proc[v], f_list[v], Z)

            # Step 2: update Z (simplex projection per column)
            Z = self._update_Z(X_proc, f_list, B_list, gamma, self.cfg.mu)

            # Step 3: update gamma
            gamma = self._update_gamma(X_proc, masks, B_list, Z)

            # monitor
            J = self._objective(X_proc, masks, B_list, Z, gamma, self.cfg.mu)
            self.history_.append({"iter": it, "obj": J, "gamma": gamma.copy()})
            if self.cfg.verbose:
                print(f"[FIMVC-VIA] iter={it:03d} obj={J:.6e} gamma={gamma}")

            if prev_J is not None:
                rel = abs(prev_J - J) / (abs(prev_J) + 1e-12)
                if rel < self.cfg.tol:
                    break
            prev_J = J

        # embedding + kmeans
        U = self._embed_from_Z(Z)
        km = KMeans(n_clusters=self.cfg.n_clusters, n_init=20, random_state=self.cfg.random_state)
        labels = km.fit_predict(U)

        self.B_list_ = B_list
        self.Z_ = Z
        self.gamma_ = gamma
        self.masks_ = masks
        self.embedding_ = U
        self.labels_ = labels
        return self

    def fit_predict(self, X_list: List[np.ndarray]) -> np.ndarray:
        self.fit(X_list)
        return self.labels_

    def get_embedding(self) -> np.ndarray:
        if self.embedding_ is None:
            raise RuntimeError("Call fit() first.")
        return self.embedding_
