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


# ----------------------------- Config -----------------------------

@dataclass
class DVSAIConfig:
    n_clusters: int
    T: int = 5
    k_base: int = 10  # paper sets l_t = m_t = t*k (t=1..T); you control k_base

    beta: float = 1e-2
    gamma: float = 1e2  # appears in alpha update via hv,t + gamma (paper Eq. 20-21)

    max_iters: int = 30
    tol: float = 1e-6
    normalize_rows: bool = True

    # SVD settings for integration (U from SVD(L))
    svd_rank: Optional[int] = None  # default = n_clusters
    svd_oversample: int = 10
    random_state: int = 0
    verbose: bool = False


# ----------------------------- DVSAI Model -----------------------------

class DVSAI_IMVC:
    """
    DVSAI: Diverse View-Shared Anchors based Incomplete Multi-View Clustering (AAAI'24)
    Adapted to NaN-row missingness.

    Inputs:
      X_list: list of views, each shape (n, d_v); missing samples = all-NaN row.

    Outputs after fit():
      - labels_ : (n,) k-means labels on embedding U
      - embedding_ : (n, k) spectral embedding (U)
      - G_list_ : list of G_t, each (m_t, n)
      - A_list_ : list of A_t, each (l_t, m_t)
      - P_list_ : list over t of list over v of P_{v,t}, each (d_v, l_t)
      - alpha_ : (V, T) weights per space (columns sum to 1)
      - masks_ : list of presence masks per view
    """

    def __init__(self, cfg: DVSAIConfig):
        self.cfg = cfg

        self.labels_: Optional[np.ndarray] = None
        self.embedding_: Optional[np.ndarray] = None

        self.G_list_: Optional[List[np.ndarray]] = None
        self.A_list_: Optional[List[np.ndarray]] = None
        self.P_list_: Optional[List[List[np.ndarray]]] = None
        self.alpha_: Optional[np.ndarray] = None

        self.masks_: Optional[List[np.ndarray]] = None
        self.history_: List[Dict] = []

    def _space_shapes(self) -> Tuple[List[int], List[int]]:
        """Return (l_list, m_list) for spaces t=1..T."""
        l_list = [(t + 1) * self.cfg.k_base for t in range(self.cfg.T)]
        m_list = [(t + 1) * self.cfg.k_base for t in range(self.cfg.T)]
        return l_list, m_list

    def _preprocess(self, X_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        V = len(X_list)
        masks = []
        X_proc = []
        for v in range(V):
            Xv = np.asarray(X_list[v], dtype=np.float64)
            mv = infer_presence_mask(Xv)
            Xv = fill_nans_in_observed_rows(Xv, mv)
            if self.cfg.normalize_rows:
                Xv[mv] = l2_normalize_rows(Xv[mv])
            # set missing rows to 0 to implement the "⊙ Fv" masking
            Xv[~mv] = 0.0
            masks.append(mv)
            X_proc.append(Xv)
        return X_proc, masks

    def _update_P_vt(self, Xv: np.ndarray, fv: np.ndarray, Gt: np.ndarray, At: np.ndarray, l_t: int) -> np.ndarray:
        """
        Step-1: P_{v,t} = UV^T where SVD of (Xv ⊙ Fv) Gt^T At^T.
        Here Xv is (n,dv), paper uses dv×n, so we compute with Xv.T.
        """
        # Xv_masked_T: (dv, n)
        Xv_masked_T = (Xv.T * fv[None, :])  # fv ∈ {0,1}^n
        # M: (dv, l_t)
        M = Xv_masked_T @ Gt.T @ At.T
        P = svd_procrustes(M, out_cols=l_t)  # dv×l_t
        return P

    def _update_A_t(self, X_list: List[np.ndarray], f_list: List[np.ndarray], P_vt_list: List[np.ndarray],
                   Gt: np.ndarray, alpha_col: np.ndarray) -> np.ndarray:
        """
        Step-2: A_t = UV^T where SVD of sum_v alpha_{v,t}^2 P_{v,t}^T (Xv ⊙ Fv) Gt^T.
        """
        V = len(X_list)
        l_t = P_vt_list[0].shape[1]
        m_t = Gt.shape[0]

        S = np.zeros((l_t, m_t), dtype=np.float64)
        for v in range(V):
            a2 = float(alpha_col[v] ** 2)
            if a2 == 0:
                continue
            Xv = X_list[v]          # (n, dv)
            fv = f_list[v].astype(np.float64)  # (n,)
            P = P_vt_list[v]        # (dv, l_t)

            Xv_masked_T = (Xv.T * fv[None, :])  # (dv, n)
            # dv×m_t
            tmp = Xv_masked_T @ Gt.T
            # l_t×m_t
            S += a2 * (P.T @ tmp)

        A = safe_svd_uvt(S)  # (l_t, m_t) with orthonormal columns if m_t<=l_t
        return A

    def _update_G_t(self, X_list: List[np.ndarray], f_list: List[np.ndarray], P_vt_list: List[np.ndarray],
                   At: np.ndarray, alpha_col: np.ndarray, beta: float) -> np.ndarray:
        """
        Step-3: Update G_t column-wise via simplex projection.
        Paper gives closed-form with truncation; we do robust simplex projection per column.
        """
        V = len(X_list)
        m_t = At.shape[1]
        n = X_list[0].shape[0]

        # numerator: m_t×n
        numer = np.zeros((m_t, n), dtype=np.float64)
        denom = np.zeros((n,), dtype=np.float64)

        AtT = At.T  # (m_t, l_t)

        for v in range(V):
            a2 = float(alpha_col[v] ** 2)
            if a2 == 0:
                continue
            Xv = X_list[v]                  # (n, dv)
            fv = f_list[v].astype(np.float64)  # (n,)
            P = P_vt_list[v]                # (dv, l_t)

            Xv_masked_T = (Xv.T * fv[None, :])   # (dv, n)
            # l_t×n
            Z = P.T @ Xv_masked_T
            # m_t×n
            numer += a2 * (AtT @ Z)

            denom += a2 * (fv + beta)

        denom = np.maximum(denom, 1e-12)
        Q = numer / denom[None, :]

        G = np.zeros((m_t, n), dtype=np.float64)
        for j in range(n):
            G[:, j] = proj_simplex(Q[:, j], 1.0)

        return G

    def _compute_h_vt(self, Xv: np.ndarray, fv: np.ndarray, P: np.ndarray, At: np.ndarray, Gt: np.ndarray,
                      beta: float, gamma: float) -> float:
        """
        hv,t = ||XvSv - PAtGtSv||_F^2 + beta||Gt||_F^2 + gamma
        Implemented with masking: missing samples (fv=0) do not contribute.
        """
        # Xv.T is dv×n
        Xv_masked_T = (Xv.T * fv[None, :])  # dv×n
        pred = P @ At @ (Gt * fv[None, :])  # dv×n, zeroed on missing columns
        resid = Xv_masked_T - pred
        return float(np.sum(resid * resid) + beta * np.sum(Gt * Gt) + gamma)

    def _update_alpha(self, H: np.ndarray) -> np.ndarray:
        """
        Step-4: alpha_{v,t} = (1/h_{v,t}) / sum_v (1/h_{v,t})  (per space t).
        """
        invH = 1.0 / np.maximum(H, 1e-12)
        alpha = invH / np.maximum(invH.sum(axis=0, keepdims=True), 1e-12)  # normalize per t
        return alpha

    def _integrate_embedding(self, G_list: List[np.ndarray], alpha: np.ndarray) -> np.ndarray:
        """
        Integration (paper Eq. 24-25):
          - Compute Gb_t = M_t^{-1/2} G_t (row-normalized by row sums)
          - Build L by concatenating blocks alpha_{v,t} * Gb_t^T for all (v,t)
          - SVD(L) => U; return U[:, :k]
        """
        V, T = alpha.shape
        n = G_list[0].shape[1]

        blocks = []
        for t in range(T):
            Gt = G_list[t]  # (m_t, n)
            row_sums = np.sum(Gt, axis=1) + 1e-12
            Gb = (Gt / np.sqrt(row_sums)[:, None])  # M^{-1/2} G
            Bt = Gb.T  # (n, m_t)

            for v in range(V):
                blocks.append((alpha[v, t] * Bt))

        L = np.concatenate(blocks, axis=1)  # (n, sum_t m_t * V)

        k = self.cfg.svd_rank if self.cfg.svd_rank is not None else self.cfg.n_clusters
        # randomized SVD is stable for tall matrices
        U, _, _ = randomized_svd(
            L, n_components=k, n_iter=3, random_state=self.cfg.random_state
        )
        return U.astype(np.float64)

    def fit(self, X_list: List[np.ndarray]) -> "DVSAI_IMVC":
        if len(X_list) < 2:
            raise ValueError("DVSAI requires at least 2 views.")

        X_list = [np.asarray(X, dtype=np.float64) for X in X_list]
        n = X_list[0].shape[0]
        for X in X_list:
            if X.shape[0] != n:
                raise ValueError("All views must have the same number of samples (rows).")

        rng = np.random.default_rng(self.cfg.random_state)
        X_proc, masks = self._preprocess(X_list)
        f_list = [mv.astype(np.float64) for mv in masks]  # fv ∈ {0,1}^n

        V = len(X_proc)
        l_list, m_list = self._space_shapes()

        # sanity constraints
        d_list = [X.shape[1] for X in X_proc]
        for t in range(self.cfg.T):
            if l_list[t] <= 0 or m_list[t] <= 0:
                raise ValueError("Invalid space sizes.")
            if m_list[t] > l_list[t]:
                raise ValueError(f"Need m_t <= l_t for A_t^T A_t = I. Got m={m_list[t]}, l={l_list[t]}.")
            for v in range(V):
                if l_list[t] > d_list[v]:
                    raise ValueError(f"Need l_t <= d_v. Got l={l_list[t]} > d_v={d_list[v]} for view {v}.")

        # initialize variables
        P_list = []
        A_list = []
        G_list = []

        for t in range(self.cfg.T):
            l_t, m_t = l_list[t], m_list[t]

            # P_{v,t}
            P_vt = [orthonormal_columns_random(d_list[v], l_t, rng) for v in range(V)]
            P_list.append(P_vt)

            # A_t
            A_t = orthonormal_columns_random(l_t, m_t, rng)
            A_list.append(A_t)

            # G_t (simplex columns)
            G_t = rng.random((m_t, n))
            G_t /= np.maximum(G_t.sum(axis=0, keepdims=True), 1e-12)
            G_list.append(G_t.astype(np.float64))

        # alpha (V×T), normalized per t (space)
        alpha = np.full((V, self.cfg.T), 1.0 / V, dtype=np.float64)

        prev_obj = None
        for it in range(1, self.cfg.max_iters + 1):
            # Step-1: update all P_{v,t}
            for t in range(self.cfg.T):
                l_t = l_list[t]
                for v in range(V):
                    P_list[t][v] = self._update_P_vt(
                        Xv=X_proc[v], fv=f_list[v], Gt=G_list[t], At=A_list[t], l_t=l_t
                    )

            # Step-2: update all A_t
            for t in range(self.cfg.T):
                A_list[t] = self._update_A_t(
                    X_list=X_proc, f_list=f_list, P_vt_list=P_list[t],
                    Gt=G_list[t], alpha_col=alpha[:, t]
                )

            # Step-3: update all G_t
            for t in range(self.cfg.T):
                G_list[t] = self._update_G_t(
                    X_list=X_proc, f_list=f_list, P_vt_list=P_list[t],
                    At=A_list[t], alpha_col=alpha[:, t], beta=self.cfg.beta
                )

            # Step-4: update alpha via hv,t
            H = np.zeros((V, self.cfg.T), dtype=np.float64)
            for t in range(self.cfg.T):
                for v in range(V):
                    H[v, t] = self._compute_h_vt(
                        Xv=X_proc[v], fv=f_list[v], P=P_list[t][v], At=A_list[t], Gt=G_list[t],
                        beta=self.cfg.beta, gamma=self.cfg.gamma
                    )
            alpha = self._update_alpha(H)

            # monitor a proxy objective (same structure as paper, up to constants)
            obj = float(np.sum((alpha ** 2) * H))
            self.history_.append({"iter": it, "obj": obj, "alpha": alpha.copy()})
            if self.cfg.verbose:
                print(f"[DVSAI] iter={it:03d} obj={obj:.6e}")

            if prev_obj is not None:
                rel = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
                if rel < self.cfg.tol:
                    break
            prev_obj = obj

        # Integration -> embedding U
        U = self._integrate_embedding(G_list, alpha)
        # k-means on U
        km = KMeans(n_clusters=self.cfg.n_clusters, n_init=20, random_state=self.cfg.random_state)
        labels = km.fit_predict(U)

        self.embedding_ = U
        self.labels_ = labels
        self.G_list_ = G_list
        self.A_list_ = A_list
        self.P_list_ = P_list
        self.alpha_ = alpha
        self.masks_ = masks
        return self

    def fit_predict(self, X_list: List[np.ndarray]) -> np.ndarray:
        self.fit(X_list)
        return self.labels_

    def get_embedding(self) -> np.ndarray:
        if self.embedding_ is None:
            raise RuntimeError("Call fit() first.")
        return self.embedding_


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
