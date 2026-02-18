from helper_functions.embed_methods import adm_plus, apmc_embed
from helper_functions.embed_utils import Create_Asym_Tran_Kernel
from helper_functions.embed_utils import row_norm
from helper_functions.embed_methods import SVD_trick

import numpy as np
import scipy.sparse as sp
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


@dataclass
class APMCConfig:
    kernel_scale1: float = 1.0
    kernel_scale2: float = 1.0
    scale_mode: str = "median"  # "median" or "sigma"

    embed_dim: int = 30 

class APMC:
    def __init__(self, cfg: APMCConfig):
        self.cfg = cfg

    def _preprocess(
            self, 
            X1: np.ndarray, 
            X2: np.ndarray
            ):
        mask1 = infer_presence_mask(X1)
        mask2 = infer_presence_mask(X2)
        mask_both = mask1 & mask2
        masks = [mask1, mask2]
        # split to reference and aligned
        X1_ref = X1[mask_both]
        X1_aligned = np.concatenate(
            [X1_ref, X1[mask1 & ~mask_both]], axis=0)
        X2_ref = X2[mask_both]
        X2_aligned = np.concatenate(
            [X2_ref, X2[mask2 & ~mask_both]], axis=0)
        return X1_ref, X1_aligned, X2_ref, X2_aligned, masks

    def fit(self, X1: np.ndarray, X2: np.ndarray) -> "APMC":
        (X1_ref, X1_aligned, X2_ref,
          X2_aligned, masks) = self._preprocess(X1, X2)
        # get sizes
        n_anchors = X1_ref.shape[0] 
        n1_full = X1_aligned.shape[0]
        n2_full = X2_aligned.shape[0]

        # compute kernels
        A1, _, _ = Create_Asym_Tran_Kernel(X1_aligned, X1_ref, 
                                             self.cfg.kernel_scale1, 
                                             self.cfg.scale_mode)
        A2, _, _ = Create_Asym_Tran_Kernel(X2_aligned, X2_ref,
                                             self.cfg.kernel_scale2, 
                                             self.cfg.scale_mode)
        # compute the embedding for each mode
        Q1 = row_norm(A1)
        Q2 = row_norm(A2)
        mask_both = masks[0] & masks[1]
        # unify kernels
        Z = np.zeros((n1_full + n2_full - n_anchors, 
                      n_anchors), dtype=np.float64)
        Z[mask_both, :] = 0.5*(Q2[:n_anchors, :] + Q1[:n_anchors, :])
        Z[masks[0] & ~mask_both, :] = Q1[n_anchors:, :]
        Z[masks[1] & ~mask_both, :] = Q2[n_anchors:, :]

        # normalize Z
        col_sum = np.array(Z.sum(axis=0)).flatten()
        # Handle division by zero in col_sum
        col_sum[col_sum == 0] = 1
        # Compute the inverse of column sums
        inv_sum = col_sum ** (-0.5)
        # remove inf values in inv_sum
        inv_sum[np.isinf(inv_sum)] = 1
        # Create diagonal matrix with inverse column sums
        norm_mat = sp.diags(inv_sum)
        # Normalize columns
        Z_norm = Z @ norm_mat

        vals, vecs = SVD_trick(Z_norm, self.cfg.embed_dim)
        return vecs[:, 1:self.cfg.embed_dim + 1]
    
    def get_embedding(self) -> np.ndarray:
        if self.embedding_ is None:
            raise RuntimeError("Call fit() first.")
        return self.embedding_
# ----------------------------- Config -----------------------------

@dataclass
class ADMPlusConfig:
    t: int = 0.1
    kernel_scale1: float = 1.0
    kernel_scale2: float = 1.0
    scale_mode: str = "median"  # "median" or "sigma"
    embed_dim: int = 30  

# ----------------------------- Model -----------------------------

class ADM_PLUS:
    """
    Class for our ADM+ algorirthm for embedding multiview
    data with missing samples from both views.
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

    def __init__(self, cfg: ADMPlusConfig):
        self.cfg = cfg
        self.embedding_: Optional[np.ndarray] = None
        self.vecs_: Optional[np.ndarray] = None
        self.vals_: Optional[np.ndarray] = None

    def _preprocess(
            self, 
            X1: np.ndarray, 
            X2: np.ndarray
            ):
        mask1 = infer_presence_mask(X1)
        mask2 = infer_presence_mask(X2)
        mask_both = mask1 & mask2
        masks = [mask1, mask2]
        # split to reference and aligned
        X1_ref = X1[mask_both]
        X1_aligned = np.concatenate(
            [X1_ref, X1[mask1 & ~mask_both]], axis=0)
        X2_ref = X2[mask_both]
        X2_aligned = np.concatenate(
            [X2_ref, X2[mask2 & ~mask_both]], axis=0)
        return X1_ref, X1_aligned, X2_ref, X2_aligned, masks

    def fit(
            self, 
            X1: np.ndarray,
            X2: np.ndarray
            ) -> "ADM_PLUS":

        (X1_ref, X1_aligned, X2_ref,
          X2_aligned, masks) = self._preprocess(X1, X2)
        # get sizes
        n_anchors = X1_ref.shape[0] 
        n1_full = X1.shape[0]
        n2_full = X2.shape[0]

        # compute kernels
        A1, _, _ = Create_Asym_Tran_Kernel(X1_aligned, X1_ref, 
                                             self.cfg.kernel_scale1, 
                                             self.cfg.scale_mode)
        A2, _, _ = Create_Asym_Tran_Kernel(X2_aligned, X2_ref,
                                             self.cfg.kernel_scale2, 
                                             self.cfg.scale_mode)
        # compute the embedding for each mode
        embed1 = adm_plus(embed_dim=self.cfg.embed_dim,
                          t=self.cfg.t, A1=A1, 
                          K2_ref=A2[:n_anchors, :],
                          return_vecs=False)
        embed2 = adm_plus(embed_dim=self.cfg.embed_dim,
                          t=self.cfg.t, A1=A2,
                          K2_ref=A1[:n_anchors, :],
                          return_vecs=False)
        # add nans back to the aligned embeddings
        # to apply APMC on the full set of samples
        embed1_full = np.full((n1_full, self.cfg.embed_dim), np.nan, dtype=np.float64)
        embed2_full = np.full((n2_full, self.cfg.embed_dim), np.nan, dtype=np.float64)
        embed1_full[masks[0], :] = embed1
        embed2_full[masks[1], :] = embed2
        # combine the two embeddings with APMC
        apmc_config = APMCConfig(
            kernel_scale1=1.0, 
            kernel_scale2=1.0, 
            scale_mode="median", 
            embed_dim=self.cfg.embed_dim
        )
        apmc = APMC(apmc_config)
        self.embedding_ = apmc.fit(embed1_full, embed2_full).get_embedding()
        return self


    def get_embedding(self) -> np.ndarray:
        if self.embedding_ is None:
            raise RuntimeError("Call fit() first.")
        return self.embedding_
