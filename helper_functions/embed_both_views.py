from helper_functions.embed_methods import adm_plus, apmc_embed
from helper_functions.embed_utils import Create_Asym_Tran_Kernel, dist2kernel
from helper_functions.embed_utils import row_norm
from helper_functions.embed_methods import SVD_trick

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


@dataclass
class EmbeddingConfig:
    kernel_scale1: float = 1.0
    kernel_scale2: float = 1.0
    scale_mode: str = "median"  # "median" or "sigma"
    embed_dim: int = 30 


class EmbeddingMethod:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    def _preprocess(
            self, 
            X1: np.ndarray, 
            X2: np.ndarray
            ):
        """
        Preprocess two views with missing samples to get reference and aligned matrices for each view.
        This is done by inferring presense mask from NaN rows, splitting into reference (both views)
        adn aligned (entire view measurements) such that the first n_anchors rows of aligned are the reference samples, 
        and the rest are the view-specific samples.

        """
        mask1 = infer_presence_mask(X1)
        mask2 = infer_presence_mask(X2)
        mask_both = mask1 & mask2
        masks = [mask1, mask2]
        
        # Get indices for reconstruction
        indices_both = np.where(mask_both)[0]
        indices_v1_only = np.where(mask1 & ~mask_both)[0]
        indices_v2_only = np.where(mask2 & ~mask_both)[0]
        
        # split to reference and aligned
        X1_ref = X1[mask_both]
        X1_aligned = np.concatenate(
            [X1_ref, X1[mask1 & ~mask_both]], axis=0)
        X2_ref = X2[mask_both]
        X2_aligned = np.concatenate(
            [X2_ref, X2[mask2 & ~mask_both]], axis=0)
        
        # Create index mappings to revert to original order
        # aligned_to_original[i] tells you which original index aligned row i corresponds to
        aligned_to_original_v1 = np.concatenate([indices_both, indices_v1_only])
        aligned_to_original_v2 = np.concatenate([indices_both, indices_v2_only])
        
        index_info = {
            'aligned_to_original_v1': aligned_to_original_v1,
            'aligned_to_original_v2': aligned_to_original_v2,
            'indices_both': indices_both,
            'indices_v1_only': indices_v1_only,
            'indices_v2_only': indices_v2_only
        }
        
        return X1_ref, X1_aligned, X2_ref, X2_aligned, masks, index_info


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

# ----------------------------- Naive Embed -----------------------------

class NaiveEmbed(EmbeddingMethod):
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    def fit(self, X1: np.ndarray, X2: np.ndarray) -> "NaiveEmbed":
        (X1_ref, X1_aligned, X2_ref,
          X2_aligned, masks, index_info) = self._preprocess(X1, X2)
        # get sizes
        N = X1.shape[0]
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
        mask_both = masks[0] & masks[1]
        # unify kernels
        Z = np.zeros((N, n_anchors), dtype=np.float64)
        Z[index_info['indices_v1_only'], :] = A1[n_anchors:, :]
        Z[index_info['indices_v2_only'], :] = A2[n_anchors:, :]
        Z[index_info['indices_both'], :] = 0.5*(A2[:n_anchors, :] + A1[:n_anchors, :])

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
        self.embedding_ = vecs[:, 1:self.cfg.embed_dim + 1]
        return self
    
    def get_embedding(self) -> np.ndarray:
        if self.embedding_ is None:
            raise RuntimeError("Call fit() first.")
        return self.embedding_
    

class RoselandEmbed(EmbeddingMethod):
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    def fit(self, X1: np.ndarray, X2: np.ndarray) -> "RoselandEmbed":
        (X1_ref, X1_aligned, X2_ref,
          X2_aligned, masks, index_info) = self._preprocess(X1, X2)
        # get sizes
        N = X1.shape[0]
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
        mask_both = masks[0] & masks[1]
        # unify kernels
        Z = np.zeros((N, n_anchors), dtype=np.float64)
        Z[index_info['indices_v1_only'], :] = A1[n_anchors:, :]
        Z[index_info['indices_v2_only'], :] = A2[n_anchors:, :]
        Z[index_info['indices_both'], :] = 0.5*(A2[:n_anchors, :] + A1[:n_anchors, :])

        # normalize Z
        # col_sum = np.array(Z.sum(axis=0)).flatten()
        col_sum = Z.T @ np.ones(N)  # Roseland normalization
        col_sum = Z @ col_sum
        # Handle division by zero in col_sum
        col_sum[col_sum == 0] = 1
        # Compute the inverse of column sums
        inv_sum = col_sum ** (-0.5)
        # remove inf values in inv_sum
        inv_sum[np.isinf(inv_sum)] = 1
        # Create diagonal matrix with inverse column sums
        norm_mat = sp.diags(inv_sum)
        # Normalize columns
        Z_norm = norm_mat @ Z

        vals, vecs = SVD_trick(Z_norm, self.cfg.embed_dim)
        self.embedding_ = vecs[:, 1:self.cfg.embed_dim + 1]
        return self
    
    def get_embedding(self) -> np.ndarray:
        if self.embedding_ is None:
            raise RuntimeError("Call fit() first.")
        return self.embedding_

# ----------------------------- APMC -----------------------------


class APMC(EmbeddingMethod):
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    def _embed_from_kernels(
            self,
            A1: np.ndarray,
            A2: np.ndarray, 
            n_anchors: int,
            N: int,
            index_info: Dict[str, np.ndarray]
    ) -> "APMC":
        
        # compute the embedding for each mode
        Q1 = row_norm(A1)
        Q2 = row_norm(A2)
        # unify kernels
        Z = np.zeros((N, n_anchors), dtype=np.float64)
        Z[index_info['indices_v1_only'], :] = Q1[n_anchors:, :]
        Z[index_info['indices_v2_only'], :] = Q2[n_anchors:, :]
        Z[index_info['indices_both'], :] = 0.5*(Q2[:n_anchors, :] + Q1[:n_anchors, :])

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
    
    def fit_from_dist_mats(
            self, 
            D1: np.ndarray,
            D2: np.ndarray,
            X1: np.ndarray,
            X2: np.ndarray,
            ) -> "APMC":
        
        (X1_ref, X1_aligned, X2_ref,
          X2_aligned, masks, index_info) = self._preprocess(X1, X2)
        # get sizes
        N = X1.shape[0]
        n_anchors = X1_ref.shape[0] 

        # mask values from distance matrices
        mask_both = masks[0] & masks[1]
        D1_aligned = D1[index_info['aligned_to_original_v1'], :][:, index_info['indices_both']]
        D2_aligned = D2[index_info['aligned_to_original_v2'], :][:, index_info['indices_both']]

        # kernels from distance matrices
        A1 = dist2kernel(D1_aligned, scale=self.cfg.kernel_scale1, zero_diag=False, k=None)
        A2 = dist2kernel(D2_aligned, scale=self.cfg.kernel_scale2, zero_diag=False, k=None)

        self.embedding_ = self._embed_from_kernels(A1, A2, n_anchors, N, index_info)
        return self

    
    def fit(self, X1: np.ndarray, X2: np.ndarray) -> "APMC":
        (X1_ref, X1_aligned, X2_ref,
          X2_aligned, masks, index_info) = self._preprocess(X1, X2)
        # get sizes
        N = X1.shape[0]
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
        
        self.embedding_ = self._embed_from_kernels(A1, A2, n_anchors, N, index_info)
        return self
    
    
    def get_embedding(self) -> np.ndarray:
        if self.embedding_ is None:
            raise RuntimeError("Call fit() first.")
        return self.embedding_
# ----------------------------- Config -----------------------------

@dataclass
class ADMPlusConfig(EmbeddingConfig):
    t: int = 0.1
    kernel_scale1: float = 1.0
    kernel_scale2: float = 1.0
    scale_mode: str = "median"  # "median" or "sigma"
    fusion_scale: float = 1.0
    fusion_method: str = "apmc"  # "apmc" or "roseland" or "naive"
    embed_dim: int = 30  

# ----------------------------- Model -----------------------------

class ADM_PLUS(EmbeddingMethod):
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


    def _embed_from_kernels(
            self,
            A1: np.ndarray,
            A2: np.ndarray, 
            n_anchors: int,
            N: int,
            index_info: Dict[str, np.ndarray]
    ) -> "ADM_PLUS":
        
        # compute the embedding for each mode
        embed1_aligned = adm_plus(embed_dim=self.cfg.embed_dim,
                          t=self.cfg.t, A1=A1, 
                          K2_ref=A2[:n_anchors, :],
                          return_vecs=False)
        embed2_aligned = adm_plus(embed_dim=self.cfg.embed_dim,
                          t=self.cfg.t, A1=A2,
                          K2_ref=A1[:n_anchors, :],
                          return_vecs=False)
        
        # Map embeddings back to original order using index_info
        embed1_full = np.full((N, self.cfg.embed_dim), np.nan, dtype=np.float64)
        embed2_full = np.full((N, self.cfg.embed_dim), np.nan, dtype=np.float64)
        
        # Use the index mappings to restore original order
        embed1_full[index_info['aligned_to_original_v1'], :] = embed1_aligned
        embed2_full[index_info['aligned_to_original_v2'], :] = embed2_aligned
        
        # combine the two embeddings with APMC
        apmc_config = EmbeddingConfig(
            kernel_scale1=self.cfg.fusion_scale, 
            kernel_scale2=self.cfg.fusion_scale, 
            scale_mode="median", 
            embed_dim=self.cfg.embed_dim
        )
        if self.cfg.fusion_method == "apmc":
            apmc = APMC(apmc_config)
            return apmc.fit(embed1_full, embed2_full).get_embedding()
        elif self.cfg.fusion_method == "roseland":
            roseland = RoselandEmbed(apmc_config)
            return roseland.fit(embed1_full, embed2_full).get_embedding()
        elif self.cfg.fusion_method == "naive":
            naive = NaiveEmbed(apmc_config)
            return naive.fit(embed1_full, embed2_full).get_embedding()
        else:
            raise ValueError(f"Unknown fusion method: {self.cfg.fusion_method}")
    
    def fit(
            self, 
            X1: np.ndarray,
            X2: np.ndarray
            ) -> "ADM_PLUS":

        (X1_ref, X1_aligned, X2_ref,
          X2_aligned, masks, index_info) = self._preprocess(X1, X2)
        # get sizes
        N = X1.shape[0]
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
        
        self.embedding_ = self._embed_from_kernels(A1, A2, n_anchors, N, index_info)
        return self

    def fit_from_dist_mats(
            self, 
            D1: np.ndarray,
            D2: np.ndarray,
            X1: np.ndarray,
            X2: np.ndarray,
            ) -> "ADM_PLUS":
        
        (X1_ref, X1_aligned, X2_ref,
          X2_aligned, masks, index_info) = self._preprocess(X1, X2)
        # get sizes
        N = X1.shape[0]
        n_anchors = X1_ref.shape[0] 

        # select valid values from distance matrices
        D1_aligned = D1[index_info['aligned_to_original_v1'], :][:, index_info['indices_both']]
        D2_aligned = D2[index_info['aligned_to_original_v2'], :][:, index_info['indices_both']]

        # kernels from distance matrices
        A1 = dist2kernel(D1_aligned, scale=self.cfg.kernel_scale1, zero_diag=False, k=None)
        A2 = dist2kernel(D2_aligned, scale=self.cfg.kernel_scale2, zero_diag=False, k=None)

        self.embedding_ = self._embed_from_kernels(A1, A2, n_anchors, N, index_info)
        
        return self


    def get_embedding(self) -> np.ndarray:
        if self.embedding_ is None:
            raise RuntimeError("Call fit() first.")
        return self.embedding_
