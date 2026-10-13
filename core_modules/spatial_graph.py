"""Spatial graph construction and slice splitting for NicheRAGFlow.

Spatial information enters the model only through the *source* (RNA) modality:
for every cell we collect the RNA of its spatial neighbors, ordered near-then-
far, together with the neighbor's relative displacement (dx, dy) from the
center. The target (protein) is never retrieved from spatial neighbors.

The graph is always built *within* a slice-unit (natural slice, or a half-plane
pseudo-slice), never across the train/test boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class SpatialGraph:
    """Neighbor topology for a cell set.

    neighbor_indices  : (N, L) int64, ordered [near (n_near) | far (n_far)].
                        -1 marks padding (edge cells with too few neighbors).
    displacement      : (N, L, 2) float32, (dx, dy) relative to the center,
                        normalized by the per-slice nearest-neighbor scale.
    valid             : (N, L) bool, True for real neighbors.
    """
    neighbor_indices: np.ndarray
    displacement: np.ndarray
    valid: np.ndarray
    slice_units: np.ndarray

    @property
    def n_neighbors(self) -> int:
        return self.neighbor_indices.shape[1]


def extract_coordinates(adata, config) -> np.ndarray:
    """Return (N, 2) float64 coordinates from the configured source."""
    if config.coord_source == "array_row_col":
        coords = np.stack(
            [adata.obs["array_col"].to_numpy(), adata.obs["array_row"].to_numpy()],
            axis=1,
        ).astype(np.float64)
    else:  # "spatial"
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    return coords


def center_coordinates(coords: np.ndarray, slice_units: np.ndarray) -> np.ndarray:
    """Subtract each slice-unit's centroid so coordinates become relative."""
    out = coords.astype(np.float64).copy()
    for sid in np.unique(slice_units):
        m = slice_units == sid
        out[m] = out[m] - out[m].mean(axis=0, keepdims=True)
    return out


def make_slice_units(adata, config):
    """Return (slice_units, folds) where slice_units is (N,) and folds is a list
    of (train_idx, test_idx) integer index arrays.

    - half_plane: one slice cut along ``split_axis`` at its median -> two units.
    - cross_slice: natural slices; roles read from ``uns['slice_roles']`` if
      present, otherwise the first (sorted) slice is train and the rest test.
    - leave_one_slice_out: one fold per slice (each slice held out in turn).
    """
    if config.split_mode == "half_plane":
        coords = extract_coordinates(adata, config)
        axis_val = coords[:, config.split_axis]
        median = float(np.median(axis_val))
        units = np.where(axis_val < median, "sliceL", "sliceR").astype(object)
        train = np.where(axis_val < median)[0]
        test = np.where(axis_val >= median)[0]
        return units, [(train, test)]

    sid_key = "slice_id" if "slice_id" in adata.obs.columns else "slice"
    slice_ids = adata.obs[sid_key].astype(str).to_numpy()
    units = slice_ids.astype(object)

    if config.split_mode == "cross_slice":
        roles = dict(adata.uns.get("slice_roles", {}))
        unique = sorted(set(slice_ids))
        if roles:
            train_slices = [s for s in unique if roles.get(s) == "train"]
            test_slices = [s for s in unique if roles.get(s) == "test"]
        else:
            train_slices, test_slices = unique[:1], unique[1:]
        train = np.where(np.isin(slice_ids, train_slices))[0]
        test = np.where(np.isin(slice_ids, test_slices))[0]
        return units, [(train, test)]

    # leave_one_slice_out
    folds = []
    for sid in sorted(set(slice_ids)):
        test = np.where(slice_ids == sid)[0]
        train = np.where(slice_ids != sid)[0]
        folds.append((train, test))
    return units, folds


def _per_slice_nn_scale(coords: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Median nearest-neighbor distance within each slice-unit (the grid unit)."""
    scale = np.ones(len(coords), dtype=np.float64)
    for sid in np.unique(units):
        m = units == sid
        c = coords[m]
        if len(c) < 2:
            scale[m] = 1.0
            continue
        nbr = NearestNeighbors(n_neighbors=2).fit(c)
        d, _ = nbr.kneighbors(c)
        scale[m] = float(np.median(d[:, 1])) + 1e-8
    return scale


def build_spatial_graph(adata, config) -> SpatialGraph:
    """Build the per-slice-unit spatial graph and neighbor displacements."""
    coords = extract_coordinates(adata, config)
    units, _ = make_slice_units(adata, config)
    coords = center_coordinates(coords, units)

    n = adata.n_obs
    n_near, n_far = config.n_near, config.n_far
    L = n_near + n_far
    indices = np.full((n, L), -1, dtype=np.int64)
    displacement = np.zeros((n, L, 2), dtype=np.float32)
    valid = np.zeros((n, L), dtype=bool)

    for sid in np.unique(units):
        m = units == sid
        local = np.where(m)[0]
        c = coords[local]

        if config.graph_mode == "knn":
            nbr = NearestNeighbors(n_neighbors=min(config.k + 1, len(c))).fit(c)
            _, idx = nbr.kneighbors(c)
            idx = idx[:, 1:]  # drop self; remaining are distance-ordered
            for j in range(min(n_near, idx.shape[1])):
                indices[local, j] = local[idx[:, j]]
                valid[local, j] = True
            for j in range(min(n_far, idx.shape[1] - n_near)):
                src = n_near + j
                if src < idx.shape[1]:
                    indices[local, n_near + j] = local[idx[:, src]]
                    valid[local, n_near + j] = True
        else:  # radius two-ring
            near_nbr = NearestNeighbors(radius=config.radius_near).fit(c)
            near_d, near_i = near_nbr.radius_neighbors(c, return_distance=True)
            far_nbr = NearestNeighbors(radius=config.radius_far).fit(c)
            far_d, far_i = far_nbr.radius_neighbors(c, return_distance=True)
            for k, li in enumerate(local):
                ni = near_i[k][near_d[k] > 0]        # first ring, exclude self
                fi = far_i[k][far_d[k] > 0]          # within radius_far, exclude self
                fi = fi[~np.isin(fi, ni)]            # second ring = far minus near
                for j in range(min(n_near, len(ni))):
                    indices[li, j] = local[ni[j]]
                    valid[li, j] = True
                for j in range(min(n_far, len(fi))):
                    indices[li, n_near + j] = local[fi[j]]
                    valid[li, n_near + j] = True

        for slot in range(L):
            nb = indices[local, slot]
            ok = valid[local, slot]
            if ok.any():
                displacement[local[ok], slot] = (
                    coords[nb[ok]] - coords[local[ok]]
                ).astype(np.float32)

    if config.normalize_displacement:
        scale = _per_slice_nn_scale(coords, units)
        displacement = displacement / (scale[:, None, None] + 1e-8)

    return SpatialGraph(
        neighbor_indices=indices,
        displacement=displacement.astype(np.float32),
        valid=valid,
        slice_units=units,
    )


def gather_neighbor_features(features: np.ndarray, graph: SpatialGraph) -> np.ndarray:
    """Gather neighbor rows from a per-cell feature matrix.

    features: (N, D). Returns (N, L, D) with zero rows for padding.
    """
    N, L = graph.neighbor_indices.shape
    D = features.shape[1]
    out = np.zeros((N, L, D), dtype=np.float32)
    idx = graph.neighbor_indices
    for slot in range(L):
        ok = idx[:, slot] >= 0
        out[ok, slot] = features[idx[ok, slot]]
    return out


def compute_niche_context(
    features: np.ndarray,
    graph: SpatialGraph,
    spatial_temperature: float = 1.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-weighted spatial context with an optional split-aware mask.

    Returns (niche_features, disp_ctx):
      - niche_features: (N, D) distance-weighted mean of neighbor features.
        Weights are softmax(-||delta|| / spatial_temperature), zero on padding.
      - disp_ctx: (N, 5) local-geometry summary [mean_dx, std_dx, mean_dy,
        std_dy, mean_norm] computed over valid neighbors only.
    """
    neighbor = gather_neighbor_features(features, graph)  # (N, L, D)
    if valid_mask is None:
        effective_valid = graph.valid
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != graph.valid.shape:
            raise ValueError(f"valid_mask shape {valid_mask.shape} != graph.valid shape {graph.valid.shape}")
        effective_valid = graph.valid & valid_mask
    valid = effective_valid.astype(np.float32)
    disp = graph.displacement.astype(np.float32)  # (N, L, 2)

    # Zero masked neighbor content so unavailable cells cannot affect context.
    neighbor = neighbor * valid[..., None]
    disp = disp * valid[..., None]
    dist = np.linalg.norm(disp, axis=-1)  # (N, L)
    # finite large negative (not -inf) so rows with no valid neighbor don't
    # produce NaN from `-inf - (-inf)` after the max subtraction.
    logits = np.where(effective_valid, -dist / max(spatial_temperature, 1e-8), -1e9)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits) * valid
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    niche_features = np.einsum("nl,nld->nd", weights, neighbor).astype(np.float32)

    n_valid = valid.sum(axis=1) + 1e-8
    dx = disp[:, :, 0]
    dy = disp[:, :, 1]
    norm = dist
    mean_dx = (dx * valid).sum(axis=1) / n_valid
    mean_dy = (dy * valid).sum(axis=1) / n_valid
    mean_norm = (norm * valid).sum(axis=1) / n_valid
    std_dx = np.sqrt(((dx - mean_dx[:, None]) ** 2 * valid).sum(axis=1) / n_valid)
    std_dy = np.sqrt(((dy - mean_dy[:, None]) ** 2 * valid).sum(axis=1) / n_valid)
    disp_ctx = np.stack([mean_dx, std_dx, mean_dy, std_dy, mean_norm], axis=1).astype(np.float32)

    return niche_features, disp_ctx
