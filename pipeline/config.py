"""Unified configuration for scRAG-Flow pipeline stages."""
from dataclasses import dataclass, field


@dataclass
class Stage1Config:
    # Architecture
    hidden_dim: int = 256
    latent_dim: int = 64
    proj_dim: int = 32
    depth: int = 2
    dropout: float = 0.1

    # Loss weights
    alpha: float = 20.0
    beta: float = 0.5
    gamma: float = 0.5

    # Training
    epochs: int = 20
    lr: float = 4e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024

    # Gene selection
    n_top_genes: int = 3000
    use_per_protein_hvg: bool = True
    n_per_protein: int = 100
    use_protein_guided_hvg: bool = False

    # Data split
    train_ratio: float = 0.8
    seed: int = 42
    split_seed: int = 42


@dataclass
class Stage3Config:
    # Architecture
    model: str = "dit"       # "film" | "film_gate"
    hidden_dim: int = 512
    depth: int = 4
    dropout: float = 0.1
    use_adaln_zero: bool = True    # True=AdaLN-Zero, False=Standard FiLM

    # Training
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024
    finetune_stage1: bool = False

    # Retrieval
    k_retrieval: int = 150
    val_ratio: float = 0.125        # fraction of total cells for validation

    prediction_nfe: int = 8

    # Input augmentation
    mask_prob: float = 0.0
    retrieval_noise: float = 0.0

    # Evaluation
    eval_every: int = 2
    seed: int = 42


# Preset configs for each dataset
def get_eccite_config():
    s1 = Stage1Config(n_top_genes=3000, use_per_protein_hvg=True, n_per_protein=100)
    s3 = Stage3Config(use_adaln_zero=True, mask_prob=0.0, model="dit")
    return s1, s3


def get_gse3p_config():
    s1 = Stage1Config(n_top_genes=1000, use_per_protein_hvg=True, n_per_protein=100)
    s3 = Stage3Config(use_adaln_zero=True, mask_prob=0.0, model="dit")
    return s1, s3


def get_cbmc_config():
    s1 = Stage1Config(n_top_genes=1000, use_per_protein_hvg=True,
                       n_per_protein=100, use_protein_guided_hvg=False)
    s3 = Stage3Config(use_adaln_zero=False, mask_prob=0.2, model="dit")
    return s1, s3


@dataclass
class SpatialConfig:
    """Per-dataset spatial graph + slice-split configuration (NicheRAGFlow).

    Grid datasets use a two-ring radius graph (near = first ring, far = second
    ring); single-cell datasets use KNN with distance-ordered halves. Relative
    coordinates are computed per slice-unit so that train/test halves and
    independent slices share a common frame.
    """
    # Split scheme
    split_mode: str = "cross_slice"   # "cross_slice" | "leave_one_slice_out" | "half_plane"
    split_axis: int = 0               # coordinate axis used by half_plane split

    # Neighbor graph
    graph_mode: str = "radius"        # "radius" | "knn"
    k: int = 12                       # KNN neighbor count (single-cell)
    radius_near: float = 1.0          # first-ring radius (grid)
    radius_far: float = 2.0           # second-ring radius (grid)
    n_near: int = 4                   # expected near-neighbor count (padding target)
    n_far: int = 4                    # expected far-neighbor count (padding target)

    # Coordinate source
    coord_source: str = "spatial"     # "spatial" (obsm) | "array_row_col" (obs)

    # Relative-coordinate handling
    center_per_slice: bool = True     # subtract per-slice centroid
    normalize_displacement: bool = True

    # Niche context (Stage-1 spatial fusion)
    spatial_temperature: float = 2.0  # sharpness of neighbor weighting in niche RNA


def get_breast_spatial_config() -> SpatialConfig:
    # Single-cell Xenium (continuous coords), one slice -> cut in half along x.
    return SpatialConfig(
        split_mode="half_plane", split_axis=0,
        graph_mode="knn", k=12, n_near=6, n_far=6,
        coord_source="spatial",
    )


def get_lymph_spatial_config() -> SpatialConfig:
    # Grid (checkerboard, NN=sqrt2), two slices -> slice1 train / slice2 test.
    return SpatialConfig(
        split_mode="cross_slice",
        graph_mode="radius", radius_near=2 ** 0.5, radius_far=2.0, n_near=4, n_far=4,
        coord_source="spatial",
    )


def get_postnatal_spatial_config() -> SpatialConfig:
    # Visium grid (square, NN=1), three slices -> leave-one-slice-out.
    return SpatialConfig(
        split_mode="leave_one_slice_out",
        graph_mode="radius", radius_near=1.0, radius_far=2 ** 0.5, n_near=4, n_far=4,
        coord_source="array_row_col",
    )


def get_tonsil_spatial_config() -> SpatialConfig:
    # Visium grid (checkerboard, NN=√2), single slice split into train/test.
    return SpatialConfig(
        split_mode="cross_slice",
        graph_mode="radius", radius_near=2 ** 0.5, radius_far=2.0, n_near=4, n_far=4,
        coord_source="spatial",
    )


def get_example_spatial_config() -> SpatialConfig:
    # Synthetic two-slice grid demo (see examples/make_example_data.py).
    return SpatialConfig(
        split_mode="cross_slice",
        graph_mode="radius", radius_near=2 ** 0.5, radius_far=2.0, n_near=4, n_far=4,
        coord_source="spatial",
    )


def get_postnatal_p21_spatial_config() -> SpatialConfig:
    # Visium grid (square, NN=1), two slices (P21S1/P21S3) -> leave-one-slice-out.
    return SpatialConfig(
        split_mode="leave_one_slice_out",
        graph_mode="radius", radius_near=1.0, radius_far=2 ** 0.5, n_near=4, n_far=4,
        coord_source="array_row_col",
    )


def get_spatial_config(dataset: str) -> SpatialConfig:
    if dataset == "breast":
        return get_breast_spatial_config()
    if dataset == "lymph":
        return get_lymph_spatial_config()
    if dataset == "tonsil":
        return get_tonsil_spatial_config()
    if dataset == "example":
        return get_example_spatial_config()
    if dataset == "postnatal_p21":
        return get_postnatal_p21_spatial_config()
    return get_postnatal_spatial_config()
