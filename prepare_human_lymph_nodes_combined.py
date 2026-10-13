#!/usr/bin/env python3
"""
将 Human lymph nodes 的两张切片 RNA + ADT 合并为一个统一 .h5ad 文件。

输入目录结构：
data/Human_lymph_nodes/
├── slice1/
│   ├── s1_adata_rna.h5ad
│   └── s1_adata_adt.h5ad
└── slice2/
    ├── s2_adata_rna.h5ad
    └── s2_adata_adt.h5ad

输出统一格式：
adata.X
    RNA 表达矩阵，所有 slice1 + slice2 细胞纵向合并

adata.obsm["protein_expression"]
    31 个 ADT 蛋白表达，DataFrame，保留蛋白名称

adata.obsm["spatial"]
    空间坐标。注意两张切片坐标各自属于独立坐标系，
    不应把两张切片当作同一连续组织空间。

adata.obs["slice_id"]
    "slice1" 或 "slice2"

adata.obs["split"]
    默认 slice1="train"，slice2="test"

运行示例：
python prepare_human_lymph_nodes_combined.py \
  --input_dir data/Human_lymph_nodes \
  --output_file data_processed/Human_lymph_nodes/Human_lymph_nodes_combined.h5ad
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SLICE_CONFIG = {
    "slice1": {
        "rna": "slice1/s1_adata_rna.h5ad",
        "adt": "slice1/s1_adata_adt.h5ad",
        "split": "train",
    },
    "slice2": {
        "rna": "slice2/s2_adata_rna.h5ad",
        "adt": "slice2/s2_adata_adt.h5ad",
        "split": "test",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Human lymph node RNA and ADT data into one h5ad."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出文件。",
    )
    return parser.parse_args()


def dense_or_sparse_float32(x):
    if sp.issparse(x):
        return x.astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def read_and_merge_single_slice(
    input_dir: Path,
    slice_id: str,
    config: dict,
) -> ad.AnnData:
    rna_path = input_dir / config["rna"]
    adt_path = input_dir / config["adt"]

    if not rna_path.exists():
        raise FileNotFoundError(f"缺少 RNA 文件：{rna_path}")
    if not adt_path.exists():
        raise FileNotFoundError(f"缺少 ADT 文件：{adt_path}")

    rna = ad.read_h5ad(rna_path)
    adt = ad.read_h5ad(adt_path)

    if rna.obs_names.has_duplicates:
        raise ValueError(f"{slice_id} RNA obs_names 存在重复。")
    if adt.obs_names.has_duplicates:
        raise ValueError(f"{slice_id} ADT obs_names 存在重复。")
    if rna.var_names.has_duplicates:
        raise ValueError(f"{slice_id} RNA var_names 存在重复。")
    if adt.var_names.has_duplicates:
        raise ValueError(f"{slice_id} ADT var_names 存在重复。")

    common = rna.obs_names.intersection(adt.obs_names)

    if len(common) != rna.n_obs or len(common) != adt.n_obs:
        raise ValueError(
            f"{slice_id} RNA 和 ADT 细胞不能完全对应："
            f"RNA={rna.n_obs}, ADT={adt.n_obs}, common={len(common)}"
        )

    # 即使当前顺序一致，也始终显式按 RNA 顺序重排 ADT，避免潜在错位。
    adt = adt[rna.obs_names].copy()

    if not rna.obs_names.equals(adt.obs_names):
        raise RuntimeError(f"{slice_id} 重排后 RNA 与 ADT 细胞顺序仍不一致。")

    if "spatial" not in rna.obsm:
        raise KeyError(f"{slice_id} RNA 文件缺少 obsm['spatial']。")

    spatial = np.asarray(rna.obsm["spatial"], dtype=np.float32)
    if spatial.shape != (rna.n_obs, 2):
        raise ValueError(
            f"{slice_id} spatial shape={spatial.shape}，"
            f"期望 ({rna.n_obs}, 2)。"
        )

    protein = adt.X
    if sp.issparse(protein):
        protein = protein.toarray()
    protein = np.asarray(protein, dtype=np.float32)

    if protein.shape != (rna.n_obs, adt.n_vars):
        raise ValueError(
            f"{slice_id} ADT shape 异常：{protein.shape}"
        )

    result = rna.copy()
    result.X = dense_or_sparse_float32(result.X)
    result.obsm["spatial"] = spatial

    protein_df = pd.DataFrame(
        protein,
        index=result.obs_names.copy(),
        columns=adt.var_names.astype(str).tolist(),
    )
    result.obsm["protein_expression"] = protein_df

    result.obs["slice_id"] = slice_id
    result.obs["split"] = config["split"]
    result.obs["dataset_name"] = "Human_lymph_nodes"
    result.obs["x"] = spatial[:, 0]
    result.obs["y"] = spatial[:, 1]

    result.uns["protein_names"] = adt.var_names.astype(str).tolist()
    result.uns["has_protein"] = True
    result.uns["slice_id"] = slice_id
    result.uns["split"] = config["split"]

    return result


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出文件已存在：{output_file}\n"
            "使用 --overwrite 允许覆盖。"
        )

    slices = {
        slice_id: read_and_merge_single_slice(
            input_dir=input_dir,
            slice_id=slice_id,
            config=config,
        )
        for slice_id, config in SLICE_CONFIG.items()
    }

    # 两张切片应具有完全一致的 RNA 基因和 ADT 蛋白。
    if not slices["slice1"].var_names.equals(slices["slice2"].var_names):
        common_genes = slices["slice1"].var_names.intersection(
            slices["slice2"].var_names
        )
        if len(common_genes) == 0:
            raise ValueError("slice1 和 slice2 没有共同 RNA 基因。")

        print(
            "警告：两张切片基因集合或顺序不一致，"
            f"将保留 {len(common_genes)} 个共同基因。"
        )
        slices["slice1"] = slices["slice1"][:, common_genes].copy()
        slices["slice2"] = slices["slice2"][:, common_genes].copy()

    proteins1 = list(slices["slice1"].obsm["protein_expression"].columns)
    proteins2 = list(slices["slice2"].obsm["protein_expression"].columns)
    if proteins1 != proteins2:
        raise ValueError(
            "slice1 和 slice2 的 ADT 蛋白名称或顺序不一致。"
        )

    # label 参数会再次创建 slice_id，因此这里使用自定义 key 避免覆盖。
    combined = ad.concat(
        [slices["slice1"], slices["slice2"]],
        axis=0,
        join="inner",
        merge="same",
        uns_merge="same",
        index_unique=None,
    )

    # 确保蛋白 DataFrame 在 concat 后仍然保留列名。
    combined.obsm["protein_expression"] = pd.concat(
        [
            slices["slice1"].obsm["protein_expression"],
            slices["slice2"].obsm["protein_expression"],
        ],
        axis=0,
    ).loc[combined.obs_names]

    combined.obsm["spatial"] = np.vstack(
        [
            np.asarray(slices["slice1"].obsm["spatial"], dtype=np.float32),
            np.asarray(slices["slice2"].obsm["spatial"], dtype=np.float32),
        ]
    )

    combined.uns["dataset_name"] = "Human_lymph_nodes"
    combined.uns["task"] = "RNA_to_protein"
    combined.uns["source_modality"] = "RNA"
    combined.uns["target_modality"] = "protein"
    combined.uns["protein_names"] = proteins1
    combined.uns["has_protein"] = True
    combined.uns["schema_version"] = "rna_protein_spatial_v1"
    combined.uns["slice_roles"] = {
        "slice1": "train",
        "slice2": "test",
    }
    combined.uns["important_note"] = (
        "slice1 and slice2 have independent spatial coordinate systems. "
        "Build spatial neighbors separately within each slice_id."
    )
    combined.uns["preprocessing"] = {
        "rna": "preserved from *_adata_rna.h5ad; cast to float32",
        "protein": "copied from *_adata_adt.h5ad into obsm['protein_expression']",
        "spatial": "preserved from RNA files",
        "normalization": "not performed",
        "hvg_selection": "not performed",
        "cell_alignment": "ADT reordered explicitly to RNA obs_names",
    }

    if combined.obs_names.has_duplicates:
        raise ValueError(
            "合并后 obs_names 出现重复。"
            "当前文件通常以 s1-/s2- 为前缀，理论上不应重复。"
        )

    if combined.obsm["protein_expression"].shape[0] != combined.n_obs:
        raise RuntimeError("合并后蛋白矩阵与细胞数不一致。")

    if combined.obsm["spatial"].shape != (combined.n_obs, 2):
        raise RuntimeError("合并后空间坐标与细胞数不一致。")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(output_file)

    # 重新读取做最终验证。
    check = ad.read_h5ad(output_file)

    report = {
        "output_file": str(output_file),
        "combined_shape": list(check.shape),
        "n_cells": int(check.n_obs),
        "n_genes": int(check.n_vars),
        "protein_shape": list(check.obsm["protein_expression"].shape),
        "spatial_shape": list(check.obsm["spatial"].shape),
        "protein_names": list(check.obsm["protein_expression"].columns),
        "slice_counts": {
            key: int(value)
            for key, value in check.obs["slice_id"].value_counts().to_dict().items()
        },
        "split_counts": {
            key: int(value)
            for key, value in check.obs["split"].value_counts().to_dict().items()
        },
    }

    report_path = output_file.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n处理完成")
    print("输出文件：", output_file)
    print("报告文件：", report_path)
    print("Combined shape:", check.shape)
    print("Protein shape:", check.obsm["protein_expression"].shape)
    print("Spatial shape:", check.obsm["spatial"].shape)
    print("Slice counts:")
    print(check.obs["slice_id"].value_counts())
    print("Split counts:")
    print(check.obs["split"].value_counts())
    print("Protein names:")
    print(list(check.obsm["protein_expression"].columns))


if __name__ == "__main__":
    main()
