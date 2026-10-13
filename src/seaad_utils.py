"""Streaming extraction helpers for the large SEA-AD MTG H5AD object."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


SEAAD_CELL_TYPE_MAP = {
    "Astrocyte": "Astrocytes",
    "Endothelial": "Endothelial cells",
    "Immune": "Microglia",
    "OPC": "OPCs",
    "Oligodendrocyte": "Oligodendrocytes",
    "VLMC & Perivascular": "VLMC and perivascular cells",
}


def read_h5ad_dataframe_column(group: h5py.Group, column: str) -> np.ndarray:
    """Read a categorical or plain dataframe column from an H5AD group."""
    element = group[column]
    if isinstance(element, h5py.Group) and {"categories", "codes"}.issubset(element.keys()):
        categories = _decode_strings(element["categories"][:])
        codes = np.asarray(element["codes"][:], dtype=np.int64)
        values = np.full(codes.shape, None, dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values
    values = element[:]
    if values.dtype.kind in {"S", "O", "U"}:
        return _decode_strings(values)
    return values


def extract_h5ad_csr_columns(
    path: str | Path,
    group_path: str,
    requested_genes: list[str],
    rows_per_block: int = 2000,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Stream selected gene columns from an on-disk CSR matrix without loading all features."""
    path = Path(path)
    requested = list(dict.fromkeys(str(gene) for gene in requested_genes))
    with h5py.File(path, "r") as handle:
        var_group = handle["var"]
        index_name = _decode_attr(var_group.attrs.get("_index", "_index"))
        source_genes = _decode_strings(var_group[index_name][:]).astype(str)
        source_position = {gene: index for index, gene in enumerate(source_genes)}
        present = [gene for gene in requested if gene in source_position]
        source_columns = np.asarray([source_position[gene] for gene in present], dtype=np.int64)
        output_lookup = np.full(source_genes.shape[0], -1, dtype=np.int32)
        output_lookup[source_columns] = np.arange(len(present), dtype=np.int32)

        matrix_group = handle[group_path]
        shape = tuple(int(value) for value in matrix_group.attrs["shape"])
        indptr = np.asarray(matrix_group["indptr"][:], dtype=np.int64)
        indices_dataset = matrix_group["indices"]
        data_dataset = matrix_group["data"]
        blocks: list[sparse.csr_matrix] = []
        for row_start in range(0, shape[0], rows_per_block):
            row_stop = min(shape[0], row_start + rows_per_block)
            entry_start = int(indptr[row_start])
            entry_stop = int(indptr[row_stop])
            source_indices = np.asarray(indices_dataset[entry_start:entry_stop], dtype=np.int64)
            mapped = output_lookup[source_indices]
            selected_positions = np.flatnonzero(mapped >= 0)
            local_indptr = indptr[row_start : row_stop + 1] - entry_start
            selected_rows = np.searchsorted(local_indptr[1:], selected_positions, side="right")
            source_values = np.asarray(data_dataset[entry_start:entry_stop])
            block = sparse.coo_matrix(
                (
                    source_values[selected_positions].astype(np.float32, copy=False),
                    (selected_rows, mapped[selected_positions]),
                ),
                shape=(row_stop - row_start, len(present)),
                dtype=np.float32,
            ).tocsr()
            blocks.append(block)
    matrix = sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, len(present)))
    var = pd.DataFrame(
        {"source_feature_index": source_columns},
        index=pd.Index(present, name="gene_symbol"),
    )
    return matrix, var


def map_seaad_cell_types(classes: np.ndarray, subclasses: np.ndarray) -> np.ndarray:
    """Collapse author taxonomy to broad canonical populations without marker inference."""
    result = np.full(len(subclasses), "Unassigned", dtype=object)
    class_values = pd.Series(classes, dtype="string").fillna("")
    subclass_values = pd.Series(subclasses, dtype="string").fillna("")
    result[class_values.str.contains("Glutamatergic", case=False).to_numpy()] = "Excitatory neurons"
    result[class_values.str.contains("GABAergic", case=False).to_numpy()] = "Inhibitory neurons"
    for source, target in SEAAD_CELL_TYPE_MAP.items():
        result[subclass_values.eq(source).to_numpy()] = target
    return result


def _decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def _decode_attr(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
