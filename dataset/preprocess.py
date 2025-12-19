import numpy as np
import torch
import time

from evenet.dataset.types import Batch, Source, AssignmentTargets
import pyarrow as pa

def process_event_batch(batch: dict[str, np.ndarray], shape_metadata: dict, unflatten, drop_column_prefix: str = None) -> dict[str, np.ndarray]:
    return unflatten(batch, shape_metadata, drop_column_prefix=drop_column_prefix)


def convert_batch_to_torch_tensor(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """
    Convert a batch of data from numpy arrays to torch tensors.
    :param batch: Batch of data as a dictionary with numpy arrays.
    :return: Batch of data as a dictionary with torch tensors.
    """
    return {k: torch.tensor(v) for k, v in batch.items()}


def flatten_dict(data: dict, delimiter: str = ":"):
    flat_columns = {}
    shape_metadata = {}

    for key, arr in data.items():
        shape = arr.shape[1:]
        shape_metadata[key] = shape

        if arr.ndim == 1:
            flat_columns[key] = pa.array(arr)
        else:
            flat_arr = arr.reshape(arr.shape[0], -1)
            for i in range(flat_arr.shape[1]):
                suffix = np.unravel_index(i, shape)
                col_key = f"{key}{delimiter}" + delimiter.join(map(str, suffix))
                flat_columns[col_key] = pa.array(flat_arr[:, i])

    table = pa.table(flat_columns)
    return table, shape_metadata


def unflatten_dict(
        table: dict[str, np.ndarray],
        shape_metadata: dict,
        drop_column_prefix: list[str] = None,
        delimiter: str = ":",
):
    reconstructed = {}
    grouped = {}
    for col in table:
        base = col.split(delimiter)[0]
        grouped.setdefault(base, []).append(col)
    for base, columns in grouped.items():
        if drop_column_prefix is not None:
            if any(base.startswith(prefix) for prefix in drop_column_prefix):
                continue

        if base not in shape_metadata:
            reconstructed[base] = table[columns[0]]
        else:
            shape = tuple(shape_metadata[base])
            sorted_columns = sorted(columns, key=lambda x: tuple(map(int, x.split(delimiter)[1:])))
            flat = np.stack([table[col] for col in sorted_columns], axis=1)
            full_shape = (flat.shape[0],) + shape
            reconstructed[base] = flat.reshape(full_shape)
    # Print shapes
    # for k, v in reconstructed.items():
    #     print(f"{k}: {v.shape}")

    return reconstructed
