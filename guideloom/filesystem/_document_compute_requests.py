from collections.abc import Sequence

from numba import from_dtype, int32, int64, njit, types
import numpy as np

from ..types import Vector

document_req_dtype = np.dtype(
    [
        ("shard", np.int64),
        ("shard_index", np.int32),
    ],
    align=True,
)

index_req_dtype = np.dtype(
    [
        ("shard", np.int64),
        ("index", np.int32),
        ("length", np.int32),
    ],
    align=True,
)

index_data_dtype = np.dtype("<i8")

data_req_dtype = np.dtype([("shard", np.int64), ("offset", np.int64), ("length", np.int64)], align=True)

unpack_info_dtype = np.dtype([("request", np.int64), ("start", np.int64), ("end", np.int64)], align=True)


def concat_fast[DType: np.generic](arrays: Sequence[Vector[DType]], dtype: np.dtype[DType]) -> Vector[DType]:
    # concat stream data with fast path
    match len(arrays):
        case 1:
            stream_data = arrays[0]
        case 0:
            stream_data = np.empty((0,), dtype=dtype)
        case _:
            stream_data = np.concatenate(arrays, axis=0)
    return stream_data


@njit((types.Array(from_dtype(document_req_dtype), 1, "C"),), nogil=True)
def validate_document_requests(requests: Vector[np.void]) -> int:
    if np.any(requests["shard"] < 0):
        return 1
    if np.any(requests["shard_index"] < 0):
        return 2
    return 0


@njit((types.Array(from_dtype(document_req_dtype), 1, "C"),), nogil=True)
def compute_index_requests(requests: Vector[np.void]) -> Vector[np.void]:
    index_reqs = np.empty(requests.size, dtype=index_req_dtype)

    # compact index requests
    curr_shard = requests[0]["shard"]
    start_index = requests[0]["shard_index"]
    prev_index = start_index
    read_length = int32(2)
    req_index = int64(0)
    for request in requests[int64(1) :]:
        if request["shard"] == curr_shard and (prev_index + int32(1)) == request["shard_index"]:
            read_length += int32(1)  # type: ignore
        else:
            index_reqs[req_index]["shard"] = curr_shard  # type: ignore
            index_reqs[req_index]["index"] = start_index  # type: ignore
            index_reqs[req_index]["length"] = read_length  # type: ignore

            curr_shard = request["shard"]
            start_index = request["shard_index"]
            read_length = int32(2)  # type: ignore
            req_index += int64(1)  # type: ignore

        prev_index = request["shard_index"]

    index_reqs[req_index]["shard"] = curr_shard  # type: ignore
    index_reqs[req_index]["index"] = start_index  # type: ignore
    index_reqs[req_index]["length"] = read_length  # type: ignore
    req_index += int64(1)  # type: ignore

    return index_reqs[:req_index]  # type: ignore


@njit(
    (types.Array(from_dtype(index_req_dtype), 1, "C"), types.Array(from_dtype(index_data_dtype), 1, "C")),
    nogil=True,
)
def compute_data_requests(
    index_reqs: Vector[np.void], index_data: Vector[np.int64]
) -> tuple[Vector[np.void], Vector[np.void]]:
    data_reqs = np.empty(index_reqs.size, dtype=data_req_dtype)
    unpack_info = np.empty(index_data.size - index_reqs.size, dtype=unpack_info_dtype)  # truncate

    index_data_start = int64(0)
    unpack_index_offset = int64(0)
    for req_index, index_req in enumerate(index_reqs):
        first_offset = index_data[index_data_start]  # type: ignore
        for unpack_index in range(index_req["length"] - int64(1)):  # type: ignore
            unpack_info[unpack_index_offset + unpack_index]["request"] = req_index  # type: ignore
            unpack_info[unpack_index_offset + unpack_index]["start"] = (
                index_data[index_data_start + unpack_index] - first_offset
            )  # type: ignore
            unpack_info[unpack_index_offset + unpack_index]["end"] = (
                index_data[index_data_start + unpack_index + 1] - first_offset
            )  # type: ignore

        data_reqs[req_index]["shard"] = index_req["shard"]  # type: ignore
        data_reqs[req_index]["offset"] = first_offset  # type: ignore
        data_reqs[req_index]["length"] = index_data[index_data_start + index_req["length"] - 1] - first_offset  # type: ignore
        index_data_start += index_req["length"]
        unpack_index_offset += index_req["length"] - int64(1)

    return (data_reqs, unpack_info)  # type: ignore
