from collections.abc import Sequence

from numba import from_dtype, int32, int64, njit, types
import numpy as np

from ..types import Vector

stream_req_dtype = np.dtype(
    [("shard", np.int64), ("global_offset", np.int64), ("shard_offset", np.int64), ("length", np.int64)],
    align=True,
)


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


@njit((int64, int64, int64[:], int64[:]), nogil=True)
def compute_stream_requests(
    global_index: int, global_length: int, offset_index: Vector[np.int64], shard_length: Vector[np.int64]
) -> Vector[np.void]:
    # get the shard where the data is stored, and the index into the shard
    start_shard = np.searchsorted(offset_index, global_index, side="right") - 1
    start_index_in_shard = global_index - offset_index[start_shard]

    # count how many requests needs to be made
    num_of_requests = int64(0)
    read_length = int64(0)
    shard = start_shard
    index_in_shard = start_index_in_shard
    while read_length < global_length:  # type: ignore
        read_length += min(global_length - read_length, shard_length[shard] - index_in_shard)  # type: ignore
        shard = (shard + int64(1)) % shard_length.size  # type: ignore
        num_of_requests += int64(1)  # type: ignore
        index_in_shard = int64(0)

    # data accumulation across all shards
    stream_req_list: Vector[np.void] = np.empty(num_of_requests, dtype=stream_req_dtype)  # type: ignore

    # retrive stream index map
    read_length = int64(0)
    shard = start_shard
    index_in_shard = start_index_in_shard
    request_index = int64(0)
    while read_length < global_length:  # type: ignore
        # get read size for this data
        shard_read_length = min(global_length - read_length, shard_length[shard] - index_in_shard)  # type: ignore

        # write request
        stream_req_list[request_index]["shard"] = shard  # type: ignore
        stream_req_list[request_index]["global_offset"] = read_length  # type: ignore
        stream_req_list[request_index]["shard_offset"] = index_in_shard  # type: ignore
        stream_req_list[request_index]["length"] = shard_read_length  # type: ignore

        # move to next shard if required
        read_length += shard_read_length
        shard = (shard + int64(1)) % shard_length.size  # type: ignore
        index_in_shard = int64(0)
        request_index += int64(1)  # type: ignore

    return stream_req_list


document_req_dtype = np.dtype(
    [
        ("shard", np.int64),
        ("shard_index", np.int32),
    ],
    align=True,
)


@njit((int64, types.Array(from_dtype(stream_req_dtype), 1, "C"), int32[:]), nogil=True)
def compute_document_requests_and_index(
    global_length: int, stream_req_list: Vector[np.void], all_document_indexes: Vector[np.int32]
) -> tuple[Vector[np.void], Vector[np.int32]]:
    # data accumulation across all shards
    document_req = np.empty(global_length, dtype=document_req_dtype)
    document_local_indexes = np.empty(global_length, dtype=np.int32)

    # retrive document map
    curr_document_local_index = int32(-1)
    curr_document_index_offset = int32(0)
    local_document_i = int32(0)
    for stream_req in stream_req_list:
        prev_shard_document_global_index = int32(-1)
        for _ in range(stream_req["length"]):
            shard_document_index_global = all_document_indexes[curr_document_index_offset]  # type: ignore

            # detected new document part, so issue request
            if shard_document_index_global != prev_shard_document_global_index:
                curr_document_local_index += int32(1)  # type: ignore
                document_req[curr_document_local_index]["shard"] = stream_req["shard"]  # type: ignore
                document_req[curr_document_local_index]["shard_index"] = shard_document_index_global  # type: ignore

            document_local_indexes[local_document_i] = curr_document_local_index  # type: ignore
            local_document_i += int32(1)  # type: ignore
            curr_document_index_offset += int32(1)  # type: ignore
            prev_shard_document_global_index = shard_document_index_global  # type: ignore

    return (document_req[: (curr_document_local_index + 1)], document_local_indexes)  # type: ignore
