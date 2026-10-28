"""Generate compatibility fixtures and compare stored data against the same generators."""

from pathlib import Path
import sys

import numpy as np
import pytest

from guideloom.storage import DocumentStorage, StreamStorage, StreamWithMetaStorage

from .test_storage_document import ConceptStorageFormat as DocumentFormat
from .test_storage_document import InDocumentData
from .test_storage_stream import ConceptStorageFormat as StreamFormat
from .test_storage_stream import DocumentData as StreamData
from .test_storage_stream_with_meta import ConceptStorageFormat as MetaFormat
from .test_storage_stream_with_meta import DocumentData as MetaData


@pytest.fixture
def fixtures() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "compatibility"


def documents() -> list[tuple[int, InDocumentData]]:
    return [
        (shard, {"doc_id": shard * 4 + index, "data": f"Document {shard}-{index}"})
        for shard in range(3)
        for index in range(4)
    ]


def stream_documents() -> list[tuple[int, StreamData[np.int32]]]:
    rng = np.random.default_rng(0)
    return [
        (
            shard,
            {
                "doc_id": shard * 2 + index,
                "stream_data": rng.integers(0, 100, size=int(rng.integers(3, 10)), dtype=np.int32),
            },
        )
        for shard in range(3)
        for index in range(2)
    ]


def meta_documents() -> list[tuple[int, MetaData[np.int32]]]:
    rng = np.random.default_rng(1)
    return [
        (
            shard,
            {
                **document,
                "document_id": np.arange(len(document["stream_data"]), dtype=np.int32) // 2,
                "document_data": rng.integers(0, 100, size=(len(document["stream_data"]) + 1) // 2).tolist(),
            },
        )
        for shard, document in stream_documents()
    ]


def write_fixtures(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with DocumentStorage(output / "document", DocumentFormat()).sync_writer() as writer:
        for shard, document in documents():
            writer.append(shard, document)
    with StreamStorage(output / "stream", StreamFormat(np.dtype(np.int32))).sync_writer() as writer:
        for shard, document in stream_documents():
            writer.append(shard, document)
    with StreamWithMetaStorage(
        output / "stream_with_meta", MetaFormat(np.dtype(np.int32))
    ).sync_writer() as writer:
        for shard, document in meta_documents():
            writer.append(shard, document)


def test_document_compatibility(fixtures: Path) -> None:
    expected = documents()
    with DocumentStorage(fixtures / "document", DocumentFormat()).sync_reader() as reader:
        assert reader.num_shards == 3
        assert reader.total_length == len(expected)
        for index, (shard, document) in enumerate(expected):
            assert reader.read(index) == {**document, "shard": shard}


def test_stream_compatibility(fixtures: Path) -> None:
    expected = stream_documents()
    tokens = np.concat([document["stream_data"] for _, document in expected])
    with StreamStorage(fixtures / "stream", StreamFormat(np.dtype(np.int32))).sync_reader() as reader:
        assert reader.num_shards == 3
        assert reader.total_length == len(tokens)
        actual = reader.read(0, len(tokens))
        np.testing.assert_array_equal(actual["stream_data"], tokens, strict=True)
        np.testing.assert_array_equal(
            actual["doc_ids"],
            np.concat(
                [np.full(len(doc["stream_data"]), doc["doc_id"], dtype=np.int32) for _, doc in expected]
            ),
        )
        assert "shard_ids" in actual
        np.testing.assert_array_equal(
            actual["shard_ids"],
            np.concat([np.full(len(doc["stream_data"]), shard, dtype=np.int32) for shard, doc in expected]),
        )


def test_stream_with_meta_compatibility(fixtures: Path) -> None:
    expected = meta_documents()
    tokens = np.concat([document["stream_data"] for _, document in expected])
    with StreamWithMetaStorage(
        fixtures / "stream_with_meta", MetaFormat(np.dtype(np.int32))
    ).sync_reader() as reader:
        assert reader.num_shards == 3
        assert reader.total_length == len(tokens)
        actual = reader.read(0, len(tokens))
        np.testing.assert_array_equal(actual["stream_data"], tokens, strict=True)
        assert [actual["document_data"][index] for index in actual["document_id"].tolist()] == [
            {"doc_id": document["doc_id"], "value": document["document_data"][index]}
            for _, document in expected
            for index in document["document_id"].tolist()
        ]
        assert "shard" in actual
        np.testing.assert_array_equal(
            actual["shard"][actual["document_id"]],
            np.concat([np.full(len(doc["stream_data"]), shard, dtype=np.int64) for shard, doc in expected]),
        )


if __name__ == "__main__":
    write_fixtures(Path(sys.argv[1]))
