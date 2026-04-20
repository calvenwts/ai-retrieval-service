"""Unit tests for the chunking logic."""

import pytest

from app.chunker import chunk_text


def test_empty_string():
    assert chunk_text("") == []


def test_short_text_single_chunk():
    text = "Hello world"
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_exact_size_boundary():
    text = "a" * 500
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) == 1


def test_overlap_produces_correct_chunks():
    text = "a" * 1000
    chunks = chunk_text(text, size=500, overlap=100)
    # First chunk: 0-500, second: 400-900, third: 800-1000
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500
    assert len(chunks[2]) == 200


def test_no_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, size=500, overlap=0)
    assert len(chunks) == 2
    assert all(len(c) == 500 for c in chunks)


def test_overlap_content_is_correct():
    text = "ABCDEFGHIJ"
    chunks = chunk_text(text, size=5, overlap=2)
    # chunks: "ABCDE", "DEFGH", "GHIJ"
    assert chunks[0] == "ABCDE"
    assert chunks[1] == "DEFGH"
    assert chunks[2] == "GHIJ"
