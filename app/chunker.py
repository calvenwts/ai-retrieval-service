"""Document chunking with configurable size and overlap."""

from app.config import settings


def chunk_text(
    text: str,
    size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[str]:
    """Split text into chunks of `size` characters with `overlap`."""
    if not text:
        return []
    chunks = []
    i = 0
    step = size - overlap
    while i < len(text):
        chunk = text[i : i + size]
        # Skip trailing slivers that are fully covered by the previous chunk's overlap
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(chunk)
        i += step
    return chunks
