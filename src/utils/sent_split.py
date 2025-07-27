import re
from typing import List

import nltk
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize

def greedy_sent_split(text: str, max_chunk_size: int) -> List[str]:
    """Greedily split *text* into chunks no longer than *max_chunk_size* characters.

    The algorithm proceeds in three passes:
    1. Split the text into sentences (using ``nltk.sent_tokenize``) and greedily
       joins them while the resulting chunk stays within the limit.
    2. Any chunk that still exceeds the limit is further split by commas or
       semicolons and greedily regrouped.
    3. Remaining over‑long chunks are finally split on whitespace to guarantee
       the size constraint.

    This strategy minimises the total number of chunks while ensuring the
    constraint is met.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [text]

    def _greedy_pack(tokens: List[str]) -> List[str]:
        """Pack tokens into chunks not exceeding *max_chunk_size*."""
        chunks: List[str] = []
        current = ""
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            prospective_len = len(current) + (1 if current else 0) + len(token)
            if prospective_len <= max_chunk_size:
                current = f"{current} {token}".strip()
            else:
                if current:
                    chunks.append(current)
                # Token itself may be longer than limit – handled by caller
                current = token if len(token) <= max_chunk_size else ""
                if len(token) > max_chunk_size:
                    # Split oversized token further by spaces immediately
                    chunks.extend(_greedy_pack(token.split()))
        if current:
            chunks.append(current)
        return chunks

    # Pass 1 – sentences
    chunks = _greedy_pack(sent_tokenize(text))

    # Pass 2 – commas / semicolons
    refined: List[str] = []
    for c in chunks:
        if len(c) <= max_chunk_size:
            refined.append(c)
        else:
            refined.extend(_greedy_pack(re.split(r",|;", c)))

    # Pass 3 – whitespace fallback
    final: List[str] = []
    for c in refined:
        if len(c) <= max_chunk_size:
            final.append(c)
        else:
            final.extend(_greedy_pack(c.split()))

    return [chunk.strip() for chunk in final if chunk.strip()]
