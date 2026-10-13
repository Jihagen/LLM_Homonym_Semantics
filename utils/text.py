"""Pure-Python text helpers with no torch/transformers dependency."""

import re
from typing import Optional, Tuple


def find_target_span(text: str, target: str) -> Optional[Tuple[int, int]]:
    """
    Return the (start, end) character span of the first occurrence of `target`
    (or its regular plural) in `text` that sits at a word boundary — i.e. not
    inside a longer, unrelated word.

    Example: target='bat', text='The batsman hit the bat.'
    → returns (20, 23) for the standalone 'bat', not (4, 7) inside 'batsman'.

    Example: target='match', text='She kept a box of matches.'
    → matches 'matches', since a bare plural is still the same word/sense —
    unlike derivations such as 'lighter' or 'Sunlight', which are excluded.

    Returns None if no boundary-respecting match is found.
    """
    pattern = r"(?<![a-zA-Z])" + re.escape(target.lower()) + r"(e?s)?(?![a-zA-Z])"
    m = re.search(pattern, text.lower())
    return (m.start(), m.end()) if m else None
