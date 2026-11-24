"""Detect candidates that copy a code example shown in their own prompt.

Scored-context (Arm A) prompts contain working teacher kernels. Without this
gate a policy can earn the teacher's speedup by pasting the example back with
renamed variables, which measures copying rather than optimization. Only the
"Historical Implementations" examples count: in multi-turn prompts the
trajectory's own previous best ("Current best implementation") may be
resubmitted, identically for both arms.
"""

import difflib
import re
from typing import List

_EXAMPLE_SECTION = re.compile(
    r"Historical Implementations.*?(?=Analysis and Synthesis Instructions|"
    r"Multi-turn refinement state|\Z)",
    re.DOTALL,
)
_CODE_BLOCK = re.compile(r"```(?:cpp|cuda|c\+\+)?\s*(.*?)```", re.DOTALL)
_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?[fF]?|[^\w\s]")
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def prompt_examples(prompt_text: str) -> List[str]:
    """Code blocks from the prompt's example section (not the model's own code)."""
    section = _EXAMPLE_SECTION.search(prompt_text or "")
    if not section:
        return []
    return [block.strip() for block in _CODE_BLOCK.findall(section.group(0)) if block.strip()]


def _tokens(code: str) -> List[str]:
    return _TOKEN.findall(_COMMENT.sub(" ", code))


def token_similarity(a: str, b: str) -> float:
    """Ratio in [0, 1] over code tokens; insensitive to whitespace and comments."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


def max_example_similarity(code: str, prompt_text: str) -> float:
    """Highest similarity between `code` and any example in `prompt_text`."""
    return max((token_similarity(code, ex) for ex in prompt_examples(prompt_text)), default=0.0)
