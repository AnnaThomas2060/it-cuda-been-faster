"""Isolated GPU compilation, correctness verification and latency profiling.

Owner: Pramod Goyal. Importing this package does not import torch or touch
the GPU; only the child worker process does.
"""

from src.verifier.config import VerifierConfig
from src.verifier.harness import KernelVerifier

__all__ = ["KernelVerifier", "VerifierConfig"]
