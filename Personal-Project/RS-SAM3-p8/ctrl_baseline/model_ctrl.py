"""CTRL baseline: direct Plan7-A architecture (gate-only, no cross-attn/bias).

This is an alias for Plan7PromptMFNet from Plan7-A. The MMAdapter variant
is gate-only 3-way fusion (RGB/DSM/prompt) without any cross-attention or
attention bias mechanism.
"""

from __future__ import annotations

import sys

# Import Plan7PromptMFNet from Plan7-A
PHASE_DIR = "/root/Mynet/RS-SAM3-p7/phase_a_dsm_prompt"
if PHASE_DIR not in sys.path:
    sys.path.insert(0, PHASE_DIR)

from model import Plan7PromptMFNet  # noqa: E402, F401

__all__ = ["Plan7PromptMFNet"]
