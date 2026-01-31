"""
Phase 1 - Geography Structure Learning Model

This module implements the complete Phase 1 architecture:
- Dual encoder (CLIP + Scene) with gated fusion
- Province head
- Province-gated geocell heads (one per province)
- Cell & Province embeddings
- Cell-aware offset head
- Auxiliary coarse regression head
"""

from .geopak_phase1 import GeopakPhase1Model

__all__ = ['GeopakPhase1Model']
