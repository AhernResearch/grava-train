"""Explicit names for the supported ms-swift reward experiments."""

from swift.rewards import orms

from .driving import (
    BoundaryMarginReward,
    BoundaryReward,
    CDSReward,
    ComfortReward,
    FDEProgressReward,
    GatedProgressReward,
    NCReward,
    PDMSFDEReward,
    PDMSMonitorReward,
    PDMSReward,
    ProgressReward,
    SafetyGateReward,
    SafetyReward,
    TTCReward,
)
from .reasoning import GRAFormatReward, ModeAppropriatenessReward

REWARDS = {
    "progress": ProgressReward,
    "fde_progress": FDEProgressReward,
    "pdms_fde": PDMSFDEReward,
    "safety": SafetyReward,
    "comfort": ComfortReward,
    "ttc": TTCReward,
    "pdms": PDMSReward,
    "cds": CDSReward,
    "safety_gate": SafetyGateReward,
    "gated_progress": GatedProgressReward,
    "pdms_monitor": PDMSMonitorReward,
    "nc": NCReward,
    "boundary": BoundaryReward,
    "boundary_margin": BoundaryMarginReward,
    "chain_bonus": ModeAppropriatenessReward,
    "gra_format": GRAFormatReward,
}


def register_rewards() -> None:
    """Register the GRAVA reward names with ms-swift."""
    orms.update(REWARDS)
