"""Schemas and constants used by forms and views."""

# predictor/schemas.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Keep any UI choice lists here too (no heavy deps).
HRT_CHOICES = [(0, "Never/Unknown"), (1, "Former"), (2, "Current")]
SMOKE_CHOICES = [(0, "Never"), (1, "Former"), (2, "Current")]
MAG_CHOICES = [(224, "224×224")]

@dataclass(frozen=True)
class RiskFactors:
    age: float
    first_degree_relative: int
    onset_age_relative: Optional[float] = None
    brca1: int = 0
    brca2: int = 0
    menarche_age: Optional[float] = None
    menopause_age: Optional[float] = None
    parity: Optional[float] = None
    hrt: int = 0
    bmi: Optional[float] = None
    alcohol_units_per_week: Optional[float] = None
    smoking_status: int = 0
    activity_hours_per_week: Optional[float] = None

