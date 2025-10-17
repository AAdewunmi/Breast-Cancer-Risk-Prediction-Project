"""Schemas and constants used by forms and views."""

from __future__ import annotations

from dataclasses import dataclass

# Keep any UI choice lists here too (no heavy deps).
MAG_CHOICES = [(40, "40x"), (100, "100x")]
HRT_CHOICES = [(0, "Never"), (1, "Former"), (2, "Current")]
SMOKE_CHOICES = [(0, "Never"), (1, "Former"), (2, "Current")]


@dataclass
class RiskFactors:
    age: float
    first_degree_relative: int
    onset_age_relative: float | None = None
    brca1: int = 0
    brca2: int = 0
    menarche_age: float | None = None
    menopause_age: float | None = None
    parity: float | None = None
    hrt: int = 0
    bmi: float | None = None
    alcohol_units_per_week: float | None = None
    smoking_status: int = 0
    activity_hours_per_week: float | None = None
