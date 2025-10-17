"""Django forms for image upload and risk-factor inputs."""

from __future__ import annotations
from typing import Any
from django import forms
from .schemas import MAG_CHOICES, HRT_CHOICES, SMOKE_CHOICES
from .services.preprocess import RiskFactors


class ImagePredictForm(forms.Form):
    """Form to collect an image and context for the image-based model."""
    image = forms.ImageField(required=True)
    magnification = forms.ChoiceField(choices=MAG_CHOICES, required=False)
    consent = forms.BooleanField(required=True, label="I understand this is a research prototype and not diagnostic.")

    def clean_image(self):
        """Validate MIME and size limits to reduce risk and server load."""
        f = self.cleaned_data["image"]
        if f.content_type not in ("image/png", "image/jpeg"):
            raise forms.ValidationError("Please upload a PNG or JPG image.")
        if f.size > 10 * 1024 * 1024:
            raise forms.ValidationError("Image size must be ≤ 10 MB.")
        return f


class RiskFactorsForm(forms.Form):
    """Form to collect structured risk-factors used by the tabular model."""
    age = forms.FloatField(min_value=0, max_value=120)
    first_degree_relative = forms.ChoiceField(choices=[(1, "Yes"), (0, "No")], initial=0)
    onset_age_relative = forms.FloatField(required=False, min_value=0, max_value=120)
    brca1 = forms.BooleanField(required=False, initial=False)
    brca2 = forms.BooleanField(required=False, initial=False)
    menarche_age = forms.FloatField(required=False, min_value=8, max_value=25)
    menopause_age = forms.FloatField(required=False, min_value=20, max_value=70)
    parity = forms.FloatField(required=False, min_value=0, max_value=20)
    hrt = forms.ChoiceField(choices=HRT_CHOICES, initial=0)
    bmi = forms.FloatField(required=False, min_value=8, max_value=70)
    alcohol_units_per_week = forms.FloatField(required=False, min_value=0, max_value=100)
    smoking_status = forms.ChoiceField(choices=SMOKE_CHOICES, initial=0)
    activity_hours_per_week = forms.FloatField(required=False, min_value=0, max_value=168)

    def clean(self):
        """Cross-field validations."""
        cleaned = super().clean()
        fdr = int(cleaned.get("first_degree_relative") or 0)
        onset = cleaned.get("onset_age_relative")
        if fdr == 1 and onset is None:
            self.add_error("onset_age_relative", "Provide onset age when a first-degree relative is selected.")
        return cleaned

    def to_dataclass(self) -> RiskFactors:
        """Convert cleaned form values to RiskFactors dataclass."""
        cd = self.cleaned_data
        return RiskFactors(
            age=float(cd["age"]),
            first_degree_relative=int(cd["first_degree_relative"]),
            onset_age_relative=float(cd["onset_age_relative"]) if cd.get("onset_age_relative") is not None else None,
            brca1=1 if cd.get("brca1") else 0,
            brca2=1 if cd.get("brca2") else 0,
            menarche_age=float(cd["menarche_age"]) if cd.get("menarche_age") is not None else None,
            menopause_age=float(cd["menopause_age"]) if cd.get("menopause_age") is not None else None,
            parity=float(cd["parity"]) if cd.get("parity") is not None else None,
            hrt=int(cd["hrt"]),
            bmi=float(cd["bmi"]) if cd.get("bmi") is not None else None,
            alcohol_units_per_week=float(cd["alcohol_units_per_week"]) if cd.get("alcohol_units_per_week") is not None else None,
            smoking_status=int(cd["smoking_status"]),
            activity_hours_per_week=float(cd["activity_hours_per_week"]) if cd.get("activity_hours_per_week") is not None else None,
        )
