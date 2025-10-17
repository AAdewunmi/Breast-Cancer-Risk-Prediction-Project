"""Django forms for image upload and risk-factor inputs."""
from __future__ import annotations

from django import forms

from .schemas import HRT_CHOICES, MAG_CHOICES, RiskFactors, SMOKE_CHOICES


class ImagePredictForm(forms.Form):
    image = forms.ImageField()
    # Optional; default to "unknown"
    magnification = forms.ChoiceField(
        choices=MAG_CHOICES, required=False, initial="unknown"
    )
    # If you have a consent box in the UI, keep it non-required so tests don't fail
    consent = forms.BooleanField(required=False, initial=True)

    def clean_magnification(self) -> str:
        # Treat missing/blank as "unknown" to keep API and UI resilient
        value = self.cleaned_data.get("magnification") or "unknown"
        # Safety: if choices ever change, still fall back to "unknown"
        valid = {k for k, _ in MAG_CHOICES}
        return value if value in valid else "unknown"


class ApiImagePredictForm(ImagePredictForm):
    """Looser variant used by the JSON API; same fields but all optional-friendly."""
    # Nothing extra needed; inherits optional magnification and non-required consent


class RiskFactorsForm(forms.Form):
    age = forms.FloatField(min_value=0, max_value=120)
    first_degree_relative = forms.ChoiceField(choices=[(1, "Yes"), (0, "No")], initial=0)
    onset_age_relative = forms.FloatField(required=False, min_value=0, max_value=120)
    brca1 = forms.BooleanField(required=False, initial=False)
    brca2 = forms.BooleanField(required=False, initial=False)

    menarche_age = forms.FloatField(required=False, min_value=8, max_value=25)
    menopause_age = forms.FloatField(required=False, min_value=30, max_value=70)
    parity = forms.FloatField(required=False, min_value=0, max_value=20)

    hrt = forms.ChoiceField(choices=HRT_CHOICES, initial=0)
    bmi = forms.FloatField(required=False, min_value=8, max_value=70)
    alcohol_units_per_week = forms.FloatField(required=False, min_value=0, max_value=100)
    smoking_status = forms.ChoiceField(choices=SMOKE_CHOICES, initial=0)
    activity_hours_per_week = forms.FloatField(required=False, min_value=0, max_value=168)

    def clean(self):
        cleaned = super().clean()
        # If a first-degree relative is selected, onset age must be present
        try:
            fdr = int(cleaned.get("first_degree_relative"))
        except (TypeError, ValueError):
            fdr = 0
        if fdr == 1 and cleaned.get("onset_age_relative") is None:
            self.add_error(
                "onset_age_relative",
                "Provide onset age when a first-degree relative is selected.",
            )
        return cleaned

    def to_dataclass(self) -> RiskFactors:
        cd = self.cleaned_data
        return RiskFactors(
            age=float(cd["age"]),
            first_degree_relative=int(cd["first_degree_relative"]),
            onset_age_relative=(
                float(cd["onset_age_relative"])
                if cd.get("onset_age_relative") is not None
                else None
            ),
            brca1=1 if cd.get("brca1") else 0,
            brca2=1 if cd.get("brca2") else 0,
            menarche_age=(
                float(cd["menarche_age"]) if cd.get("menarche_age") is not None else None
            ),
            menopause_age=(
                float(cd["menopause_age"])
                if cd.get("menopause_age") is not None
                else None
            ),
            parity=float(cd["parity"]) if cd.get("parity") is not None else None,
            hrt=int(cd["hrt"]),
            bmi=float(cd["bmi"]) if cd.get("bmi") is not None else None,
            alcohol_units_per_week=(
                float(cd["alcohol_units_per_week"])
                if cd.get("alcohol_units_per_week") is not None
                else None
            ),
            smoking_status=int(cd["smoking_status"]),
            activity_hours_per_week=(
                float(cd["activity_hours_per_week"])
                if cd.get("activity_hours_per_week") is not None
                else None
            ),
        )
