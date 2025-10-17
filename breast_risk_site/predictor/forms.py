"""Django forms for image upload and risk-factor inputs."""

from __future__ import annotations
from django import forms
from .schemas import MAG_CHOICES, HRT_CHOICES, SMOKE_CHOICES, RiskFactors


class ImagePredictForm(forms.Form):
    """UI form: image upload + consent required."""

    image = forms.ImageField(required=True)
    magnification = forms.ChoiceField(choices=MAG_CHOICES, required=False)
    consent = forms.BooleanField(
        required=True,
        label="I understand this is a research prototype and not diagnostic.",
    )

    def clean_image(self):
        f = self.cleaned_data["image"]
        if f.content_type not in ("image/png", "image/jpeg"):
            raise forms.ValidationError("Please upload a PNG or JPG image.")
        if f.size > 10 * 1024 * 1024:
            raise forms.ValidationError("Image size must be ≤ 10 MB.")
        return f


class ApiImagePredictForm(ImagePredictForm):
    """API variant: consent optional for programmatic access."""

    consent = forms.BooleanField(required=False)


class RiskFactorsForm(forms.Form):
    """Structured risk factors with basic + cross-field validation."""

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
        cleaned = super().clean()
        fdr = int(cleaned.get("first_degree_relative") or 0)
        onset = cleaned.get("onset_age_relative")
        if fdr == 1 and onset is None:
            self.add_error(
                "onset_age_relative",
                "Provide onset age when a first-degree relative is selected.",
            )
        return cleaned

    def to_dataclass(self) -> RiskFactors:
        cd = self.cleaned_data

        def opt_float(x):
            return None if x in (None, "") else float(x)

        return RiskFactors(
            age=float(cd["age"]),
            first_degree_relative=int(cd["first_degree_relative"]),
            onset_age_relative=opt_float(cd.get("onset_age_relative")),
            brca1=1 if cd.get("brca1") else 0,
            brca2=1 if cd.get("brca2") else 0,
            menarche_age=opt_float(cd.get("menarche_age")),
            menopause_age=opt_float(cd.get("menopause_age")),
            parity=opt_float(cd.get("parity")),
            hrt=int(cd["hrt"]),
            bmi=opt_float(cd.get("bmi")),
            alcohol_units_per_week=opt_float(cd.get("alcohol_units_per_week")),
            smoking_status=int(cd["smoking_status"]),
            activity_hours_per_week=opt_float(cd.get("activity_hours_per_week")),
        )
