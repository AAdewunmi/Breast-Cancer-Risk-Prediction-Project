"""WTForms definitions for image upload + risk-factor forms.

Fields are minimal and intended as examples. Add validators as required.
"""
from flask_wtf import FlaskForm
from wtforms import BooleanField, FileField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange

MAG_CHOICES = [("unknown", "Unknown"), ("10x", "10x"), ("20x", "20x"), ("40x", "40x")]

SMOKING_CHOICES = [("never", "Never"), ("former", "Former"), ("current", "Current")]

HRT_CHOICES = [("never", "Never"), ("current", "Current"), ("past", "Past")]


class ImagePredictForm(FlaskForm):
    """Form for uploading an image and selecting metadata."""
    consent = BooleanField("I consent", default=False, validators=[Optional()])
    magnification = SelectField("Magnification", choices=MAG_CHOICES, default="unknown")
    image = FileField("Image file", validators=[Optional()])
    submit = SubmitField("Predict")


class FactorsForm(FlaskForm):
    """Form for entering risk-factor values."""
    age = IntegerField("Age", validators=[Optional(), NumberRange(min=0, max=120)])
    first_degree_relative = SelectField("First degree relative", choices=[("no", "No"), ("yes", "Yes")])
    brca1 = SelectField("BRCA1", choices=[("no", "No"), ("yes", "Yes")])
    brca2 = SelectField("BRCA2", choices=[("no", "No"), ("yes", "Yes")])
    menarche_age = IntegerField("Menarche age", validators=[Optional(), NumberRange(min=6, max=30)])
    menopause_age = IntegerField("Menopause age", validators=[Optional(), NumberRange(min=30, max=70)])
    parity = IntegerField("Parity", validators=[Optional(), NumberRange(min=0, max=20)])
    hrt = SelectField("HRT", choices=HRT_CHOICES, default="never")
    bmi = IntegerField("BMI", validators=[Optional(), NumberRange(min=10, max=80)])
    alcohol_units = IntegerField("Alcohol units per week", validators=[Optional(), NumberRange(min=0, max=100)])
    smoking = SelectField("Smoking status", choices=SMOKING_CHOICES, default="never")
    activity_hours = IntegerField("Activity hours per week", validators=[Optional(), NumberRange(min=0, max=168)])
    submit = SubmitField("Predict")
