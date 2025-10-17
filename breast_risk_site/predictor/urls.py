from django.urls import path
from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.PredictView.as_view(), name="predict"),
    path("about/", views.about, name="about"),
    path("resources/", views.resources, name="resources"),
    path("privacy/", views.privacy, name="privacy"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
