"""Warm up model registry by loading models into memory."""

from django.core.management.base import BaseCommand
from predictor.services.registry import ModelRegistry


class Command(BaseCommand):
    help = "Preload image and risk-factor models to reduce first-request latency."

    def handle(self, *args, **options):
        img = ModelRegistry.image_model()
        fac = ModelRegistry.risk_model()
        self.stdout.write(
            self.style.SUCCESS(f"Models loaded: {type(img).__name__}, {type(fac).__name__}")
        )
