"""Preload/initialize models into memory (useful for cold-start mitigation)."""

from django.core.management.base import BaseCommand
from predictor.services.registry import ModelRegistry

class Command(BaseCommand):
    help = "Warm up model registry by loading models into memory."

    def handle(self, *args, **options):
        img = ModelRegistry.image_model()
        fac = ModelRegistry.risk_model()
        self.stdout.write(self.style.SUCCESS("Models loaded: %s, %s" % (type(img).__name__, type(fac).__name__)))
