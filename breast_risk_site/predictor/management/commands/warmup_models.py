from __future__ import annotations

from django.core.management.base import BaseCommand

from predictor.services.registry import ModelRegistry


class Command(BaseCommand):
    help = "Load image and risk models so the first request is fast."

    def handle(self, *args, **options):
        img = ModelRegistry.image_model()
        fac = ModelRegistry.risk_model()
        msg = f"Models loaded: {type(img).__name__}, " f"{type(fac).__name__}"
        self.stdout.write(self.style.SUCCESS(msg))
