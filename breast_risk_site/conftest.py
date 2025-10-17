# conftest.py
import os, warnings

# Silence TF INFO logs like AVX2/FMA + oneDNN
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Trim proto deprecation spam from TF-generated modules
warnings.filterwarnings(
    "ignore",
    message=r"Call to deprecated create function .*Descriptor\(\)\.",
    category=DeprecationWarning,
    module=r"tensorflow\..*",
)
