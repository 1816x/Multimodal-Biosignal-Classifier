"""FastAPI service for the multimodal biosignal classifier.

Educational prototype — NOT an approved medical or diagnostic tool.
"""
__version__ = "0.1.0"

# Canonical educational disclaimer. Lives here (not in main.py) so both the API
# routes and the Claude explanation layer can import it without a circular import,
# and so every prediction/report response carries the exact same text.
DISCLAIMER = (
    "EDUCATIONAL PROTOTYPE — NOT an approved medical or diagnostic tool. "
    "This service must not be used for clinical decisions. Model outputs are "
    "illustrative only."
)
