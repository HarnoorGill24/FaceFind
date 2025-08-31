# facefind/io_schema.py
"""Shared CSV schema for FaceFind predictions."""

# Canonical column names
PREDICTIONS_SCHEMA = ("path", "label", "prob")

# Optional comment line we put at the top of CSVs
SCHEMA_MAGIC = "# FaceFindPredictions,v1"

# Helpful alias sets for tolerant readers
PATH_ALIASES = ("path", "file", "image")
LABEL_ALIASES = ("label", "prediction", "pred_label")
PROB_ALIASES = ("prob", "score", "confidence")
