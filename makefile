
# FaceFind Makefile
# Quick commands for the end-to-end pipeline.
# Override variables on the command line, e.g.:
#   make scan INPUT=/data/photos STRICTNESS=normal
#   make predict INPUT=outputs/crops MODELS=models OUT=outputs/predictions.csv

# ---------- Defaults ----------
PY          ?= python
PIP         ?= pip
VENV        ?= .venv
ACTIVATE    ?= . $(VENV)/bin/activate

INPUT       ?= /PATH/TO/MEDIA
OUTPUTS     ?= outputs
MODELS      ?= models
STRICTNESS  ?= strict
VIDEO_STEP  ?= 5
REJECT_DIR  ?= $(OUTPUTS)/crops/rejects
PRED_CSV    ?= $(OUTPUTS)/predictions.csv

# Split targets
CSV_PATH    ?= $(PRED_CSV)
OUT_DIR     ?= $(OUTPUTS)/people_by_pred
COPY        ?= --copy

# ---------- Phony ----------
.PHONY: help venv install scan verify train predict split report clean

help:
	@echo "FaceFind Makefile targets:"
	@echo "  make venv                   # Create a Python venv (.venv)"
	@echo "  make install                # Install requirements into venv"
	@echo "  make scan                   # Run main.py to detect faces & write crops_manifest.csv"
	@echo "  make verify                 # Re-check crops and move rejects"
	@echo "  make train                  # Train classifier from labeled folders"
	@echo "  make predict                # Predict labels for a folder of images/crops"
	@echo "  make split                  # Split CSV (clusters/predictions) into folders"
	@echo "  make report                 # Summarize outputs → report.json"
	@echo ""
	@echo "Variables you can override:"
	@echo "  INPUT, OUTPUTS, MODELS, STRICTNESS, VIDEO_STEP, REJECT_DIR, PRED_CSV, CSV_PATH, OUT_DIR, COPY"
	@echo "Examples:"
	@echo "  make scan INPUT=/data/photos STRICTNESS=normal"
	@echo "  make verify REJECT_DIR=$(REJECT_DIR) STRICTNESS=$(STRICTNESS)"
	@echo "  make train MODELS=$(MODELS)"
	@echo "  make predict INPUT=$(OUTPUTS)/crops MODELS=$(MODELS) OUT=$(PRED_CSV)"
	@echo "  make split CSV_PATH=$(PRED_CSV) OUT_DIR=$(OUTPUTS)/people_by_pred)"
	@echo "  make report OUTPUTS=$(OUTPUTS) MODELS=$(MODELS) PREDICTIONS=$(PRED_CSV)"

venv:
	$(PY) -m venv $(VENV)
	@echo "Run: source $(VENV)/bin/activate"

install: venv
	$(ACTIVATE) && $(PIP) install -r requirements.txt

scan:
	$(ACTIVATE) && $(PY) main.py \
		--input "$(INPUT)" \
		--output "$(OUTPUTS)" \
		--video-step $(VIDEO_STEP) \
		--strictness $(STRICTNESS)

verify:
	$(ACTIVATE) && $(PY) verify_crops.py \
		"$(OUTPUTS)/crops/pending" \
		--reject-dir "$(REJECT_DIR)" \
		--strictness $(STRICTNESS)

train:
	$(ACTIVATE) && $(PY) train_face_classifier.py \
		--data "$(OUTPUTS)/people_by_cluster" \
		--out "$(MODELS)" \
		--strictness $(STRICTNESS)

predict:
	$(ACTIVATE) && $(PY) predict_face.py \
		"$(INPUT)" \
		--model-dir "$(MODELS)" \
		--out "$(PRED_CSV)" \
		--strictness $(STRICTNESS)

split:
	$(ACTIVATE) && $(PY) split_clusters.py \
		"$(CSV_PATH)" \
		"$(OUT_DIR)" \
		$(COPY)

report:
	$(ACTIVATE) && $(PY) report.py \
		--outputs "$(OUTPUTS)" \
		--models "$(MODELS)" \
		--predictions "$(PRED_CSV)" \
		--save-json "$(OUTPUTS)/report.json"

clean:
	rm -rf "$(OUTPUTS)/crops/pending" "$(OUTPUTS)/crops/rejects" "$(OUTPUTS)/people_by_cluster" "$(OUTPUTS)/people_by_pred" \
	       "$(OUTPUTS)/crops_manifest.csv" "$(OUTPUTS)/crops_verified.csv" "$(OUTPUTS)/predictions.csv" "$(OUTPUTS)/report.json"
