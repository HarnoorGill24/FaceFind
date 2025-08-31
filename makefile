# =========================
# FaceFind Makefile
# Run `make help` to see usage.
# =========================

# --- Defaults (override via: make detect INPUT=~/Pictures) ---
INPUT            ?= $(HOME)/Documents/July22/Temp
OUTPUT           ?= outputs
MODEL_DIR        ?= models
PEOPLE_DIR       ?= outputs/people_by_cluster
CROPS_DIR        := $(OUTPUT)/crops/pending
REJECT_DIR       ?= $(OUTPUT)/crops/rejects
PRED_CSV         ?= $(OUTPUT)/predictions.csv

DEVICE           ?= mps
STRICTNESS       ?= strict
VIDEO_STEP       ?= 5
MAX_PER_MEDIA    ?= 50
PROGRESS_EVERY   ?= 100
# set LOG_NO_FACE=1 to enable
ifdef LOG_NO_FACE
LOG_NO_FACE_FLAG := --log-no-face
endif

# Verify runs are safest on CPU by default
VERIFY_DEVICE    ?= cpu
VERIFY_STRICTNESS?= normal

ACCEPT           ?= 0.80
REVIEW           ?= 0.50
COPY             ?= 0       # set COPY=1 to copy instead of hard-link

PYTHON           ?= python
PIP              ?= pip

# ---- Helpers ----
ifeq ($(COPY),1)
COPY_FLAG := --copy
endif

.PHONY: help env venv install clean clean-models clean-outputs clean-caches clean-all \
        detect verify stage-verified train predict autosort report all tests smoke \
        open-verified open-autosort stats

## help: Show this help
help:
	@grep -E '(^##)|(: .*)' $(MAKEFILE_LIST) | sed -E 's/Makefile://g' | sed -E 's/^## //'

## env: Print key variables
env:
	@echo "INPUT=$(INPUT)"
	@echo "OUTPUT=$(OUTPUT)"
	@echo "MODEL_DIR=$(MODEL_DIR)"
	@echo "PEOPLE_DIR=$(PEOPLE_DIR)"
	@echo "DEVICE=$(DEVICE)"
	@echo "STRICTNESS=$(STRICTNESS)"

## venv: Create a local virtualenv in .venv
venv:
	$(PYTHON) -m venv .venv
	@echo "Run: source .venv/bin/activate"

## install: Install requirements into the active venv
install:
	$(PIP) install -r requirements.txt

## clean-outputs: Remove generated outputs (crops, manifests, reports)
clean-outputs:
	rm -rf $(OUTPUT)/

## clean-models: Remove model artifacts
clean-models:
	rm -rf $(MODEL_DIR)/

## clean-caches: Remove Python/pytest caches
clean-caches:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .pytest_cache/ .cache/

## clean-all: Wipe outputs, models, caches
clean-all: clean-outputs clean-models clean-caches

## detect: Scan media, detect faces, save crops + manifest
detect:
	$(PYTHON) main.py \
		--input "$(INPUT)" \
		--output "$(OUTPUT)" \
		--video-step $(VIDEO_STEP) \
		--strictness $(STRICTNESS) \
		--device $(DEVICE) \
		--max-per-media $(MAX_PER_MEDIA) \
		--progress-every $(PROGRESS_EVERY) \
		$(LOG_NO_FACE_FLAG)

## verify: Re-check crops (CPU default) and move rejects
verify:
	mkdir -p "$(REJECT_DIR)"
	$(PYTHON) verify_crops.py "$(CROPS_DIR)" \
		--reject-dir "$(REJECT_DIR)" \
		--strictness $(VERIFY_STRICTNESS) \
		--device $(VERIFY_DEVICE)

## stage-verified: Copy verified crops listed in crops_verified.csv into a staging folder
stage-verified:
	mkdir -p $(OUTPUT)/crops/verified
	@[ -f "$(OUTPUT)/crops_verified.csv" ] || (echo "Missing $(OUTPUT)/crops_verified.csv"; exit 1)
	@awk -F, 'NR>1 {print $$1}' "$(OUTPUT)/crops_verified.csv" | while read -r p; do \
		[ -f "$$p" ] && cp "$$p" "$(OUTPUT)/crops/verified/"; \
	done
	@echo "Staged verified crops -> $(OUTPUT)/crops/verified"

## train: Train classifier from labeled folders under PEOPLE_DIR
train:
	$(PYTHON) train_face_classifier.py \
		--data "$(PEOPLE_DIR)" \
		--out "$(MODEL_DIR)" \
		--device $(DEVICE)

## predict: Predict labels for pending crops
predict:
	$(PYTHON) predict_face.py "$(CROPS_DIR)" \
		--model-dir "$(MODEL_DIR)" \
		--out "$(PRED_CSV)" \
		--device $(DEVICE)

## autosort: Place accepted/review images by confidence and update people_dir for accepts
autosort:
	$(PYTHON) apply_predictions.py "$(PRED_CSV)" \
		--people-dir "$(PEOPLE_DIR)" \
		--out-dir "$(OUTPUT)/autosort" \
		--accept-threshold $(ACCEPT) \
		--review-threshold $(REVIEW) \
		$(COPY_FLAG)

## report: Generate summary report (if your report.py supports it)
report:
	$(PYTHON) report.py "$(OUTPUT)" || true

## all: Run detect → verify → train → predict → autosort (report optional)
all: detect verify train predict autosort

## smoke: Quick pipeline with more permissive thresholds (good for first pass)
smoke:
	$(MAKE) detect STRICTNESS=normal
	$(MAKE) verify VERIFY_STRICTNESS=normal VERIFY_DEVICE=cpu
	$(MAKE) train
	$(MAKE) predict
	$(MAKE) autosort ACCEPT=0.70 REVIEW=0.40

## tests: Run pytest quickly
tests:
	pytest -q || true

## open-verified: (macOS) open Finder to verified staging folder
open-verified:
	open "$(OUTPUT)/crops/verified" || true

## open-autosort: (macOS) open Finder to autosort results
open-autosort:
	open "$(OUTPUT)/autosort" || true

## stats: Show a tiny prediction confidence summary
stats:
	@[ -f "$(PRED_CSV)" ] || (echo "Missing $(PRED_CSV)"; exit 1)
	@awk -F, 'NR>1 {print $$3}' "$(PRED_CSV)" | \
	awk '{p=$$1+0; n+=1; s+=p; if(p>=0.8) a+=1; else if(p>=0.5) b+=1; else c+=1} \
	     END {printf "count=%d  >=0.80:%d  0.50-0.79:%d  <0.50:%d  mean=%.3f\n", n,a,b,c,(n?s/n:0)}'

.PHONY: verify fix

verify:
	@echo "== Syntax =="
	@python -m compileall -q .
	@echo "== Ruff =="
	@ruff check .
	@echo "== Black (check) =="
	@black --check .
	@echo "== Mypy =="
	@mypy --ignore-missing-imports .
	@echo "== Pytest =="
	@pytest -q || echo "(no tests?)"

fix:
	@ruff check . --fix
	@black .
