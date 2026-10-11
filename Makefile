# Native CUDA worker. CI/release dispatch works from git bash or mac; local targets
# build the CPU reference backend (macOS/Accelerate) used to validate the model code.

CI_WORKFLOW := .github/workflows/ci.yml
RELEASE_WORKFLOW := .github/workflows/release.yml

REPO ?= $(shell git remote get-url origin 2>/dev/null | sed -E 's|git@github\.com:||; s|https://github\.com/||; s|\.git$$||')
GH_R := $(if $(REPO),-R "$(REPO)",)
REF ?= $(shell git branch --show-current 2>/dev/null)
PUBLISH_GH_RELEASE ?= true

MODELS_TAG := $(shell grep '^MODELS_RELEASE_TAG=' project.conf | cut -d= -f2)
MAC_SILICON ?= ../demucs-mac-silicon
PYTHON ?= $(MAC_SILICON)/.venv/bin/python
FP32_WEIGHTS ?= $(MAC_SILICON)/models/htdemucs_6s.safetensors
FP32_CONFIG ?= $(MAC_SILICON)/models/htdemucs_6s.json
GOLDENS ?= goldens/htdemucs_6s/goldens.safetensors
CPU_BUILD := worker-cuda/build-cpu

.PHONY: ci ci-watch release release-watch models goldens cpu-build test test-compare models-release

ci:
	@test -n "$(REPO)" || (echo "ERROR: could not resolve origin repo; set REPO=owner/name" >&2; exit 1)
	gh $(GH_R) workflow run "$(CI_WORKFLOW)" \
		$(if $(REF),-r "$(REF)",) \
		$(if $(VERSION),-f version="$(VERSION)",)

ci-watch: ci
	@sleep 2
	@RID=$$(gh $(GH_R) run list --workflow="$(CI_WORKFLOW)" -L 1 --json databaseId -q '.[0].databaseId'); \
		test -n "$$RID"; \
		gh $(GH_R) run watch "$$RID"

release:
	@test -n "$(REPO)" || (echo "ERROR: could not resolve origin repo; set REPO=owner/name" >&2; exit 1)
	gh $(GH_R) workflow run "$(RELEASE_WORKFLOW)" \
		$(if $(REF),-r "$(REF)",) \
		-f publish_gh_release="$(PUBLISH_GH_RELEASE)" \
		$(if $(VERSION),-f version="$(VERSION)",)

release-watch: release
	@sleep 2
	@RID=$$(gh $(GH_R) run list --workflow="$(RELEASE_WORKFLOW)" -L 1 --json databaseId -q '.[0].databaseId'); \
		test -n "$$RID"; \
		gh $(GH_R) run watch "$$RID"

models/htdemucs_6s.fp16.safetensors: $(FP32_WEIGHTS)
	$(PYTHON) scripts/export_fp16_safetensors.py $(FP32_WEIGHTS) $(FP32_CONFIG) models

models: models/htdemucs_6s.fp16.safetensors

$(GOLDENS):
	$(PYTHON) scripts/make_reference_tensors.py --mac-silicon $(MAC_SILICON) -o $(dir $(GOLDENS))

goldens: $(GOLDENS)

cpu-build:
	cmake -S worker-cuda -B $(CPU_BUILD) -DDC_BACKEND=cpu -DCMAKE_BUILD_TYPE=Release
	cmake --build $(CPU_BUILD) -j

test: cpu-build
	$(CPU_BUILD)/test_stft
	$(CPU_BUILD)/test_kernels

test-compare: cpu-build models goldens
	$(CPU_BUILD)/test_compare models/htdemucs_6s.fp16.safetensors models/htdemucs_6s.json $(GOLDENS)

# One-time (per MODELS_RELEASE_TAG) upload of the fp16 weights CI packages into the worker zip.
models-release: models
	@test -n "$(REPO)" || (echo "ERROR: could not resolve origin repo; set REPO=owner/name" >&2; exit 1)
	@if gh $(GH_R) release view "$(MODELS_TAG)" >/dev/null 2>&1; then \
		echo "ERROR: $(MODELS_TAG) already exists; bump MODELS_RELEASE_TAG in project.conf" >&2; exit 1; fi
	rm -rf release-assets/models && mkdir -p release-assets/models
	cp models/htdemucs_6s.fp16.safetensors models/htdemucs_6s.json release-assets/models/
	cd release-assets && rm -f htdemucs-6s-fp16.zip && zip -9 -r htdemucs-6s-fp16.zip models \
		&& shasum -a 256 htdemucs-6s-fp16.zip > htdemucs-6s-fp16.zip.sha256
	gh $(GH_R) release create "$(MODELS_TAG)" --latest=false \
		--title "fp16 weights $(MODELS_TAG)" \
		--notes "Immutable htdemucs_6s fp16 safetensors for demucs_cuda_worker. Do not overwrite; bump MODELS_RELEASE_TAG to publish new weights." \
		release-assets/htdemucs-6s-fp16.zip release-assets/htdemucs-6s-fp16.zip.sha256
