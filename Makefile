# Reproduction entry points. All targets read only the released data under
# data/ and need only the packages in requirements.txt.
PYTHON ?= python

.PHONY: all figures validate web test clean

all: validate figures web

figures:        ## regenerate every published figure into figures/
	$(PYTHON) -m scripts.make_figures

validate:       ## check schemas, row counts, checksums and report headline values
	$(PYTHON) -m scripts.validate_release

web:            ## write the browser-facing JSON files into web_export/
	$(PYTHON) -m scripts.export_web_data

test:           ## unit tests (needs requirements-full.txt)
	$(PYTHON) -m pytest -q tests

clean:
	rm -rf build
