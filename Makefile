VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PIP_INSTALL = $(PIP) install --exists-action=w

setup: venv deps

deps: venv
	@$(PIP_INSTALL) -r requirements.txt

venv:
	@pyvenv $(VENV)
	@$(PIP_INSTALL) -U pip setuptools pylint

clean_pyc:
	find . -not \( -path './venv' -prune \) -name '*.pyc' -exec rm -f {} \;
