setup: venv deps

lint: venv clean
	- bash tests/ensure_flake8.sh
	- venv/bin/flake8 demos/ --format=pylint || true

deps: venv
	- venv/bin/pip install -r requirements.txt

venv:
	- virtualenv --python=$(shell which python3.6) --prompt '<venv:soph>' venv
	- venv/bin/pip install setuptools pip -U

clean:
	- find . -iname "*__pycache__" | xargs rm -rf
	- find . -iname "*.pyc" | xargs rm -rf

