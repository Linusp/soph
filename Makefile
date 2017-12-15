setup: venv deps

deps: venv
	- venv/bin/pip install -r requirements.txt

venv:
	- virtualenv --python=$(shell which python3.6) --prompt '<venv:soph>' venv
	- venv/bin/pip install setuptools pylint -U

clean:
	- find . -iname "*__pycache__" | xargs rm -rf
	- find . -iname "*.pyc" | xargs rm -rf

