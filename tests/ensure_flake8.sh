#!/bin/bash

PIP=""
if [ -e "venv" ];then
    PIP=venv/bin/pip
elif [ -e "$(which pip3.6)" ];then
    PIP=pip3.6
elif [ -e "$(which pip3.5)" ];then
    PIP=pip3.5
elif [ -e "$(which pip3)" ];then
    PIP=pip3
else
    PIP=pip
fi

${PIP} install flake8 --quiet
