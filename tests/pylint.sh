#!/bin/bash

REPO_URL=$(echo "${CI_BUILD_REPO}" | sed -r 's#([^:]+://)[^:]+:[^@]+@(.*)#\1\2#g')

# CI jobs are triggerd by `git push` in GitLab, and for newly pushed branches,
# $CI_BUILD_BEFORE_SHA is always `0000000000000000000000000000000000000000`,
# which is useless for diff calculation.
PREV_HEAD=$(echo "${CI_BUILD_BEFORE_SHA}" | sed -r '/^[0]+$/d')
CURRENT_HEAD="${CI_BUILD_REF}"

# Assuming any branch outside upstream repo is forked from upstream/master,
# we can fall back to that as a reasonable starting point.
if [[ -z "${PREV_HEAD}" || "${REPO_URL}" != "${UPSTREAM_URL}" ]]
then
    PREV_HEAD='origin/master'
fi

list_changed_py_files() {
    git diff --name-status \
            "${PREV_HEAD}".."${CURRENT_HEAD}" \
    | sed -n '/^[^ D].*.py$/p' \
    | cut -f 2 \
    | sort --unique
}

NUM_OF_FILES=$(list_changed_py_files | wc -l)

if [[ "${NUM_OF_FILES}" != "0" ]]
then
    list_changed_py_files | xargs pylint --rcfile=tests/pylintrc
else
    echo "No .py gets changed."
fi
