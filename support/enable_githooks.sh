#!/bin/bash

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd $REPO_ROOT/support
wget https://github.com/DoozyX/clang-format-lint-action/raw/master/clang-format/clang-format14.0.0
mv clang-format14.0.0 clang-format
chmod +x clang-format
cd $REPO_ROOT
git config --local core.hooksPath .githooks/

