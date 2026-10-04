#!/bin/bash

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd $REPO_ROOT/support
wget https://github.com/DoozyX/clang-format-lint-action/raw/master/clang-format/clang-format18.1.8
mv clang-format18.1.8 clang-format
chmod +x clang-format
cd $REPO_ROOT
git config --local core.hooksPath .githooks/

