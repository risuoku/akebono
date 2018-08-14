#!/bin/bash

TARGETS=$(find ./_notebooks -type f -regex "^\.\/_notebooks\/.*\.ipynb$" | grep -ve "ipynb_checkpoints")

rm -rf docs/notebooks && mkdir docs/notebooks

for t in $TARGETS
do
  jupyter nbconvert --to markdown --output-dir docs/notebooks $t
done
