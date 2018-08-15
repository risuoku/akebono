#!/bin/bash

if [ ! -d ./site ]; then
  echo "./site does not exist. create it before sphinx-apidoc"
  exit 1
fi
rm -rf ./site/sphinxapidocs

cd _sphinxapidocs \
  && rm -rf _build \
  && PYTHONPATH=../.. make html \
  && mv _build/html ../site/sphinxapidocs
