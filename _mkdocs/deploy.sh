#!/bin/bash

CURRENT_NAME=$(basename `pwd`)
DOCS_DIR="../docs"

if [ "$CURRENT_NAME" != "_mkdocs" ]; then
  echo "invalid dir."
  exit 1
fi

if [ -d $DOCS_DIR ]; then
  rm -rf $DOCS_DIR
fi


# deploy
mkdocs build && mv site $DOCS_DIR
