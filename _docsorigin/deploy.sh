#!/bin/bash

CURRENT_NAME=$(basename `pwd`)
DOCS_DIR="../docs"

if [ "$CURRENT_NAME" != "_docsorigin" ]; then
  echo "invalid dir."
  exit 1
fi

if [ -d $DOCS_DIR ]; then
  rm -rf $DOCS_DIR
fi


# deploy
./jupyter_nbconvert_all.sh \
  && mkdocs build --clean \
  && ./build_sphinxapidocs.sh \
  && mv site $DOCS_DIR && touch $DOCS_DIR/.nojekyll
