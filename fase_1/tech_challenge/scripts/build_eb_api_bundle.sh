#!/bin/sh

set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DIST_DIR="$PROJECT_ROOT/dist"
STAGE_DIR="$DIST_DIR/elastic_beanstalk_api"
ZIP_PATH="$DIST_DIR/elastic_beanstalk_api.zip"

if [ "${1-}" ]; then
    MODEL_DIR="$1"
else
    MODEL_FILE=$(find "$PROJECT_ROOT/mlartifacts" -type f -name MLmodel | sort | head -n 1 || true)

    if [ -z "$MODEL_FILE" ]; then
        echo "Nenhum artefato MLflow com arquivo MLmodel foi encontrado em mlartifacts/." >&2
        exit 1
    fi

    MODEL_DIR=$(dirname "$MODEL_FILE")
fi

if [ ! -f "$MODEL_DIR/MLmodel" ]; then
    echo "Diretório de modelo inválido: $MODEL_DIR" >&2
    exit 1
fi

rm -rf "$STAGE_DIR" "$ZIP_PATH"
mkdir -p "$STAGE_DIR/src" "$DIST_DIR"

cp "$PROJECT_ROOT/Dockerfile.beanstalk.api" "$STAGE_DIR/Dockerfile"
cp "$PROJECT_ROOT/pyproject.toml" "$STAGE_DIR/pyproject.toml"
cp -R "$PROJECT_ROOT/src/api" "$STAGE_DIR/src/api"
cp -R "$MODEL_DIR" "$STAGE_DIR/model"

(cd "$STAGE_DIR" && zip -qr "$ZIP_PATH" .)

echo "$ZIP_PATH"