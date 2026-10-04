#!/bin/bash

set -e

docker run \
    --rm \
    -it \
    --gpus all \
    -v "$(pwd):/workspace" \
    -w /workspace \
    dp4a-poc
