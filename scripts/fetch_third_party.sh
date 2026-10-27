#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

clone() {
    if [ ! -d "third_party/$1" ]; then
        git clone --quiet "$2" "third_party/$1"
        git -C "third_party/$1" checkout --quiet "$3"
    fi
}

clone CSDI https://github.com/ermongroup/CSDI.git 7f24a436f08d98853a6b43d4f7f04e5a65ecdf27
clone pytorch-ts https://github.com/zalandoresearch/pytorch-ts.git v0.6.0
clone LTSF-Linear https://github.com/cure-lab/LTSF-Linear.git 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6

GLUONTS=third_party/gluonts_v0.9.0/src/gluonts/torch/modules
mkdir -p "$GLUONTS"
for f in lambda_layer.py distribution_output.py; do
    curl -sSfL -o "$GLUONTS/$f" "https://raw.githubusercontent.com/awslabs/gluonts/v0.9.0/src/gluonts/torch/modules/$f"
done
