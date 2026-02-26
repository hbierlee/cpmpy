#!/usr/bin/env bash

declare -a ARGS
ARGS+=(--sort-legend alpha)
# ARGS+=(--small 100)
ARGS+=(--time-limit 600)
# ARGS+=(--no-errors)
# ARGS+=(--compare all bool)
ARGS+=(--compare bool coverlift_input)
ARGS+=(--paper)
ARGS+=(--glob-alias bool gleb all coverlift_input none shrink)
# ARGS+=(--no-errors)
# ARGS+=(--tex analysis)
ARGS+=(--sync pinac42:/cw/dtailocal/henk/Projects/cpmpy)


rm -rf analysis
rm -rf ~/utm/analysis
mkdir analysis
python cpmpy/tools/xcsp3/analyze.py results/2026-02-21 --plot analysis --tex "analysis" "${ARGS[@]}"
cp -r analysis ~/utm/
rm -f ~/utm/analysis/*.png
ls ~/utm/analysis
