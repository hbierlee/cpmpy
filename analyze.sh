#!/usr/bin/env bash
declare -a ARGS
ARGS+=(--sort-legend alpha)
ARGS+=(--small 100)
ARGS+=(--time-limit 600)
# ARGS+=(--no-errors)
ARGS+=(--scatter all bool)
ARGS+=(--paper)
# ARGS+=(--tex analysis)
# ARGS+=(--sync pinac42:/cw/dtailocal/henk/Projects/cpmpy)


rm -rf analysis
rm -rf ~/utm/analysis
mkdir analysis
python cpmpy/tools/xcsp3/analyze.py results/2026-02-12-csp  --plot analysis/csp --tex analysis/csp "${ARGS[@]}"
python cpmpy/tools/xcsp3/analyze.py results/2026-02-11-cop  --plot analysis/cop --tex analysis/cop "${ARGS[@]}"
cp -r analysis ~/utm/
rm -f ~/utm/analysis/*.png
