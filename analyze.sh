#!/usr/bin/env bash

res="/cw/dtailocal/henk/Projects/cpmpy/results/*"
rsync -r himec03:$res ./results/himec03
rsync -r himec04:$res ./results/himec04

declare -a DIRS
DIRS=(
results/himec03/2026-02-28
results/himec03/2026-02-29-ablation
results/himec04/2026-02-29-remaining-base
# results/himec04/2026-03-02-grb-13-bests-reduce
)
declare -a ARGS
ARGS+=(--sort-legend alpha)
# ARGS+=(--small 100)
ARGS+=(--time-limit 600)
# ARGS+=(--no-errors)
#
# ARGS+=(--compare all bool)
# ARGS+=(--compare bool all)
ARGS+=(--compare mdd-reduce-fiedler all)
# ARGS+=(--compare mdd-reduce-fiedler cutoff_1000)
# ARGS+=(--compare mdd-reduce-fiedler cutoff_100)

# ARGS+=(--paper)
# ARGS+=(--glob-alias bool gleb all coverlift_input frac none mdd-reduce mdd-noreduce-dom-incr ortools cutoff)
# ARGS+=(--exclude-alias mdd-reduce-dom-decr)

# ARGS+=(--no-errors)
# ARGS+=(--tex analysis)
# ARGS+=(--sync pinac42:/cw/dtailocal/henk/Projects/cpmpy)


rm -rf analysis
rm -rf ~/utm/analysis
mkdir analysis
python cpmpy/tools/xcsp3/analyze.py "${DIRS[@]}"  --plot analysis --tex analysis "${ARGS[@]}"
cp -r analysis ~/utm/
rm -f ~/utm/analysis/*.png
ls ~/utm/analysis
