#!/usr/bin/env bash

res="/cw/dtailocal/henk/Projects/cpmpy/results/*"
rsync -r himec03:$res ./results/himec03
rsync -r himec04:$res ./results/himec04

rm -f results/himec03/2026-03-05-lazy/xcsp3_2025_CSP22to25_lazy_gurobi-best_no_cutoff.csv
rm -f results/himec03/2026-03-05-lazy/xcsp3_2025_COP22to25_lazy_gurobi-best_no_cutoff.csv

declare -a DIRS
DIRS=(
# results/himec03/2026-02-28/xcsp3_2025_COP22to25_lazy_gurobi-all.csv
# results/himec03/2026-02-28/xcsp3_2025_CSP22to25_lazy_gurobi-all.csv
# results/himec03/2026-02-29-ablation
# results/himec04/2026-02-29-remaining-base
# results/himec04/2026-03-02-grb-13-bests-reduce
results/himec03/2026-03-02-grb-13-base
results/himec03/2026-03-05-lazy
results/himec03/2026-03-06-no-neg/
#results/himec04/2026-03-05-cutoffs
# results/himec04/2026-03-02-grb-13-bests/xcsp3_2025_CSP22to25_lazy_gurobi-best.csv
# results/2026-02-30-grb-13/xcsp3_2025_COP22to25_lazy_gurobi-best-no_cutoff.csv
# results/himec03/2026-03-05-lazy  # TODO in progr
)
declare -a ARGS

ARGS+=(--sort-legend alpha)
# ARGS+=(--small 100)
ARGS+=(--time-limit 600)
# ARGS+=(--no-errors)
#
# ARGS+=(--compare all bool)
# ARGS+=(--compare bool all)
#

# ARGS+=(--compare base_gurobi-mdd-reduce-dom-incr-hashtable best)
ARGS+=(--compare base_gurobi-mdd-reduce-dom-incr-hashtable best_no_cutoff)
# ARGS+=(--compare mdd-reduce-fiedler cutoff_1000)
# ARGS+=(--compare mdd-reduce-fiedler cutoff_100)

ARGS+=(--paper)

# ARGS+=(--glob-alias bool gleb all coverlift_input frac none mdd-reduce mdd-noreduce-dom-incr ortools cutoff)
#
ARGS+=(--exclude-alias mdd-reduce-dom-decr mdd-reduce-fiedler ortools noreduce)

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
