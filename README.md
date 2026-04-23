# Table Constraints for Integer Programming (CP'26)

Authors: Hendrik 'Henk' Bierlee, Wouter Piesens, Tias Guns, and Peter Stuckey
Venue: CP'26
Corresponding author's email: henk.bierlee@kuleuven.be (or bierlee.henk@gmail.com)

With this repository, you can reproduce the results of the paper by following these steps:

- Create and activate a virtual environment. Install the dependencies with `pip install -r cp-cuts-requirements.txt`. Ensure you have a Gurobi license activated. Optionally test with `pytest`.
- Download all CSP/COP instances from http://xcsp.org/instances/, and place them in `2025/CSP22to25/` and `2025/COP22to25/` (e.g. we should have `2025/CSP22to25/Accordion-11-01_c25.xml.lzma`)
- Run the benchmarks with: `python cpmpy/tools/xcsp3/benchmark.py --time-limit $((10*60)) --results ./results/ --no-timestamp`
    - This assumes the host machine has 64Gb RAM and 8 cores. The benchmarks are run in parallel, maximizing speed but without risk of running a benchmark with less than 8Gb of memory to avoid interference between benchmarks. If your machine may have different parameters (especially if it has less RAM), you can change the following line in `cpmpy/tools/xcsp3/experiments.py`: `None: {"mem": 64, "cores": 8}` according to your specs.
    - Optional: To check solver results using the XCSP3 checker from [the tools webpage](http://xcsp.org/tools/) (which links to this [repo](https://github.com/xcsp3team/XCSP3-Java-Tools)). I've found it to be easiest to install [`jbang`](https://www.jbang.dev/documentation/jbang/latest/installation.html), then install/compile the checker from the [maven repository](https://mvnrepository.com/artifact/org.xcsp/xcsp3-tools). Once installed, add the checker path as CLI flag when you run the benchmarks: `--checker-path "jbang  --main org.xcsp.parser.callbacks.SolutionChecker org.xcsp:xcsp3-tools:2.5"`
- Once benchmarks finish, run the analysis script: `mkdir analysis ; python cpmpy/tools/xcsp3/analyze.py ./results/  --plot analysis --tex analysis --sort-legend alpha --time-limit 600 --compare base_gurobi-mdd-reduce-dom-incr hybrid_0 --save analysis/results.csv --paper`
    - This should produce an `analysis` directory containing a raw result file `analysis/results.csv`, tables as included in the paper in `*.tex` files, and figures as `*.{png,svg}` images
