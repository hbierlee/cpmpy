import itertools
import math
from cpmpy.solvers.lazy_gurobi import Heuristic, Coverlift, CPM_lazy_gurobi
from cpmpy.solvers.gurobi import CPM_gurobi, Encoding


MEM_LIMIT = 8
PINAC_42_MEM_LIMIT = 64
PINAC_42_WORKERS = 12


def calculate_workers(mem_limit, pinac_mem_limit, pinac_workers):
    workers = min(math.floor((pinac_mem_limit - 0.1) / 8), pinac_workers - 2)
    assert workers > 0
    return workers


DEFAULTS = [
    # setup
    [
        {
            "time_limit": 1 * 60,
            "first": False,
            "mem_limit": MEM_LIMIT * 1024,
            # pinac42: 1-12-20, 64Gb
            "workers": calculate_workers(MEM_LIMIT, PINAC_42_MEM_LIMIT, PINAC_42_WORKERS),
            "check_time_limit": 2 * 60,
            "output_dir": "results/dev",
            "no_timestamp": True,
            "profile": None,
        }
    ],
    # benchmarks
    [
        {
            "year": 2025,
            "track": track,
            "glob_instance": None,
        }
        for track in ["CSP22to25", "COP22to25"]
    ],
]


FEATURES = [
    ("heuristic", (Heuristic.GREEDY,)),
    (
        "fractional",
        (
            False,
            True,
        ),
    ),
    (
        "coverlift",
        (
            Coverlift.No,
            Coverlift.COM_MIN,
            Coverlift.COM_MAX,
            Coverlift.INPUT,
        ),
    ),
    (
        "shrink",
        (
            False,
            True,
        ),
    ),
    (
        "cutoff",
        (
            0,
            200,
            500,
            100,
        ),
    ),
    # ("negatives", (0,3)),
]


def glob_filter(experiments, filters):
    return [
        experiment
        for experiment in experiments
        if filters is None or all(any(v in experiment[k] for v in vs) for k, vs in filters)
    ]


def get_solvers(features=None, overrides={}, filters=None, add_all=True, add_none=True):
    return glob_filter(
        [
            *[{"solver": "ortools", "alias": "ortools"}],
            *[
                {
                    "solver": CPM_gurobi,
                    "alias": f"base_gurobi-{encoding}-reduce_{reduce}-{column_ordering}" if encoding == Encoding.MDD else f"base_gurobi-{encoding}",
                    "solver_kwargs": {"encoding": encoding, "reduce" : reduce, "column_ordering" : column_ordering,  "output_stats": True},
                    # "solve_kwargs": {"Seed": 42},
                }
                for encoding in [
                    Encoding.GLEB,
                    Encoding.BOOL,
                    Encoding.MDD
                ]

                for reduce in [True, False] if Encoding.MDD
                for column_ordering in ["input", "incr-domain", "decr-domain", "fiedler"] if Encoding.MDD
            ],
            *[
                {
                    "alias": f"{solver}-{alias}",
                    "solver": CPM_lazy_gurobi,
                    "solver_kwargs": solver_kwargs | {"output_stats": True},
                }
                for solver in ["lazy_gurobi"]
                for alias, solver_kwargs in [
                    (alias, {"env": env, "encoding": Encoding.BOOL})
                    for alias, env in ablate(
                        FEATURES if features is None else features,
                        add_none=add_none,
                        add_all=add_all,
                    )
                ]
            ],
        ],
        filters,
    )


def get_experiments(features=None, overrides={}, filters=[]):
    return glob_filter(
        experiment(
            get_solvers(features=features, overrides=overrides),
            overrides=overrides,
        ),
        filters,
    )


def experiment(experiments, overrides={}, filters=None):
    # Filter DEFAULTS to avoid duplicates when overrides specify a value
    # (e.g., when --track is specified, don't create experiments for both tracks)
    filtered_defaults = [
        [d for d in default_list if all(k not in d or d[k] is None or d[k] == v for k, v in overrides.items())]
        or default_list[:1]  # fallback to first if all filtered out
        for default_list in DEFAULTS
    ]
    return [
        experiment
        for experiment in [
            {
                **dict(it for di in experiment_ for it in di.items()),
                **overrides,
            }
            for experiment_ in itertools.product(*filtered_defaults, experiments)
        ]
        if filters is None or all(any(v in experiment[k] for v in vs) for k, vs in filters)
    ]


def ablate(feats, add_one=True, add_none=False, add_all=False, filters=None):
    return [
        *([("none", {feat: feat_vals[0] for feat, feat_vals in feats})] if add_none else []),
        *(
            [
                (
                    feat if feat_val is True else f"{feat}_{feat_val}",
                    {feat_: feat_val if feat == feat_ else feat_vals_[0] for feat_, feat_vals_ in feats},
                )
                for feat, feat_vals in feats
                if len(feat_vals) > 1 and (filters is None or feat in filters)
                for feat_val in feat_vals[1:]
            ]
            if add_one
            else []
        ),
        *([("all", {feat: feat_vals[-1] for feat, feat_vals in feats})] if add_all else []),
    ]
