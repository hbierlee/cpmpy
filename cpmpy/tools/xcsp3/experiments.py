import itertools
import math
from cpmpy.solvers.lazy_gurobi import Heuristic, Coverlift, CPM_lazy_gurobi
from cpmpy.solvers.gurobi import CPM_gurobi, Encoding, Order


MEM_LIMIT = 8
PINAC_42_MEM_LIMIT = 64
PINAC_42_WORKERS = 12

MACHINES = {
    None: {"mem": 64, "cores": 8},
    "pinac42": {"mem": 64, "cores": 12},
    "himec03": {"mem": 128, "cores": 16},
    "himec04": {"mem": 128, "cores": 16},
    "himec06": {"mem": 128, "cores": 24},
    "himec07": {"mem": 128, "cores": 24},
}


def calculate_workers(mem_limit, host_mem, host_workers):
    workers = min(math.floor((host_mem - 0.1) / 8), host_workers - 2)
    assert workers > 0
    return workers


DEFAULTS = [
    # setup
    [
        {
            "time_limit": 1 * 60,
            "first": False,
            "mem_limit": MEM_LIMIT * 1024,
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
        for track in [
            "CSP22to25",
            "COP22to25",
        ]
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
            # Coverlift.INPUT,
            # Coverlift.COM_MIN,
            Coverlift.COM_MAX,
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
            500,
            1000,
            2000,
        ),
    ),
    (
        "negatives",
        (
            False,
            True,
        ),
    ),
]


def glob_filter(experiments, filters):
    return [
        experiment
        for experiment in experiments
        if filters is None or all(any(v in experiment[k] for v in vs) for k, vs in filters)
    ]


SEED = 42


def get_solvers(features=None, overrides={}, filters=None, add_all=False, add_none=True):
    if features is None:
        features = FEATURES

    BEST = enable_all(features) | {"shrink": False}

    return glob_filter(
        [
            *[{"solver": "ortools", "alias": "ortools", "solve_kwargs": {"random_seed": SEED}}],
            *[
                {
                    "solver": CPM_gurobi,
                    "alias": f"base_gurobi-{encoding}"
                    + (
                        f"-{f'reduce' if reduce else 'noreduce'}-{order}-{f'combined' if combined else 'nocombined'}"
                        if encoding is Encoding.MDD
                        else ""
                    ),
                    "solver_kwargs": {
                        "encoding": encoding,
                        "reduce": reduce,
                        "order": order,
                        "combined": combined,
                        "output_stats": True,
                    },
                    "solve_kwargs": {"Seed": SEED},
                }
                for encoding, reduce, order, combined in [
                    (encoding_, reduce, order, combined)
                    for encoding_ in [Encoding.GLEB, Encoding.BOOL, Encoding.MDD]
                    for reduce in (
                        [
                            True,
                            False,
                        ]
                        if encoding_ is Encoding.MDD
                        else [None]
                    )
                    for order in (
                        [
                            Order.INPUT,
                            Order.DOM_INCR,
                            Order.GREEDY,
                            Order.BIDIRECTIONAL,
                            Order.FIEDLER,
                            Order.LOOKAHEAD_2,
                            Order.LOOKAHEAD_3,
                        ]
                        if encoding_ is Encoding.MDD
                        else [None]
                    )
                    for combined in (
                        [
                            True,
                            False,
                        ]
                        if encoding_ is Encoding.MDD
                        else [None]
                    )
                ]
            ],
            *[
                {
                    "alias": f"{solver}-{alias}",
                    "solver": CPM_lazy_gurobi,
                    "solver_kwargs": solver_kwargs | {"output_stats": True},
                    "solve_kwargs": {"Seed": SEED},
                }
                for solver in ["lazy_gurobi"]
                for alias, solver_kwargs in [
                    (alias, {"env": env, "encoding": Encoding.MDD, "order": Order.DOM_INCR, "reduce": True})
                    for alias, env in ablate(features, add_none=add_none, add_all=add_all)
                    + [(f"hybrid_{c}", enable_all(features) | {"cutoff": c}) for f, c in FEATURES if f == "cutoff" for c in c]
                    + [
                        (
                            "dev",
                            enable_all(features)
                            | {
                                "shrink": False,
                                "cutoff": 0,
                                "fractional": True,
                                "negatives": True,
                                "coverlift": Coverlift.COM_MAX,
                            },
                        )
                    ]
                ]
            ],
        ],
        filters,
    )


def get_experiments(features=None, overrides={}, filters=[], host=None):

    return glob_filter(
        experiment(
            get_solvers(features=features, overrides=overrides),
            overrides=overrides,
            host=host,
        ),
        filters,
    )


def experiment(experiments, overrides={}, filters=None, host=None):
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
                **{"workers": calculate_workers(MEM_LIMIT, MACHINES[host]["mem"], MACHINES[host]["cores"])},
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
        *([("all", enable_all(feats))] if add_all else []),
    ]


def enable_all(feats):
    return {feat: feat_vals[-1] for feat, feat_vals in feats}
