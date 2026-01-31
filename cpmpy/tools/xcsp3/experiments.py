import itertools
import math
from cpmpy.solvers.lazy_gurobi import Heuristic


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
            "output_dir": "results",
            "no_timestamp": True,
            "profile": None,
        }
    ],
    # benchmarks
    [
        {
            "year": 2025,
            "track": "COP25",
            "glob_instance": None,
        }
    ],
]


def get_experiments(overrides={}, filters=[]):
    return experiment(
        [
            [
                # solvers
                {"solver": "gurobi", "alias": "base_gurobi"},  # TODO gen.
                *[
                    {"alias": f"{solver}-{alias}", "solver": solver, "solver_kwargs": {"env": kw}}
                    for solver in ["lazy_gurobi"]
                    for alias, kw in [
                        *ablate(
                            [
                                ("heuristic", (Heuristic.INPUT, Heuristic.GREEDY)),
                                ("fractional", (False, True)),
                                # ("shrink", (False, True)),
                                ("coverlift", (False, True)),
                                (
                                    "cutoff",
                                    (
                                        0,
                                        125,
                                    ),
                                ),
                            ],
                            add_none=True,
                            add_all=True,
                            # filters=["heuristic"],
                        ),
                    ]
                ],
            ]
        ],
        filters=filters,
        overrides=overrides,
    )


def experiment(experiments, overrides={}, filters=None):
    return [
        experiment
        for experiment in [
            {
                **dict(it for di in experiment_ for it in di.items()),
                **overrides,
            }
            for experiment_ in itertools.product(*DEFAULTS, *experiments)
        ]
        if filters is None or all(any(v in experiment[k] for v in vs) for k, vs in filters)
    ]


def ablate(feats, add_one=True, add_none=True, add_all=False, filters=None):
    return [
        *([("none", {feat: feat_vals[0] for feat, feat_vals in feats})] if add_none else []),
        *(
            [
                (
                    feat if feat_val is True else f"{feat}-{feat_val}",
                    {feat_: feat_val if feat == feat_ else feat_vals_[0] for feat_, feat_vals_ in feats},
                )
                for feat, feat_vals in feats
                if filters is None or feat in filters
                for feat_val in feat_vals[1:]
            ]
            if add_one
            else []
        ),
        *([("all", {feat: feat_vals[1] for feat, feat_vals in feats})] if add_all else []),
    ]
