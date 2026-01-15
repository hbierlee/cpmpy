import itertools



def get_experiments(args, glob_alias=None):
    return list(
        {
            **dict(it for di in x for it in di.items()),
            **args,
        }
        for x in itertools.product(
            [
                # setup
                {
                    "time_limit": 1 * 60,
                    "check_time_limit": 10,
                    "first": False,
                    "mem_limit": 8 * 1024,
                    "cores": 1,
                    "workers": 7,  # pinac42: 1-12-20, 64Gb
                    "check_time_limit": 2 * 60,
                    "output_dir": "results",
                    "no_timestamp": True,
                }
            ],
            [
                # benchmarks
                {
                    "year": 2025,
                    "track": "COP25",
                    "glob": None,
                }
            ],
            [
                solver
                for solver in [
                    # solvers
                    {"solver": "gurobi", "alias": "base_gurobi"},  # TODO gen.
                    *[
                        {"alias": f"{solver}-{alias}", "solver": solver, "solver_kwargs": {"env": kw}}
                        for solver in ["lazy_gurobi"]
                        for alias, kw in [
                            *ablate(
                                [
                                    "fractional",
                                    "shrink",
                                    "coverlift",
                                ],
                            ),
                            *[
                                (
                                    "no_shrink",
                                    {
                                        "fractional": True,
                                        "shrink": False,
                                        "coverlift": True,
                                    },
                                )
                            ],
                        ]
                    ],
                ]
                if glob_alias is None or solver["alias"] == glob_alias
            ],
        )
    )


def ablate(feats, add_all=True):
    return [
        *[("none", {feat: False for feat in feats})],
        *[(feat, {feat_: feat_ == feat for feat_ in feats}) for feat in feats],
        *[("all", {feat: True for feat in feats})],
    ]
