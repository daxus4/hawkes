import tick.hawkes as hk

sim_hawkes = hk.SimuHawkesExpKernels(
        adjacency=caa, decays=baa, baseline=[abb], end_time=warm_up_period_duration, seed=seed
    )
