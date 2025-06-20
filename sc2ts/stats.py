from . import core
from . import jit

import numpy as np
import pandas as pd
import tskit


def convert_date(ts, time_array):
    time_zero_date = ts.metadata["time_zero_date"]
    time_zero_date = np.array([time_zero_date], dtype="datetime64[D]")[0]
    # Not clear that we're rounding things in the right direction here
    # we could output the dates in higher precision if we wanted to
    # but day precision is probably right anyway
    return time_zero_date - time_array.astype("timedelta64[D]")


def compute_mutation_states(ts):
    tables = ts.tables
    assert np.all(
        tables.mutations.derived_state_offset == np.arange(ts.num_mutations + 1)
    )
    derived_state = tables.mutations.derived_state.view("S1").astype(str)
    assert np.all(tables.sites.ancestral_state_offset == np.arange(ts.num_sites + 1))
    ancestral_state = tables.sites.ancestral_state.view("S1").astype(str)
    del tables

    inherited_state = ancestral_state[ts.mutations_site]
    mutations_with_parent = ts.mutations_parent != -1

    parent = ts.mutations_parent[mutations_with_parent]
    assert np.all(parent >= 0)
    inherited_state[mutations_with_parent] = derived_state[parent]

    return inherited_state, derived_state


def node_data(ts, inheritance_stats=True):
    """
    Return a pandas dataframe with one row for each node (in node ID order)
    from the specified tree sequence. This must be the output of
    ``sc2ts.minimise_metadata``, and will not work on the raw output of sc2ts.
    """

    md = ts.nodes_metadata
    cols = {k: md[k].astype(str) for k in md.dtype.names}
    flags = ts.nodes_flags
    cols["node_id"] = np.arange(ts.num_nodes)
    cols["is_sample"] = (flags & tskit.NODE_IS_SAMPLE) > 0
    cols["is_recombinant"] = (flags & core.NODE_IS_RECOMBINANT) > 0
    # Are other flags useful of just debug info? Lets leave them out
    # for now.
    cols["num_mutations"] = np.bincount(ts.mutations_node, minlength=ts.num_nodes)
    # This is the same as is_recombinant but less obvious
    # cols["num_parents"] = np.bincount(ts.edges_child,
    #         minlength=ts.num_edges)

    if inheritance_stats:
        counter = jit.count(ts)
        cols["max_descendant_samples"] = counter.nodes_max_descendant_samples
    if "time_zero_date" in ts.metadata:
        cols["date"] = convert_date(ts, ts.nodes_time)
    return pd.DataFrame(cols)


def mutation_data(ts, inheritance_stats=True):
    """
    Return a pandas dataframe with one row for each mutation (in mutation ID order)
    from the specified tree sequence. This must be the output of
    ``sc2ts.minimise_metadata``, and will not work on the raw output of sc2ts.
    """
    cols = {}
    inherited_state, derived_state = compute_mutation_states(ts)

    cols["mutation_id"] = np.arange(ts.num_mutations)
    cols["position"] = ts.sites_position[ts.mutations_site].astype(int)
    cols["parent"] = ts.mutations_parent
    cols["node"] = ts.mutations_node
    cols["inherited_state"] = inherited_state
    cols["derived_state"] = derived_state
    if "time_zero_date" in ts.metadata:
        cols["date"] = convert_date(ts, ts.mutations_time)
    if inheritance_stats:
        counter = jit.count(ts)
        cols["num_descendants"] = counter.mutations_num_descendants
        cols["num_inheritors"] = counter.mutations_num_inheritors

    return pd.DataFrame(cols)
