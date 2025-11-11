from . import core

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


def run_count(ts):
    # TODO Raise a friendly numba-error here
    from . import jit

    return jit.count(ts)


def node_data(ts, inheritance_stats=False):
    """
    Return a pandas dataframe with one row for each node (in node ID order)
    from the specified tree sequence. This must be the output of
    ``sc2ts.minimise_metadata``, and will not work on the raw output of sc2ts.
    """

    md = ts.nodes_metadata
    cols = {k: md[k].astype(str) for k in md.dtype.names}
    dtype = {k: pd.StringDtype() for k in md.dtype.names}
    flags = ts.nodes_flags
    cols["node_id"] = np.arange(ts.num_nodes)
    dtype["node_id"] = "int"
    cols["is_sample"] = (flags & tskit.NODE_IS_SAMPLE) > 0
    dtype["is_sample"] = "bool"
    cols["is_recombinant"] = (flags & core.NODE_IS_RECOMBINANT) > 0
    dtype["is_recombinant"] = "bool"
    # Are other flags useful of just debug info? Lets leave them out
    # for now.
    cols["num_mutations"] = np.bincount(ts.mutations_node, minlength=ts.num_nodes)
    dtype["num_mutations"] = "int"
    # This is the same as is_recombinant but less obvious
    # cols["num_parents"] = np.bincount(ts.edges_child,
    #         minlength=ts.num_edges)

    if inheritance_stats:
        counter = run_count(ts)
        cols["max_descendant_samples"] = counter.nodes_max_descendant_samples
        dtype["max_descendant_samples"] = "int"
    if "time_zero_date" in ts.metadata:
        cols["date"] = convert_date(ts, ts.nodes_time)
        # Let Pandas infer the dtype of this to get the appropriate date type
    return pd.DataFrame(cols).astype(dtype)


def mutation_data(ts, inheritance_stats=False, parsimony_stats=False):
    """
    Return a pandas dataframe with one row for each mutation (in mutation ID order)
    from the specified tree sequence. This must be the output of
    ``sc2ts.minimise_metadata``, and will not work on the raw output of sc2ts.
    """
    cols = {}
    inherited_state = ts.mutations_inherited_state
    derived_state = ts.mutations_derived_state

    cols["mutation_id"] = np.arange(ts.num_mutations)
    cols["site_id"] = ts.mutations_site
    cols["position"] = ts.sites_position[ts.mutations_site].astype(int)
    cols["parent"] = ts.mutations_parent
    cols["node"] = ts.mutations_node
    cols["inherited_state"] = inherited_state
    cols["derived_state"] = derived_state
    if not isinstance(ts.metadata, bytes) and "time_zero_date" in ts.metadata:
        cols["date"] = convert_date(ts, ts.mutations_time)
    if inheritance_stats:
        counter = run_count(ts)
        cols["num_descendants"] = counter.mutations_num_descendants
        cols["num_inheritors"] = counter.mutations_num_inheritors

    if parsimony_stats:
        parent_node = np.zeros_like(cols["node"]) - 1
        pos = cols["position"]
        node = cols["node"]
        for tree in ts.trees():
            select = (tree.interval.left <= pos) & (pos < tree.interval.right)
            parent_node[select] = tree.parent_array[node[select]]
        cols["node_parent"] = parent_node

        inherited_state = np.append(cols["inherited_state"], "N")
        parent_mutation_node = np.append(cols["node"], -1)
        parent_inherited_state = inherited_state[cols["parent"]]
        parent_mutation_node = parent_mutation_node[cols["parent"]]
        cols["parent_inherited_state"] = parent_inherited_state
        cols["parent_mutation_node"] = parent_mutation_node
        cols["is_immediate_reversion"] = np.logical_and(
            cols["derived_state"] == cols["parent_inherited_state"],
            cols["node_parent"] == cols["parent_mutation_node"],
        )

    dtype = {k: "int" for k in cols if k != "date"}
    for k in dtype:
        if k.endswith("_state"):
            dtype[k] = pd.StringDtype()
        if k.startswith("is_"):
            dtype[k] = bool

    return pd.DataFrame(cols).astype(dtype)
