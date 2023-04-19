import numpy as np
import tskit

import sc2ts


# NOTE: the current API in which we update the Sample objects is
# really horrible and we need to refactor to make it more testable.
# This function is a symptom of that.
def get_samples(ts, paths, mutations=None):
    if mutations is None:
        mutations = [[] for _ in paths]

    # Translate from site IDs to positions
    updated_mutations = []
    for sample_mutations in mutations:
        updated = [
            (ts.sites_position[site], state) for (site, state) in sample_mutations
        ]
        updated_mutations.append(updated)
    samples = [sc2ts.Sample() for _ in paths]
    sc2ts.update_path_info(samples, ts, paths, updated_mutations)
    return samples


def example_binary(n):
    base = sc2ts.initial_ts()
    tables = base.dump_tables()
    tree = tskit.Tree.generate_balanced(n, span=base.sequence_length)
    binary_tables = tree.tree_sequence.dump_tables()
    binary_tables.nodes.time += 1
    tables.nodes.time += np.max(binary_tables.nodes.time) + 1
    binary_tables.edges.child += len(tables.nodes)
    binary_tables.edges.parent += len(tables.nodes)
    for node in binary_tables.nodes:
        tables.nodes.append(node.replace(metadata={}))
    for edge in binary_tables.edges:
        tables.edges.append(edge)
    # FIXME brittle
    tables.edges.add_row(0, base.sequence_length, parent=1, child=tree.root + 2)
    tables.sort()
    return tables.tree_sequence()

