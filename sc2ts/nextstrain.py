"""
Utilities for converting a Nextstrain tree to tskit format
and comparing with an sc2ts output.
"""
import collections

import tskit

# https://github.com/nextstrain/nextclade_data/tree/release/data/datasets/sars-cov-2/references/MN908947/versions

# e.g.
# https://raw.githubusercontent.com/nextstrain/nextclade_data/release/data/datasets/sars-cov-2/references/MN908947/versions/2021-06-25T00:00:00Z/files/tree.json


def add_node(nodes, js_node, parent):
    metadata = {}
    flags = 0
    ns_name = js_node["name"]
    if not ns_name.startswith("NODE"):
        # We only really care about the samples
        metadata["strain"] = ns_name
        flags = tskit.NODE_IS_SAMPLE
    # We don't seem to have any times in these trees annoyingly
    # time = -js_node["node_attrs"]["num_date"]["value"]
    time = 0
    if parent != -1:
        parent_time = nodes.time[parent]
        if time >= parent_time:
            time = parent_time - 1

    return nodes.add_row(flags=flags, time=time, metadata=metadata)


def convert_nextstrain(document):
    root = document["tree"]
    tables = tskit.TableCollection(29904)
    nodes = tables.nodes
    nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    edges = tables.edges
    stack = [(root, add_node(nodes, root, -1))]
    mutations = collections.defaultdict(list)

    while len(stack) > 0:
        node, pid = stack.pop()
        for child in node.get("children", []):
            cid = add_node(nodes, child, pid)
            edges.add_row(0, tables.sequence_length, pid, cid)
            for mut in child["branch_attrs"]["mutations"].get("nuc", []):
                before = mut[0]
                after = mut[-1]
                pos = int(mut[1:-1])
                mutations[pos].append((cid, before, after))
            stack.append((child, cid))

    for site in sorted(mutations.keys()):
        node, ancestral_state, derived_state = mutations[site][0]
        site_id = tables.sites.add_row(position=site, ancestral_state=ancestral_state)
        tables.mutations.add_row(site=site_id, node=node, derived_state=derived_state)
        for node, _, derived_state in mutations[site][1:]:
            tables.mutations.add_row(
                site=site_id, node=node, derived_state=derived_state
            )

    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()
