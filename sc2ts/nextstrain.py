"""
Utilities for converting a Nextstrain tree to tskit format
and comparing with an sc2ts output.
"""

# NOTE this is a random collection of bits of code, should be
# reviewed and cleared up as the analysis progresses.
import collections
import json

import numpy as np
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


def keep_sites(ts, positions):
    delete_sites = []
    # Could do better, but expecting small numbers of sites here.
    for j, pos in enumerate(ts.sites_position):
        if pos not in positions:
            delete_sites.append(j)
    tables = ts.dump_tables()
    tables.delete_sites(delete_sites)
    tables.sort()
    return tables.tree_sequence()


def subset_to_intersection(tssc, tsnt):
    """
    Returns the subset of the two tree sequences for the set of sample strains
    in both.
    """
    assert tsnt.num_trees == 1
    strain_map1 = {tssc.node(u).metadata["strain"]: u for u in tssc.samples()}
    strain_map2 = {tsnt.node(u).metadata["strain"]: u for u in tsnt.samples()}
    intersection = list(set(strain_map1.keys()) & set(strain_map2.keys()))
    # Sort by date
    intersection.sort(key=lambda s: -tssc.nodes_time[strain_map1[s]])

    sc_samples = [strain_map1[key] for key in intersection]
    # Add in any recombinants encountered in the history of leaf samples
    recombinants = set()
    for tree in tssc.trees():
        for sample in sc_samples:
            u = sample
            while u != -1:
                e = tssc.edge(tree.edge(u))
                if not (e.left == 0 and e.right == tssc.sequence_length):
                    recombinants.add(u)
                u = tree.parent(u)
    recombinants = list(recombinants - set(sc_samples))
    recombinants.sort(key=lambda u: -tssc.nodes_time[u])
    # print("Recombs:", recombinants)

    tss1 = tssc.simplify(sc_samples + recombinants)
    tss2 = tsnt.simplify([strain_map2[key] for key in intersection])
    site_intersection = set(tss1.sites_position) & set(tss2.sites_position)
    tss1 = keep_sites(tss1, site_intersection)
    tss2 = keep_sites(tss2, site_intersection)
    return tss1, tss2


def get_nextstrain_intersection(nextstrain_file, ts_sc2ts):

    with open(nextstrain_file) as f:
        d = json.load(f)
        ts_ns = convert_nextstrain(d)
    tss_sc, tss_nt = subset_to_intersection(ts_sc2ts, ts_ns)
    assert tss_sc.num_samples >= tss_nt.num_samples
    assert list(tss_sc.samples()[: tss_nt.num_samples]) == list(tss_nt.samples())
    assert [
        tss_sc.node(u).metadata["strain"] == tss_nt.node(u).metadata["strain"]
        for u in tss_nt.samples()
    ]
    assert np.array_equal(tss_sc.sites_position, tss_nt.sites_position)
    return tss_sc, tss_nt


def get_mutation_path(ts, node):
    """
    Return the path of mutations grouped by depth on the tree.
    """
    # This is a sledgehammer to crack a nut.
    tss = ts.simplify([node], keep_unary=True)
    depth_map = collections.defaultdict(list)
    for tree in tss.trees():
        for site in tree.sites():
            for mut in site.mutations:
                depth = tree.depth(mut.node)
                inherited_state = site.ancestral_state
                if mut.parent != -1:
                    inherited_state = tss.mutation(mut.parent).derived_state
                depth_map[depth].append(
                    (int(site.position), inherited_state, mut.derived_state)
                )
    return depth_map
