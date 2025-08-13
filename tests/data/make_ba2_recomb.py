import tskit
import numpy as np
import pandas as pd

import pathlib

prefix = pathlib.Path("/home/jk/work/github/sc2ts-paper")

orig_ts = tskit.load(prefix / "inference/results/v1-beta1/v1-beta1_2023-02-21.ts")
recomb_df = pd.read_csv(prefix / "data/recombinants.csv").set_index("sample_id")
strain2nodeid = dict(zip(orig_ts.metadata['sc2ts']['samples_strain'], orig_ts.samples()))
ba_2_sample = "SRR17461792"
re_node = recomb_df.loc[ba_2_sample, 'recombinant']
re_node_children = orig_ts.edges_child[orig_ts.edges_parent == re_node]
keep = np.concatenate(([0, 1], [strain2nodeid[ba_2_sample]], [re_node], re_node_children))
if len(re_node_children) == 1:
    # Add a few more grandchildren
    re_node_grandchildren = orig_ts.edges_child[orig_ts.edges_parent == re_node_children[0]]
    keep = np.concatenate((keep, re_node_grandchildren[0:3]))

keep = set(keep)
tree = orig_ts.first()
keep.update(set(tree.ancestors(re_node)))
tree = orig_ts.last()
keep.update(set(tree.ancestors(re_node)))

keep = sorted(keep, key=lambda u: -orig_ts.nodes_time[u])
test_ts, node_map = orig_ts.simplify(keep, update_sample_flags=False, map_nodes=True)


tables = test_ts.dump_tables()
tables.provenances.clear()
tables.metadata = {}  # no need to keep the ~20Mb of metadata
for node in test_ts.nodes():
    md = node.metadata
    if "hmm_match" in md["sc2ts"]:
        hmm_match = md["sc2ts"]["hmm_match"]
        for d in hmm_match["path"]:
            d["parent"] = int(node_map[d["parent"]])
        row = tables.nodes[node.id]
        tables.nodes[node.id] = row.replace(metadata=md)
    # print(md)
test_ts = tables.tree_sequence()
test_ts.dump("tests/data/ba2_recomb.ts")
