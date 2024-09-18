import numpy as np
import tskit

import sc2ts


def get_match_db(ts, db_path, samples, date, num_mismatches):
    sc2ts.MatchDb.initialise(db_path)
    match_db = sc2ts.MatchDb(db_path)
    match_db.add(samples, date, num_mismatches)
    match_db.create_mask_table(ts)
    return match_db


def example_binary(n, date="2020-01-01"):
    base = sc2ts.initial_ts()
    tables = base.dump_tables()
    tree = tskit.Tree.generate_balanced(n, span=base.sequence_length)
    binary_tables = tree.tree_sequence.dump_tables()
    binary_tables.nodes.time += 1
    tables.nodes.time += np.max(binary_tables.nodes.time) + 1
    binary_tables.edges.child += len(tables.nodes)
    binary_tables.edges.parent += len(tables.nodes)
    for j, node in enumerate(binary_tables.nodes):
        md = {}
        if node.flags == tskit.NODE_IS_SAMPLE:
            md["strain"] = f"x{j}"
            md["date"] = date
        tables.nodes.append(node.replace(metadata=md))
    for edge in binary_tables.edges:
        tables.edges.append(edge)
    # FIXME brittle
    tables.edges.add_row(0, base.sequence_length, parent=1, child=tree.root + 2)
    tables.sort()
    return tables.tree_sequence()
