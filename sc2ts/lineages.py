import tskit
import tszip

import json
import numpy as np
from collections import defaultdict
import sklearn
from sklearn import tree
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time

import sys
sys.path.append("../")
import sc2ts.utils

class MutationContainer:

    def __init__(self):
        self.names = {}
        self.positions = []
        self.alts = []
        self.size = 0
        self.all_positions = set()

    def add_root(self,
                 root_lineage_name):
        self.names[root_lineage_name] = self.size
        self.size += 1
        self.positions.append([])
        self.alts.append([])

    def add_item(self,
                    item,
                    position,
                    alt):
        if item not in self.names:
            self.names[item] = self.size
            self.positions.append([position])
            self.alts.append([alt])
            self.size += 1
        else:
            index = self.names[item]
            self.positions[index].append(position)
            self.alts[index].append(alt)
        if position not in self.all_positions:
            self.all_positions.add(position)

    def get_mutations(self,
                      item):
        index = self.names[item]
        return self.positions[index], self.alts[index]


def read_in_mutations(json_filepath):

    with open(json_filepath, "r") as file:
        linmuts = json.load(file)

    # Read in lineage defining mutations
    linmuts_dict = MutationContainer()
    linmuts_dict.add_root('B')
    check_multiallelic_sites = defaultdict(set) # will check how many multi-allelic sites there are

    for item in linmuts:
        if item['alt'] != "-" and item['ref'] != "-": # ignoring indels
            linmuts_dict.add_item(item['name'], item['pos'], item['alt'])
            check_multiallelic_sites[item['pos']].add(item['ref'])
            check_multiallelic_sites[item['pos']].add(item['alt'])

    multiallelic_sites_count = 0
    for value in check_multiallelic_sites.values():
        if len(value) > 2:
            multiallelic_sites_count += 1
    print("Multiallelic sites:", multiallelic_sites_count, "out of", len(check_multiallelic_sites))
    print("Number of lineages:", linmuts_dict.size)

    return linmuts_dict


# One hot encoder using pandas get_dummies() (seems much faster than scikit OneHotEncoder)
class OHE_transform():

    def __init__(self):
        self.new_colnames = None
        self.old_colnames = None

    def fit(self, X):
        self.old_colnames = X.columns
        X = pd.get_dummies(X, drop_first=True)
        self.new_colnames = X.columns
        return X

    def transform(self, X):
        X = pd.get_dummies(X)
        X = X.reindex(columns=self.new_colnames, fill_value=0)
        return X


def read_in_mutations_json(json_filepath):
    df = pd.read_json(json_filepath)
    df = df.loc[df['ref'] != '-']
    df = df.loc[df['alt'] != '-']
    df = df.pivot_table(index='name', columns='pos', values='alt', aggfunc='min', fill_value='.')
    idx = df.index.append(pd.Index(['B']))
    df = df.reindex(idx, fill_value = '.')
    ohe = OHE_transform()
    df_ohe = ohe.fit(df)
    return df, df_ohe, ohe


# Dictionary of {node : [(pos, alt) of all mutations just above this node]}
def get_node_to_mut_dict(ts, ti, linmuts_dict):
    node_to_mut_dict = MutationContainer()
    with tqdm(total=ts.num_mutations) as pbar:
        for m in ts.mutations():
            pos = int(ts.site(m.site).position)
            if pos in linmuts_dict.all_positions:
                alt = ti.mutations_inherited_state[m.id]
                node_to_mut_dict.add_item(m.node, pos, alt)
            pbar.update(1)
    return node_to_mut_dict


class InferLineage:
    
    def __init__(self,
                 num_nodes):
        self.lineages_true = [None] * num_nodes
        self.lineages_pred = [None] * num_nodes
        self.num_nodes = num_nodes
        self.lineages_type = [0] * num_nodes # 0 if can't infer, 1 if inherited, 2 if imputed
        self.num_sample_imputed = 0
        self.num_intern_imputed = 1 # This is the root node which I'm taking to be lineage B
        self.lineages_pred[0] = 'B'
        self.lineages_type[0] = 1
        self.change = 1
        self.current_node = None
        self.linfound = False
        
    def reset(self):
        self.change = 0

    def total_inferred(self,
                       ti):
        return self.num_sample_imputed + self.num_intern_imputed
        
    def set_node(self,
                 node):
        self.current_node = node
        self.linfound = False

    def add_imputed_values(self,
                           X_index,
                           y):
        for ind, pred in zip(X_index, y):
            self.lineages_pred[ind] = pred
            self.lineages_type[ind] = 2
            if self.lineages_true[ind] is not None:
                self.num_sample_imputed += 1
            else:
                self.num_intern_imputed += 1
            self.change += 1
        
    def record_recombinants(self,
                            ts,
                            ti):
        for r in ti.recombinants:
            r_node = ts.node(r)
            if 'Nextclade_pango' not in r_node.metadata:
                # Just recording that this is a recombinant lineage for which we don't have a Pango name
                self.lineages_pred[r] = 'Recombinant'
        
    def record_true_lineage(self,
                            node):
        if 'Nextclade_pango' in node.metadata and self.lineages_true[node.id] is None:
            self.lineages_true[node.id] = node.metadata['Nextclade_pango']
            
    def inherit_from_node(self,
                          node):
        if 'Nextclade_pango' in node.metadata:
            self.lineages_pred[self.current_node.id] = node.metadata['Nextclade_pango']
            self.lineages_type[self.current_node.id] = 1
            self.linfound = True
        elif self.lineages_pred[node.id] is not None:
            self.lineages_pred[self.current_node.id] = self.lineages_pred[node.id]
            self.lineages_type[self.current_node.id] = 1
            self.linfound = True
            
    def inherit_from_children(self,
                              ts,
                              t,
                              mut_dict):
        if not self.linfound:
            for child_node_ind in t.children(self.current_node.id):
                if child_node_ind not in mut_dict.names:
                    child_node = ts.node(child_node_ind)
                    self.inherit_from_node(child_node)
                    if self.linfound:
                        break
                
    def inherit_from_parent(self,
                            ts,
                            t, 
                            mut_dict):
        if not self.linfound:
            if self.current_node.id not in mut_dict.names:
                parent_node_ind = t.parent(self.current_node.id)
                if parent_node_ind != -1:
                    self.inherit_from_node(ts.node(parent_node_ind))
    
    def update(self):
        if self.linfound:
            if self.current_node.is_sample():
                self.num_sample_imputed += 1
            else:
                self.num_intern_imputed += 1
            self.change += 1
            
    def check_node(self,
                   node,
                   ti):
        self.set_node(node)
        if self.current_node.id not in ti.recombinants and self.lineages_pred[self.current_node.id] is None:
            return True
        else:
            return False
                
    def print_info(self,
                   ts,
                   ti,
                   internal_only,
                   target):
        print("-"*30)
        if internal_only:
            target_samples = 0
        else:
            target_samples = ts.num_samples
        print("Sample nodes imputed:", self.num_sample_imputed, "out of possible", target_samples)
        print("Internal nodes imputed:", self.num_intern_imputed, "out of possible", target - target_samples)
        print("Total imputed:", self.num_sample_imputed + self.num_intern_imputed, "out of possible", target)
        print("Number of recombinants (not imputed):", len(ti.recombinants))

        print("-"*30)
        correct = incorrect = 0
        type1 = type2 = 0
        for lt, lp, ltype in zip(self.lineages_true, self.lineages_pred, self.lineages_type):
            if ltype == 1:
                type1 += 1
            elif ltype == 2:
                type2 += 1
            if not internal_only:
                if lt is not None and lp != 'Recombinant':
                    if lt == lp:
                        correct += 1
                    else:
                        incorrect += 1
        if not internal_only:
            print("Correctly imputed samples:", correct, "(", round(100 * correct / (correct + incorrect), 3), "% )")
            print("Incorrectly imputed samples:", incorrect, "(", round(100 * incorrect / (correct + incorrect), 3), "% )")
        print("Imputed using inheritance:", type1, "(", round(100 * type1/(self.total_inferred(ti)), 3), "% )",
              "decision tree:", type2, "(", round(100 * type2/(self.total_inferred(ti)), 3), "% )")
        print("-"*30)

    def get_results(self):
        all_lineages = [None] * self.num_nodes
        for i, (lt, lp) in enumerate(zip(self.lineages_true, self.lineages_pred)):
            if lt is not None:
                all_lineages[i] = lt
            else:
                all_lineages[i] = lp
        return all_lineages


def impute_lineages(ts, ti, linmuts_dict, df, ohe_encoder, clf_tree, internal_only = False):

    tic = time.time()

    inferred_lineages = InferLineage(ts.num_nodes)
    t = ts.first()

    print("Recording relevant mutations for each node...")
    node_to_mut_dict = get_node_to_mut_dict(ts, ti, linmuts_dict)

    # Assigning "Recombinant" as the lineage for recombinant nodes that don't have a Pango designation
    inferred_lineages.record_recombinants(ts, ti)

    for n in ts.nodes():
        inferred_lineages.record_true_lineage(n)

    if internal_only:
        target = len([n for n in ts.nodes() if n.id not in ti.recombinants and not n.is_sample()])
    else:
        target = ts.num_nodes - len(ti.recombinants)

    print("Inferring lineages...")
    with tqdm(total = target - 1) as pbar:
        while inferred_lineages.total_inferred(ti) < target:
            impute_lineages_inheritance(inferred_lineages, ts, t, ti, node_to_mut_dict, linmuts_dict, df, ohe_encoder, clf_tree, internal_only, pbar)
            impute_lineages_decisiontree(inferred_lineages, ts, t, ti, node_to_mut_dict, linmuts_dict, df, ohe_encoder, clf_tree, internal_only, target, pbar)
            # print("Imputed so far:", inferred_lineages.num_sample_imputed + inferred_lineages.num_intern_imputed, "out of possible", target)
    inferred_lineages.print_info(ts, ti, internal_only, target)

    edited_ts = add_lineages_to_ts(inferred_lineages, ts)

    print("Time:", time.time() - tic)

    return inferred_lineages, edited_ts


def impute_lineages_inheritance(inferred_lineages, ts, t, ti, node_to_mut_dict, linmuts_dict, df, ohe_encoder, clf_tree, internal_only, pbar):

    # print("Inheriting lineages...", end="")
    # Need to loop through until all known lineages have been copied where possible
    while inferred_lineages.change:
        inferred_lineages.reset()
        for n_ in t.nodes(order="timedesc"):
            n = ts.node(n_)
            if not internal_only or (internal_only and not n.is_sample()):
                if inferred_lineages.check_node(n, ti):
                    # Try to inherit lineage from parent or children, if there is at least one edge
                    # without a mutation
                    inferred_lineages.inherit_from_children(ts, t, node_to_mut_dict)
                    inferred_lineages.inherit_from_parent(ts, t, node_to_mut_dict)
                    inferred_lineages.update()
        # print(inferred_lineages.change, end="...")
        pbar.update(inferred_lineages.change)
    # print("done")


def impute_lineages_decisiontree(inferred_lineages, ts, t, ti, node_to_mut_dict, linmuts_dict, df, ohe_encoder, clf_tree, internal_only, target, pbar):

    # Impute lineages for the rest of the nodes where possible (one pass)
    X = pd.DataFrame(index=range(target - inferred_lineages.total_inferred(ti)), columns=df.columns)
    X_index = np.zeros(target - inferred_lineages.total_inferred(ti), dtype=int)
    ind = 0
    # print("Imputing lineages...", end = "")
    inferred_lineages.reset()
    for n_ in t.nodes(order="timedesc"):
        n = ts.node(n_)
        if not internal_only or (internal_only and not n.is_sample()):
            if inferred_lineages.check_node(n, ti):
                parent_node_ind = t.parent(inferred_lineages.current_node.id)
                if parent_node_ind != -1:
                    parent_node_md = ts.node(parent_node_ind).metadata
                    if 'Nextclade_pango' in parent_node_md or inferred_lineages.lineages_pred[parent_node_ind] is not None:
                        # Check if we can now copy the parent's lineage
                        if n_ not in node_to_mut_dict.names or ('Nextclade_pango' not in parent_node_md and inferred_lineages.lineages_pred[parent_node_ind] == 'Recombinant'):
                            inferred_lineages.inherit_from_node(ts.node(parent_node_ind))
                            inferred_lineages.update()
                        # If not, then add to dataframe for imputation
                        else:
                            if 'Nextclade_pango' in parent_node_md:
                                parent_lineage = parent_node_md['Nextclade_pango']
                            else:
                                parent_lineage = inferred_lineages.lineages_pred[parent_node_ind]
                            X_index[ind] = n_
                            X.loc[ind] = df.loc[parent_lineage]
                            positions, alts = node_to_mut_dict.get_mutations(n_)
                            X.loc[ind][positions] = alts
                            ind += 1
    if ind > 0:
        X = X.iloc[0:ind]
        X_index = X_index[0:ind]
        y = clf_tree.predict(ohe_encoder.transform(X))
        inferred_lineages.add_imputed_values(X_index, y)
    pbar.update(inferred_lineages.change)
    # print(str(inferred_lineages.change) + "...done")


def add_lineages_to_ts(il, ts):
    imputed_lineages = il.get_results()
    tables = ts.tables
    new_metadata = []
    for node in ts.nodes():
        md = node.metadata
        md['Imputed_lineage'] = imputed_lineages[node.id]
        new_metadata.append(md)
    validated_metadata = [tables.nodes.metadata_schema.validate_and_encode_row(row) for row in new_metadata]
    tables.nodes.packset_metadata(validated_metadata)
    edited_ts = tables.tree_sequence()
    return edited_ts


