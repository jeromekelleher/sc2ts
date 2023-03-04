"""
Utilities for converting a Nextstrain tree to tskit format
and comparing with an sc2ts output.
"""

# NOTE this is a random collection of bits of code, should be
# reviewed and cleared up as the analysis progresses.
import collections
import json
import re

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


def load_nextclade_json(path):
    """
    Read a json format nextclade file, e.g. from
    https://raw.githubusercontent.com/nextstrain/nextclade_data/release/data/datasets/sars-cov-2/references/MN908947/versions/2021-06-25T00:00:00Z/files/tree.json
    into a tree sequence
    """
    with open(path, "rt") as file:
        ts = convert_nextclade(json.load(file))
    return ts


def convert_nextclade(document):
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


def subset_to_intersection(tssc, tsnt, filter_sites=True, **kwargs):
    """
    Returns the subset of the two tree sequences for the set of sample strains
    in both.
    
    **kwargs are sent to the `simplify` commands used. Note that if
    `filter_nodes` is used, the samples in the returned subsets may not
    have the same nodes IDs or even be in the same order.
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

    tss1 = tssc.simplify(
        sc_samples + recombinants,
        filter_sites=filter_sites,
        **kwargs,
    )
    tss2 = tsnt.simplify(
        [strain_map2[key] for key in intersection],
        filter_sites=filter_sites,
        **kwargs,
    )
    if filter_sites:
        site_intersection = set(tss1.sites_position) & set(tss2.sites_position)
        tss1 = keep_sites(tss1, site_intersection)
        tss2 = keep_sites(tss2, site_intersection)
    return tss1, tss2


def get_nextclade_intersection(nextclade_file, ts_sc2ts):

    ts_ns = load_nextclade_json(nextclade_file)
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


def newick_from_nextstrain_with_comments(tree_string, **kwargs):
    """
    Take a newick string from a nextstrain file with embedded comments (usually this
    is extracted from a nexus file). Comments look like:
    '&clade_membership=19A,num_date=2019.98,num_date_CI={2019.87,2019.98},emerging_lineage=19A,current_frequency=0.9,region=Asia,div=0'
    These are parsed into a python dict and saved in the "comment" metadata value::
        "comment": {
            "clade_membership"="19A",
            "num_date"="2019.98",
            "num_date_CI"="{2019.87,2019.98}",
            "emerging_lineage"="19A",
            "current_frequency"="0.9",
            "region"="Asia",
            "div"="0",
        }
    We assume the time-scale is in years-ago, and convert to days-ago
    """
    prefix = "hCoV-19/"  # the node names may have this prepended, which can be removed

    # The nexus file sometimes has round braces within the [] comment string.
    # Strip these out with an ugly hack
    tidied_string = re.sub(
        r"\[.*?\]",
        lambda match: match.group(0).replace("(", "").replace(")", ""),
        tree_string,
    )
    #
    # Import this here to avoid needing to install tsconvert (which isn't on pip)
    import tsconvert

    ts = tsconvert.from_newick(
        tidied_string,
        node_name_key="strain",
        **kwargs,
    )

    # Split the comment string into a dictionary for each node
    metadata = [n.metadata for n in ts.nodes()]
    new_metadata = []
    for m in metadata:
        if "comment" in m:
            parsed_comment = {
                match.group(1).lstrip("&"): match.group(2)
                for match in re.finditer(
                    "([^=]+)=([^=]+),(?=[^=]+?(=|$))", m["comment"]
                )
            }
            m["comment"] = parsed_comment
        if "strain" in m and m["strain"].startswith(prefix):
            m["strain"] = m["strain"][len(prefix) :]
        new_metadata.append(m)
    # inject the new node metadata back into the tree sequence
    tables = ts.dump_tables()
    tables.nodes.metadata_schema = tskit.metadata.MetadataSchema(
        {
            "codec": "json",
            "properties": {
                "comment": {
                    "description": "Comment from newick file",
                    "type": ["object"],
                },
                "strain": {"description": "Name from newick file", "type": ["string"]},
            },
            "type": "object",
        }
    )
    tables.nodes.packset_metadata(
        [tables.nodes.metadata_schema.validate_and_encode_row(r) for r in new_metadata]
    )

    # Check it looks like we are measuring time in years
    if tables.nodes.time.max() - tables.nodes.time.min() > 20:
        raise ValueError(
            "Timescale assumed in years but this nextstrain tree covers > 20 time units"
        )
    tables.nodes.time = tables.nodes.time * 365  # TODO? correct for leap years?
    tables.time_units = "days"

    return tables.tree_sequence()


def extract_newick_from_nextstrain_nexus(path):
    """
    Nextstrain nexus files are the ones with clade membership annotations in the comments
    fields. This function extracts the newick string from the nexus file.

    These can be downloaded from e.g. https://nextstrain.org/ncov/gisaid/global/all-time
    """
    with open(path) as file:
        tree_line = ""
        for line in file:
            if "begin trees;" in line:
                tree_line = next(file)
                assert "end;" in next(file)
                break
        loc = tree_line.find("=")
        tree_line = tree_line[loc + 1 :].strip()
        if loc < 0 or len(tree_line) == 0 or tree_line[-1] != ";":
            raise ValueError("Can't find a valid tree line in newick format")
        return tree_line

ns_clade_colours = {
    "20H Beta V2": "#3F47C9",
    "20I Alpha V1": "#4274CE",
    "20J Gamma V3": "#4F97BB",
    "21A Delta": "#64AC99",
    "21B Kappa": "#7EB976",
    "21C Epsilon": "#9EBE5A",
    "21D Eta": "#BEBB48",
    "21E Theta": "#D9AE3E",
    "21F Iota": "#E69036",
    "21G Lambda": "#E35F2D",
    "21H": "#DB2823",
}

pango_colours = {
    "A": "#5E1D9D",
    "A.2": "#5C1D9E",
    "A.2.2": "#5A1DA0",
    "A.2.5.1": "#581DA2",
    "A.2.5.2": "#571EA3",
    "A.3": "#551EA5",
    "A.5": "#531EA6",
    "A.11": "#511EA8",
    "A.12": "#4F1EAA",
    "A.21": "#4F1FAB",
    "A.23.1": "#4E21AC",
    "A.27": "#4D22AD",
    "A.28": "#4C23AE",
    "A.29": "#4C24B0",
    "AA.1": "#4B26B1",
    "AD.2": "#4A27B2",
    "B": "#4A28B3",
    "B.1": "#4929B5",
    "B.1.1": "#482BB6",
    "B.1.1.1": "#472CB7",
    "B.1.1.7": "#472DB8",
    "B.1.1.8": "#462FB9",
    "B.1.1.10": "#4530BB",
    "B.1.1.16": "#4531BC",
    "B.1.1.25": "#4432BD",
    "B.1.1.26": "#4334BE",
    "B.1.1.28": "#4335C0",
    "B.1.1.34": "#4236C1",
    "B.1.1.37": "#4137C2",
    "B.1.1.38": "#4039C3",
    "B.1.1.39": "#403AC4",
    "B.1.1.44": "#403CC5",
    "B.1.1.47": "#403DC5",
    "B.1.1.50": "#403FC6",
    "B.1.1.70": "#4040C6",
    "B.1.1.71": "#4042C7",
    "B.1.1.89": "#3F43C8",
    "B.1.1.93": "#3F45C8",
    "B.1.1.109": "#3F47C9",
    "B.1.1.115": "#3F48C9",
    "B.1.1.121": "#3F4ACA",
    "B.1.1.122": "#3F4BCA",
    "B.1.1.129": "#3F4DCB",
    "B.1.1.130": "#3F4FCB",
    "B.1.1.141": "#3F50CC",
    "B.1.1.153": "#3F52CD",
    "B.1.1.161": "#3F53CD",
    "B.1.1.165": "#3F55CE",
    "B.1.1.170": "#3F56CE",
    "B.1.1.174": "#3E58CF",
    "B.1.1.186": "#3E5ACF",
    "B.1.1.189": "#3E5BD0",
    "B.1.1.192": "#3E5DD0",
    "B.1.1.207": "#3F5ED0",
    "B.1.1.216": "#3F60D0",
    "B.1.1.217": "#3F61D0",
    "B.1.1.220": "#3F63CF",
    "B.1.1.222": "#4064CF",
    "B.1.1.231": "#4066CF",
    "B.1.1.232": "#4067CF",
    "B.1.1.239": "#4068CF",
    "B.1.1.240": "#416ACF",
    "B.1.1.242": "#416BCE",
    "B.1.1.269": "#416DCE",
    "B.1.1.272": "#416ECE",
    "B.1.1.277": "#4270CE",
    "B.1.1.286": "#4271CE",
    "B.1.1.291": "#4273CE",
    "B.1.1.294": "#4274CD",
    "B.1.1.299": "#4376CD",
    "B.1.1.304": "#4377CD",
    "B.1.1.305": "#4379CD",
    "B.1.1.306": "#447ACD",
    "B.1.1.307": "#447CCD",
    "B.1.1.308": "#447DCC",
    "B.1.1.312": "#457ECB",
    "B.1.1.315": "#457FCB",
    "B.1.1.316": "#4680CA",
    "B.1.1.317": "#4681C9",
    "B.1.1.318": "#4683C9",
    "B.1.1.322": "#4784C8",
    "B.1.1.323": "#4785C7",
    "B.1.1.329": "#4886C6",
    "B.1.1.334": "#4887C6",
    "B.1.1.335": "#4988C5",
    "B.1.1.347": "#498AC4",
    "B.1.1.348": "#4A8BC3",
    "B.1.1.351": "#4A8CC3",
    "B.1.1.359": "#4B8DC2",
    "B.1.1.368": "#4B8EC1",
    "B.1.1.369": "#4C8FC0",
    "B.1.1.372": "#4C90C0",
    "B.1.1.374": "#4D92BF",
    "B.1.1.380": "#4D93BE",
    "B.1.1.384": "#4D94BE",
    "B.1.1.391": "#4E95BD",
    "B.1.1.398": "#4E96BC",
    "B.1.1.406": "#4F97BB",
    "B.1.1.419": "#5098BA",
    "B.1.1.420": "#5098B8",
    "B.1.1.428": "#5199B7",
    "B.1.1.432": "#529AB6",
    "B.1.1.434": "#529BB5",
    "B.1.1.453": "#539CB4",
    "B.1.1.464": "#549DB3",
    "B.1.1.485": "#549DB2",
    "B.1.1.486": "#559EB1",
    "B.1.1.517": "#569FB0",
    "B.1.1.519": "#56A0AF",
    "B.1.1.521": "#57A1AE",
    "B.1.1.523": "#57A1AC",
    "B.1.2": "#58A2AB",
    "B.1.9": "#59A3AA",
    "B.1.9.5": "#59A4A9",
    "B.1.13": "#5AA5A8",
    "B.1.22": "#5BA6A7",
    "B.1.23": "#5BA6A6",
    "B.1.36": "#5CA7A5",
    "B.1.36.1": "#5DA8A4",
    "B.1.36.8": "#5DA8A2",
    "B.1.36.9": "#5EA9A1",
    "B.1.36.10": "#5FA9A0",
    "B.1.36.16": "#60AA9F",
    "B.1.36.17": "#61AA9D",
    "B.1.36.23": "#62AB9C",
    "B.1.36.24": "#62AB9B",
    "B.1.36.27": "#63AC9A",
    "B.1.36.31": "#64AD98",
    "B.1.36.35": "#65AD97",
    "B.1.37": "#66AE96",
    "B.1.40": "#66AE95",
    "B.1.76": "#67AF94",
    "B.1.77": "#68AF92",
    "B.1.91": "#69B091",
    "B.1.96": "#6AB090",
    "B.1.104": "#6AB18F",
    "B.1.110": "#6BB18D",
    "B.1.110.3": "#6CB28C",
    "B.1.111": "#6DB28B",
    "B.1.124": "#6EB38A",
    "B.1.126": "#6FB389",
    "B.1.128": "#70B487",
    "B.1.131": "#71B486",
    "B.1.134": "#72B485",
    "B.1.139": "#73B584",
    "B.1.149": "#74B583",
    "B.1.160": "#75B582",
    "B.1.160.9": "#75B680",
    "B.1.160.11": "#76B67F",
    "B.1.160.14": "#77B67E",
    "B.1.160.16": "#78B77D",
    "B.1.160.20": "#79B77C",
    "B.1.160.22": "#7AB77B",
    "B.1.160.30": "#7BB879",
    "B.1.160.31": "#7CB878",
    "B.1.160.32": "#7DB877",
    "B.1.177": "#7EB976",
    "B.1.177.4": "#7FB975",
    "B.1.177.5": "#80B974",
    "B.1.177.7": "#81BA72",
    "B.1.177.8": "#82BA71",
    "B.1.177.15": "#83BA70",
    "B.1.177.17": "#84BA6F",
    "B.1.177.18": "#85BA6E",
    "B.1.177.19": "#86BB6D",
    "B.1.177.20": "#88BB6C",
    "B.1.177.23": "#89BB6B",
    "B.1.177.43": "#8ABB6A",
    "B.1.177.44": "#8BBB69",
    "B.1.177.47": "#8CBB68",
    "B.1.177.48": "#8DBC68",
    "B.1.177.52": "#8EBC67",
    "B.1.177.55": "#8FBC66",
    "B.1.177.56": "#90BC65",
    "B.1.177.57": "#91BC64",
    "B.1.177.58": "#93BC63",
    "B.1.177.60": "#94BD62",
    "B.1.177.62": "#95BD61",
    "B.1.177.63": "#96BD60",
    "B.1.177.64": "#97BD5F",
    "B.1.177.73": "#98BD5E",
    "B.1.177.75": "#99BD5D",
    "B.1.177.81": "#9ABE5C",
    "B.1.177.82": "#9BBE5B",
    "B.1.177.83": "#9DBE5A",
    "B.1.177.85": "#9EBE5A",
    "B.1.190": "#9FBE59",
    "B.1.199": "#A0BE58",
    "B.1.201": "#A1BE58",
    "B.1.210": "#A2BE57",
    "B.1.214.2": "#A3BE56",
    "B.1.214.3": "#A5BE56",
    "B.1.221": "#A6BE55",
    "B.1.221.1": "#A7BE54",
    "B.1.234": "#A8BD54",
    "B.1.235": "#A9BD53",
    "B.1.236": "#AABD52",
    "B.1.240": "#ABBD51",
    "B.1.241": "#ACBD51",
    "B.1.243": "#AEBD50",
    "B.1.243.1": "#AFBD4F",
    "B.1.258": "#B0BD4F",
    "B.1.258.3": "#B1BD4E",
    "B.1.258.7": "#B2BD4D",
    "B.1.258.14": "#B3BD4D",
    "B.1.258.17": "#B4BD4C",
    "B.1.264.1": "#B5BD4C",
    "B.1.268": "#B6BD4B",
    "B.1.276": "#B8BC4B",
    "B.1.280": "#B9BC4A",
    "B.1.289": "#BABC4A",
    "B.1.298": "#BBBC49",
    "B.1.305": "#BCBB49",
    "B.1.306": "#BDBB48",
    "B.1.311": "#BEBB48",
    "B.1.319": "#BFBB48",
    "B.1.320": "#C0BA47",
    "B.1.324": "#C1BA47",
    "B.1.337": "#C2BA46",
    "B.1.338": "#C3BA46",
    "B.1.346": "#C4B945",
    "B.1.349": "#C5B945",
    "B.1.351": "#C6B944",
    "B.1.351.2": "#C7B944",
    "B.1.351.3": "#C8B844",
    "B.1.356": "#C9B843",
    "B.1.361": "#CAB843",
    "B.1.362": "#CBB742",
    "B.1.369": "#CCB742",
    "B.1.375": "#CDB642",
    "B.1.378": "#CEB641",
    "B.1.382": "#CEB541",
    "B.1.384": "#CFB541",
    "B.1.389": "#D0B441",
    "B.1.390": "#D1B340",
    "B.1.396": "#D2B340",
    "B.1.397": "#D2B240",
    "B.1.399": "#D3B23F",
    "B.1.400": "#D4B13F",
    "B.1.404": "#D5B03F",
    "B.1.406": "#D6B03F",
    "B.1.413": "#D6AF3E",
    "B.1.416.1": "#D7AF3E",
    "B.1.420": "#D8AE3E",
    "B.1.427": "#D9AD3D",
    "B.1.429": "#DAAD3D",
    "B.1.433": "#DAAC3D",
    "B.1.438": "#DBAC3D",
    "B.1.438.1": "#DCAB3C",
    "B.1.443": "#DDAA3C",
    "B.1.452": "#DDA93C",
    "B.1.453": "#DDA83C",
    "B.1.466.2": "#DEA73B",
    "B.1.480": "#DEA63B",
    "B.1.498": "#DFA53B",
    "B.1.499": "#DFA43B",
    "B.1.503": "#E0A33A",
    "B.1.509": "#E0A23A",
    "B.1.517": "#E1A13A",
    "B.1.521": "#E1A03A",
    "B.1.523": "#E19F3A",
    "B.1.525": "#E29E39",
    "B.1.526": "#E29D39",
    "B.1.545": "#E39C39",
    "B.1.546": "#E39B39",
    "B.1.550": "#E49A38",
    "B.1.551": "#E49938",
    "B.1.552": "#E49838",
    "B.1.561": "#E59738",
    "B.1.564": "#E59637",
    "B.1.565": "#E69537",
    "B.1.568": "#E69337",
    "B.1.575": "#E69237",
    "B.1.577": "#E69036",
    "B.1.582": "#E68F36",
    "B.1.587": "#E68D36",
    "B.1.588": "#E68C35",
    "B.1.590": "#E68A35",
    "B.1.595": "#E68935",
    "B.1.595.3": "#E68735",
    "B.1.596": "#E68634",
    "B.1.600": "#E68434",
    "B.1.603": "#E68334",
    "B.1.605": "#E68133",
    "B.1.609": "#E68033",
    "B.1.612": "#E67E33",
    "B.1.617.1": "#E67D33",
    "B.1.617.2": "#E67B32",
    "B.1.619": "#E67932",
    "B.1.620": "#E67832",
    "B.1.621": "#E67631",
    "B.1.623": "#E67531",
    "B.3": "#E67331",
    "B.4": "#E67231",
    "B.4.5": "#E67030",
    "B.6": "#E56E30",
    "B.6.6": "#E56C2F",
    "B.13": "#E56A2F",
    "B.23": "#E4682F",
    "B.28": "#E4662E",
    "B.29": "#E4642E",
    "B.31": "#E4632E",
    "B.40": "#E3612D",
    "B.43": "#E35F2D",
    "B.46": "#E35D2C",
    "B.51": "#E25B2C",
    "B.58": "#E2592C",
    "C.3": "#E2572B",
    "C.11": "#E2552B",
    "C.13": "#E1532B",
    "C.16": "#E1522A",
    "C.23": "#E1502A",
    "C.25": "#E04E2A",
    "C.26": "#E04C29",
    "C.27": "#E04A29",
    "C.31": "#E04828",
    "C.35": "#DF4628",
    "C.36": "#DF4428",
    "C.36.2": "#DF4327",
    "C.37": "#DF4127",
    "D.2": "#DE3F27",
    "D.5": "#DE3D26",
    "L.3": "#DE3B26",
    "N.4": "#DE3926",
    "N.5": "#DD3725",
    "P.1": "#DD3525",
    "P.1.1": "#DD3325",
    "P.2": "#DD3225",
    "P.3": "#DC3024",
    "R.1": "#DC2E24",
    "R.2": "#DC2C24",
    "W.1": "#DC2A23",
    "W.4": "#DB2823",
}
