import dataclasses
import json

import tskit
import numpy as np

__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass

TIME_UNITS = "days"

REFERENCE_STRAIN = "Wuhan/Hu-1/2019"
REFERENCE_DATE = "2019-12-26"
REFERENCE_GENBANK = "MN908947"
REFERENCE_SEQUENCE_LENGTH = 29904

NODE_IS_MUTATION_OVERLAP = 1 << 21
NODE_IS_REVERSION_PUSH = 1 << 22
NODE_IS_RECOMBINANT = 1 << 23
NODE_IS_EXACT_MATCH = 1 << 24
NODE_IS_IMMEDIATE_REVERSION_MARKER = 1 << 25
NODE_IS_REFERENCE = 1 << 26
NODE_IS_UNCONDITIONALLY_INCLUDED = 1 << 27


@dataclasses.dataclass(frozen=True)
class FlagValue:
    value: int
    short: str
    long: str
    description: str


flag_values = [
    FlagValue(tskit.NODE_IS_SAMPLE, "S", "Sample", "Tskit defined sample node"),
    FlagValue(
        NODE_IS_MUTATION_OVERLAP,
        "O",
        "MutationOverlap",
        "Node created by coalescing mutations shared by siblings",
    ),
    FlagValue(
        NODE_IS_REVERSION_PUSH,
        "P",
        "ReversionPush",
        "Node created by pushing immediate reversions upwards",
    ),
    FlagValue(
        NODE_IS_RECOMBINANT,
        "R",
        "Recombinant",
        "Node has two or more parents",
    ),
    FlagValue(
        NODE_IS_EXACT_MATCH,
        "E",
        "ExactMatch",
        "Node is an exact match of its parent",
    ),
    FlagValue(
        NODE_IS_IMMEDIATE_REVERSION_MARKER,
        "I",
        "ImmediateReversion",
        "Node is marking the existance of an immediate reversion which "
        "has not been removed for technical reasons",
    ),
    FlagValue(
        NODE_IS_REFERENCE,
        "F",
        "Reference",
        "Node is a reference sequence",
    ),
    FlagValue(
        NODE_IS_UNCONDITIONALLY_INCLUDED,
        "U",
        "UnconditionalInclude",
        "A sample that was flagged for unconditional inclusion",
    ),
]


def decode_flags(f):
    return [v for v in flag_values if (v.value & f) > 0]


def flags_summary(f):
    return "".join([v.short if (v.value & f) > 0 else "_" for v in flag_values])


# We omit N here as it's mapped to -1. Make "-" the 5th allele
# as this is a valid allele for us.
IUPAC_ALLELES = "ACGT-RYSWKMBDHV."
