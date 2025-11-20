import dataclasses

import tskit

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

# We omit N here as it's mapped to -1. Make "-" the 5th allele
# as this is a valid allele for us.
# NOTE!! This string is also used in the jit module where it's
# hard-coded into a numba function, so if this ever changes
# it needs to be updated there also!
IUPAC_ALLELES = "ACGT-RYSWKMBDHV."

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
    """
    Return the list of FlagValue records set in the given flag mask.

    The input ``f`` is interpreted as a bitmask of tskit and sc2ts-specific
    node flags. Each returned FlagValue describes a single flag,
    including its integer value, short code, long name, and text description.

    :param int f: Integer bitmask of node flags to decode.
    :return: A list of flag descriptors corresponding to the bits set in ``f``.
    :rtype: list
    """
    return [v for v in flag_values if (v.value & f) > 0]


def flags_summary(f):
    """
    Return a compact string summarising the flags set in the given mask.

    Each character in the returned string corresponds to a known flag; set
    flags are shown using their short code and unset flags as ``\"_\"``.

    :param int f: Integer bitmask of node flags to summarise.
    :return: Summary string showing short codes for set flags.
    :rtype: str
    """
    return "".join([v.short if (v.value & f) > 0 else "_" for v in flag_values])
