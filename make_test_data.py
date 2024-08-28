# Script to make the test subset data.
import contextlib
import sys

import pandas as pd

import sc2ts


def write_fasta(strain, alignment, out):
    s = bytes(alignment.astype("S1").data).decode()
    print(f">{strain}", file=out)
    print(s[1:], file=out)


def main(alignment_store, metadata_db, output_prefix):
    dates = metadata_db.get_days()
    alignment_path = output_prefix + "/alignments.fasta"
    metadata_path = output_prefix + "/metadata.tsv"
    metadata = []
    with open(alignment_path, "w") as alignment_file:
        strains = []
        for date in dates[:20]:
            records = list(metadata_db.get(date))
            for record in records[:5]:
                strain = record["strain"]
                alignment = alignment_store[strain]
                write_fasta(strain, alignment, alignment_file)
                metadata.append(record)

    # Add a record for a missing sequence.
    record = dict(record)
    record["strain"] = "ERR_MISSING"
    metadata.append(record)

    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, sep="\t", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError(
            "Usage: python3 make_test_data.py [alignment_db] [metadata_db] [prefix]"
        )
    with contextlib.ExitStack() as exit_stack:
        alignment_store = exit_stack.enter_context(sc2ts.AlignmentStore(sys.argv[1]))
        metadata_db = exit_stack.enter_context(sc2ts.MetadataDb(sys.argv[2]))
        main(alignment_store, metadata_db, sys.argv[3])
