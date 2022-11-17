import pathlib

import pyfasta
import numpy as np

def get_reference_sequence():

    # NEED packagedata etc.
    data_path = pathlib.Path("sc2ts/data")

    ref_fasta = pyfasta.Fasta(
        str(data_path / "reference.fasta"), record_class=pyfasta.MemoryRecord
    )
    a = np.array(ref_fasta["MN908947 (Wuhan-Hu-1/2019)"]).astype(str)
    reference = np.append(["X"], a)
    return reference
