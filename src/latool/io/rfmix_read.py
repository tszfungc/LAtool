import logging

import numpy as np
import xarray as xr


def read_rfmix_fb(
    fname: str,
) -> xr.Dataset:
    """Reader for RFMIX .fb.tsv output

    Args:
        fname: Path to RFMIX output

    Return:
        Dataset containing local ancestry

    Example
    -------
    >>> from latool.io import read_rfmix_fb
    >>> ds = read_rfmix_fb("tests/testdata/example.fb.tsv")
    >>> ds
    <xarray.Dataset>
    Dimensions:   (marker: 8, sample: 39, ploidy: 2, ancestry: 2)
    Coordinates:
      * marker    (marker) int64 1 6 12 20 25 31 36 43
      * sample    (sample) <U6 'HCB182' 'HCB190' 'HCB191' ... 'JPT266' 'JPT267'
      * ploidy    (ploidy) int8 0 1
      * ancestry  (ancestry) <U3 'HCB' 'JPT'
    Data variables:
        locanc    (marker, sample, ploidy, ancestry) float32 1.0 0.0 1.0 ... 0.0 1.0

    """

    # Read ancestry line
    f_handle = open(fname, "r")
    comment = f_handle.readline()
    pops = comment.strip().split("\t")[1:]
    n_pops = len(pops)

    # header line
    # RFMIX output dim: (marker by (sample x ploidy x ancestry))
    header = f_handle.readline()
    indiv = list(map(lambda x: x.split(":::")[0], header.strip().split("\t")[4:]))
    indiv = np.array(indiv[:: (2 * n_pops)], dtype=str)
    N = indiv.shape[0]

    # data lines
    # create an (marker x sample x ploidy) by (ancestry) DataFrame, then xarray
    chrom = None
    pos = []

    LA_matrix = []  # read into (marker by (sample x ploidy x ancestry))
    for i, line in enumerate(f_handle):
        if i % 100 == 0:
            logging.info(f"processing {i}-th marker")

        line_split = line.strip().split("\t")
        LA_matrix.append(line_split[4:])

        pos.append(int(line_split[1]))

        if chrom is None:
            chrom = int(line_split[0])

    LA_matrix = np.float32(LA_matrix).reshape(-1, N, 2, n_pops)

    ds = xr.Dataset(
        data_vars={
            "locanc": (["marker", "sample", "ploidy", "ancestry"], LA_matrix),
        },
        coords={
            "marker": pos,
            "sample": np.array(indiv, dtype=str),
            "ploidy": np.array([0, 1], dtype=np.int8),
            "ancestry": np.array(pops, dtype=str),
        },
    )

    return ds
