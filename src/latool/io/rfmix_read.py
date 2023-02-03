import logging

import numpy as np
import xarray as xr
from numba import guvectorize


@guvectorize(["(uint32[:], uint8[:], uint8[:])"], "(), (n) -> (n)")
def _ohe(value_in, _, arr_out):
    arr_out[:] = 0
    arr_out[value_in] = 1


def read_rfmix_fb(
    fname: str,
) -> xr.Dataset:
    """Reader for RFMIX .fb.tsv output

    | Reading RFMIX .fb.tsv output as probability dosages, and
    | convert it to a xarray dataset

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
    Dimensions:           (marker: 8, sample: 39, ploidy: 2, ancestry: 2)
    Coordinates:
      * marker            (marker) int64 1 6 12 20 25 31 36 43
      * sample            (sample) <U6 'HCB182' 'HCB190' ... 'JPT266' 'JPT267'
      * ploidy            (ploidy) int8 0 1
      * ancestry          (ancestry) <U3 'HCB' 'JPT'
    Data variables:
        locanc            (marker, sample, ploidy, ancestry) float32 1.0 0.0 ... 1.0
        genetic_position  (marker) float32 1e-05 6e-05 0.00012 ... 0.00036 0.00043

    """

    # Read ancestry line
    f_handle = open(fname, "r")
    comment = f_handle.readline()
    pops = comment.strip().split("\t")[1:]
    n_pops = len(pops)

    # header line
    # RFMIX output dim: (marker , (sample x ploidy x ancestry))
    header = f_handle.readline()
    indiv = list(map(lambda x: x.split(":::")[0], header.strip().split("\t")[4:]))
    indiv = np.array(indiv[:: (2 * n_pops)], dtype=str)
    N = indiv.shape[0]

    # data lines
    # Reshape to (marker, sample, ploidy, ancestry) array, then xarray
    chrom = None
    pos = []
    genetic_pos = []

    LA_matrix = []  # read into (marker by (sample x ploidy x ancestry))
    for i, line in enumerate(f_handle):
        if i % 100 == 0:
            logging.info(f"processing {i}-th marker")

        line_split = line.strip().split("\t")
        LA_matrix.append(line_split[4:])

        pos.append(int(line_split[1]))
        if line_split[2] == ".":
            genetic_pos.append(np.nan)
        else:
            genetic_pos.append(line_split[2])

        if chrom is None:
            chrom = int(line_split[0])

    LA_matrix = np.float32(LA_matrix).reshape(-1, N, 2, n_pops)
    genetic_pos = np.float32(genetic_pos)
    pos = np.uint32(pos)

    ds = xr.Dataset(
        data_vars={
            "locanc": (["marker", "sample", "ploidy", "ancestry"], LA_matrix),
            "genetic_position": ("marker", genetic_pos),
        },
        coords={
            "marker": pos,
            "sample": np.array(indiv, dtype=str),
            "ploidy": np.array([0, 1], dtype=np.int8),
            "ancestry": np.array(pops, dtype=str),
        },
    )

    return ds


def read_rfmix_msp(
    fname: str,
) -> xr.Dataset:
    """Reader for RFMIX .msp.tsv output

    Args:
        fname: Path to RFMIX output

    Return:
        Dataset containing local ancestry

    Example
    -------

    """

    # Read ancestry line
    f_handle = open(fname, "r")
    comment = f_handle.readline()
    popcode = comment.strip().split(" ")[-1].split("\t")
    pops = [pc.split("=")[0] for pc in popcode]
    n_pops = len(pops)

    # header line
    # RFMIX output dim: (marker by (sample x ploidy))
    header = f_handle.readline()
    indiv = list(map(lambda x: x[:-2], header.strip().split("\t")[6:]))
    indiv = np.array(indiv[::2], dtype=str)
    N = indiv.shape[0]

    # data lines
    # reshape to (marker, sample, ploidy)
    # one hot encode to expand entry to (ancestry, )
    chrom = None
    lpos, rpos, pos = [], [], []

    LA_matrix = []  # read into (marker by (sample x ploidy x ancestry))
    for i, line in enumerate(f_handle):
        if i % 100 == 0:
            logging.info(f"processing {i}-th marker")

        line_split = line.strip().split("\t")
        LA_matrix.append(line_split[6:])

        lpos.append(int(line_split[1]))
        rpos.append(int(line_split[2]))
        pos.append(0.5 * (rpos[-1] + lpos[-1]))

        if chrom is None:
            chrom = int(line_split[0])

    LA_matrix = np.uint32(LA_matrix).reshape(-1, N, 2)
    LA_matrix = _ohe(LA_matrix, np.zeros(n_pops).astype("uint8"))
    lpos, rpos = np.uint32(lpos), np.uint32(rpos)
    pos = np.uint32(pos)

    ds = xr.Dataset(
        data_vars={
            "locanc": (["marker", "sample", "ploidy", "ancestry"], LA_matrix),
            "left_position": ("marker", lpos),
            "right_position": ("marker", rpos),
        },
        coords={
            "marker": pos,
            "sample": np.array(indiv, dtype=str),
            "ploidy": np.array([0, 1], dtype=np.int8),
            "ancestry": np.array(pops, dtype=str),
        },
    )

    return ds
