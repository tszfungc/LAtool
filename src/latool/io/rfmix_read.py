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
    """

    # Read ancestry line
    f_handle = open(fname, 'r')
    comment = f_handle.readline()
    pops = comment.strip().split("\t")[1:]
    n_pops = len(pops)

    # header line
    # RFMIX output dim: (marker by (sample x ploidy x ancestry))
    header = f_handle.readline()
    indiv = list(map(lambda x: x.split(":::")[0], header.strip().split("\t")[4:]))
    indiv = np.array(indiv[ :: (2 * n_pops)], dtype=str)
    N = indiv.shape[0]


    # data lines
    # create an (marker x sample x ploidy) by (ancestry) DataFrame, then xarray
    chrom = None

    ds = None

    return ds
