import logging

import numpy as np
import pandas as pd
import pgenlib as pg
import xarray as xr


def write_pgen(
    out: str,
    ds: xr.Dataset,
    anc_name: str,
    pos_coord: str = "marker",
    chrom: int = 1,
) -> None:
    """Writing local ancestry dosage to plink2 .pgen .fam .psam

    | According to specification in plink2 python API \\
    | https://github.com/chrchang/plink-ng/blob/master/2.0/Python/python_api.txt

    Args:
        out: output filename prefix
        ds: xarray Dataset containing ``locanc`` in the data_vars
        anc_name: name of target ancestry. Must be present in the coords ``ancestry``
        pos_coord: the name of coordinates used as position

    """

    if "locanc" not in ds:
        raise KeyError("No local ancestry data_vars is found in the dataset")
    if anc_name not in ds["ancestry"]:
        raise KeyError(f"No ancestry {anc_name} found")

    N, M = ds.dims["sample"], ds.dims["marker"]
    pos = ds[pos_coord].values
    iid = ds["sample"].values

    # pgen
    da_locanc = ds["locanc"].sel(ancestry=anc_name).sum(dim="ploidy").compute()
    with pg.PgenWriter(
        f"{out}.pgen".encode("utf-8"), N, M, False, dosage_present=True
    ) as pgwrite:
        for i in range(M):
            pgwrite.append_dosages(np.float32(da_locanc[i]))
    logging.info("Finish writing pgen file")

    # psam
    psam_df = pd.DataFrame({"#IID": iid}).assign(SEX="NA")
    psam_df.to_csv(f"{out}.psam", index=False, sep="\t")
    logging.info("Finish writing psam file")

    # pvar
    pvar_df = pd.DataFrame(
        {"#CHROM": np.repeat(chrom, M), "POS": pos, "ID": [f"{chrom}:{p}" for p in pos]}
    ).assign(REF="T", ALT="A")
    pvar_df.to_csv(f"{out}.pvar", index=False, sep="\t")
    logging.info("Finish writing pvar file")
