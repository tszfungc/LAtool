""""Module for annotating Xarray with external annotations"""

import numpy as np
import pandas as pd
import xarray as xr


def genetic_distance(
    ds: xr.Dataset,
    genetic_map: str,
    chrom: int,
    chrom_col: int = 0,
    pos_col: int = 1,
    cM_col: int = 3,
    pd_kwargs: dict = {},
) -> xr.Dataset:
    """Annotate xarray with genetic distance from a genetic map

    Args:
        ds: Dataset containing local ancestry
        genetic_map: path or url to the genetic map
        chrom: the chromosome to be used in the genetic map
        pos_col: column number (0-based) of the position in the genetic map
        cM_col: column number (0-based) of the cM in the genetic map
        pd_kwargs: keyword arguments to be passed to pandas.read_csv


    Returns:
        Dataset with annotated genetic position in data variables

    """

    csv_kwargs = {"sep": " "}
    csv_kwargs.update(pd_kwargs)

    genetic_map = pd.read_csv(genetic_map, **csv_kwargs)
    genetic_map = genetic_map[genetic_map.iloc[:, chrom_col] == chrom]

    _genetic_position = np.interp(
        x=ds["marker"].values,
        xp=genetic_map.iloc[:, pos_col],
        fp=genetic_map.iloc[:, cM_col],
    )

    ds["genetic_position"] = ("marker", _genetic_position)

    return ds
