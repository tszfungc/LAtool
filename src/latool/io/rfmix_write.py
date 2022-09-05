import pandas as pd
import xarray as xr


def write_Q(
    ds: xr.Dataset,
    out: str,
) -> pd.DataFrame:
    """Write global ancestry in rfmix.Q format

    args:
        ds: xarray Dataset containing ``locanc`` in the data_vars
        out: output filename

    """
    ga = ds.locanc.mean(dim=["marker", "ploidy"]).to_dataframe().reset_index()
    ga = ga.pivot(index="sample", columns="ancestry", values="locanc").reset_index()
    ga = ga.rename({"sample": "#sample"}, axis=1)

    with open(out, "w") as f:
        f.write("#rfmix diploid global ancestry .Q format output\n")
        ga.to_csv(f, sep="\t", index=None)

    return ga
