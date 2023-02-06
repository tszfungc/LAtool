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


def write_rfmix_fb(
    ds: xr.Dataset,
    out: str,
):
    samples = ds["sample"].values
    ancestries = ds["ancestry"].values
    samples_col = [
        f"{s}:::{h}:::{a}"
        for s in samples
        for h in ["hap1", "hap2"]
        for a in ancestries
    ]
    pos = ds["marker"].values + 1

    header = ["#reference_panel_population:\t" + "\t".join(ds["ancestry"].values)]
    header.append(
        "chromosome\tphysical_position\tgenetic_position\tgenetic_marker_index\t"
        + "\t".join(samples_col)
    )

    header = "\n".join(header)

    locanc_2d = ds["locanc"].values.reshape(ds.dims["marker"], -1)

    f = open(out, "w")

    print(header, file=f)
    for i, row in enumerate(locanc_2d):
        if i % 10000 == 0:
            print(f"Writing {i+1} / {pos.shape[0]} marker")

        row_str = f"1\t{pos[i]}\t.\t{i}\t" + "\t".join(row.astype(str))
        print(row_str, file=f)
