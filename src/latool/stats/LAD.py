import numpy as np
import xarray as xr


def empirical_LAD(
    da_locanc: xr.DataArray,
) -> xr.DataArray:
    """Compute empirical local ancestry linkage disequilibrium

    Args:
        da_locanc: DataArray storing the local ancestry dosage


    Returns:
        A marker by marker matrix of empirical LAD

    """

    LAD = xr.DataArray(
        name="Empirical LAD",
        data=np.corrcoef(da_locanc.values),
        dims=["marker1", "marker2"],
        coords={
            "marker1": da_locanc.marker.values,
            "marker2": da_locanc.marker.values,
        },
    )

    return LAD


def theoretical_LAD(approx=False, genetic_map=None):
    """Compute theoretical local ancestry linkage disequilibrium from genetic map"""
    pass
