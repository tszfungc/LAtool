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

    lad = xr.DataArray(
        name="Empirical LAD",
        data=np.corrcoef(da_locanc.values),
        dims=["marker1", "marker2"],
        coords={
            "marker1": da_locanc.marker.values,
            "marker2": da_locanc.marker.values,
        },
    )

    return lad


def theoretical_LAD(
    approx: bool =True,
    genetic_map: xr.DataArray=None,
    g: int = 10,
    ) -> xr.DataArray:
    """Compute theoretical local ancestry linkage disequilibrium from genetic map

    corr(A,A) = (1-theta)^g, where theta is recombination probability and g is number of generation since admixture
    corr(A,A) approx exp(-g*lambda), where lambda is genetic distance in cM

    Returns:
        A marker by marker matrix of theoretical LAD

    """

    dist_mat = np.abs(genetic_map.values - genetic_map.values.reshape(-1, 1))
    if approx:
        corr = np.exp(- 0.01 * dist_mat * g)
    else:
        corr = (0.5*(1+np.exp(-2*dist_mat)))**g



    lad = xr.DataArray(
        name="Empirical LAD",
        data=corr,
        dims=["marker1", "marker2"],
        coords={
            "marker1": genetic_map.marker.values,
            "marker2": genetic_map.marker.values,
        },
    )

    return lad
