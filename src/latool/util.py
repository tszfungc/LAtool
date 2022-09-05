import numpy as np
import xarray as xr


def fill_pos(ds: xr.Dataset) -> xr.Dataset:
    pass


def simplify(
    ds: xr.Dataset,
) -> xr.Dataset:
    """Simplify the local ancestry data.

    Merge identical consecutive markers

    args:
        ds: xarray Dataset containing ``locanc`` in the data_vars

    returns:
        Dataset where markers are removed if the local ancestry
        is identical to the previous marker

    Example
    -------
    >>> from latool.io import read_msp_ts
    >>> from latool.util import simplify
    >>> ds = read_msp_ts(
    ...     fname="tests/testdata/example.ts",
    ...     admixpop='ADMIX',
    ...     ancpop=['EUR', 'AFR'])
    >>> ds
    <xarray.Dataset>
    Dimensions:         (marker: 5, sample: 10, ploidy: 2, ancestry: 2)
    Coordinates:
      * marker          (marker) uint64 15472077 15671821 15704336 15776649 15883331
      * sample          (sample) int64 0 1 2 3 4 5 6 7 8 9
      * ploidy          (ploidy) int64 0 1
      * ancestry        (ancestry) object 'AFR' 'EUR'
    Data variables:
        locanc          (marker, sample, ploidy, ancestry) float32 1.0 0.0 ... 0.0
        left_position   (marker) uint64 15295879 15648276 15695366 15713306 15839992
        right_position  (marker) uint64 15648276 15695366 15713306 15839992 15926670
    >>> ds.locanc[0] = ds.locanc[1]
    >>> simplify(ds)
    <xarray.Dataset>
    Dimensions:         (marker: 4, sample: 10, ploidy: 2, ancestry: 2)
    Coordinates:
      * marker          (marker) uint64 15472077 15704336 15776649 15883331
      * sample          (sample) int64 0 1 2 3 4 5 6 7 8 9
      * ploidy          (ploidy) int64 0 1
      * ancestry        (ancestry) object 'AFR' 'EUR'
    Data variables:
        locanc          (marker, sample, ploidy, ancestry) float32 1.0 0.0 ... 0.0
        left_position   (marker) uint64 15295879 15695366 15713306 15839992
        right_position  (marker) uint64 15648276 15713306 15839992 15926670

    """
    if ds.dims["marker"] == 1:
        return ds

    arr = ds["locanc"].values
    non_dup = np.append([True], (~np.all(np.equal(arr[1:], arr[:-1]), axis=(1, 2, 3))))

    ds = ds.isel(marker=non_dup)

    # TODO: Update positions

    return ds
