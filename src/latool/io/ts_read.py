"""Module for reading local ancestry from tree sequences

Output from software to be supported:
- msprime

"""
import logging
from typing import Any, List

import msprime
import numpy as np
import pandas as pd
import tskit
import xarray as xr

_logger = logging.getLogger(__name__)

def _trace_anc(
    treeseq: tskit.TreeSequence,
    node_admixed: List[int],
    node_ancestor: List[int],
    ancestries: List[Any],
) -> xr.Dataset:
    """Trace ancestry

    Args:
        treeseq: Tree sequence
        node_admixed: list of admixed node in the tree sequence
        node_ancestor: list of ancestor node in the tree sequence at census time
        ancestries: list of ancestry names

    Returns:
        Dataset containing local ancestries

    """
    anc_dict = {i.metadata["name"]: i.id for i in treeseq.populations()}
    # Trace ancestor id at census time
    locanc_tbl = treeseq.tables.link_ancestors(node_admixed, node_ancestor)

    # Convert ancestors id to pop id
    # Convert child's haplotype id to individual id
    traced_pop = [treeseq.node(i).population for i in locanc_tbl.parent]
    traced_indiv = [treeseq.node(i).individual for i in locanc_tbl.child]

    # ploidy id: 0 if even else 1.
    ds_dict = {
        "Start": locanc_tbl.left,
        "Node": locanc_tbl.child % 2,
        "individual": traced_indiv,
    }

    # one hot encode ancestries
    for a in ancestries:
        ds_dict[a] = np.equal(anc_dict[a], traced_pop).astype(np.int8)

    # prep dict for making xarray
    loc_arr = pd.DataFrame(ds_dict)
    loc_arr.columns = ["marker", "ploidy", "sample"] + ancestries
    loc_arr = pd.melt(
        loc_arr,
        id_vars=["marker", "ploidy", "sample"],
        var_name="ancestry",
        value_vars=ancestries,
        value_name="locanc",
    )
    loc_arr.set_index(["marker", "sample", "ploidy", "ancestry"], inplace=True)

    # Convert pandas to xarray.Dataset, propagate all entry to the following nan
    ds = xr.Dataset.from_dataframe(loc_arr)

    return ds


def read_msp_ts(
    fname: str,
    admixpop: str,
    ancpop: List[str],
    keep: Any = None,
    extract: Any = None,
) -> xr.Dataset:

    """Trace ancestry in tree sequence output from msprime

    The ancestors and the population they belong to at the census time are traced.

    Args:
        fname: path to tree sequence
        admixpop: population name of the admixed population
        ancpop: list of names of the ancestral populations
        keep: id of admixed individuals to be included

    Returns:
        Dataset containing local ancestry

    Example
    -------
    >>> from latool.io import read_msp_ts
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

    """

    ts = tskit.load(fname)
    ancpop = list(ancpop)

    # nodes to be traced
    node_admixed = np.array(
        [
            i.id
            for i in ts.nodes()
            if ts.population(i.population).metadata["name"] == admixpop
            and i.time == 0.0
        ]
    )

    if keep is not None:
        node_admixed = node_admixed[np.in1d(node_admixed, keep)]

    _logger.info(f"Number of admixed individuals kept: {len(node_admixed)//2}")
    node_ancestor = [i.id for i in ts.nodes() if i.flags == msprime.NODE_IS_CEN_EVENT]

    _logger.info(ancpop)

    if len(node_ancestor) == 0:
        raise RuntimeError("No Census event found: No ancestors can be traced")

    def indiv_batch_generator(nodes, size=20):
        n = len(nodes)
        for left in range(0, n, size):
            right = min(left + size, n)
            yield nodes[left:right]

    xarr_list = []
    for idx, batch in enumerate(indiv_batch_generator(node_admixed)):
        if idx % 10 == 0:
            _logger.info(f"tracing {idx+1}-th batch of {len(batch)//2} individuals")
        xarr_list.append(
            _trace_anc(
                treeseq=ts,
                node_admixed=batch,
                node_ancestor=node_ancestor,
                ancestries=ancpop,
            )
        )
    if extract is not None:
        for di, _ in enumerate(xarr_list):
            xarr_list[di] = xarr_list[di].ffill(dim='marker')
            xarr_list[di] = xarr_list[di].sel(marker=extract, method='ffill')
            xarr_list[di]['marker'] = extract

    xarr_ = xr.concat(xarr_list, dim="sample")
    # xarr_ = laxr_simplify(xarr_)

    xarr_ = xarr_.ffill(dim="marker")

    rmost_pos = ts.sequence_length
    lpos = np.uint(xarr_.marker.values)
    rpos = np.append(xarr_.marker.values[1:], rmost_pos).astype(np.uint)

    # set marker to mid(lpos, rpos). Annotate left pos and right pos.
    # xarr_["marker"] = np.uint(0.5 * (lpos + rpos))
    # xarr_["left_position"] = ("marker", lpos)
    # xarr_["right_position"] = ("marker", rpos)
    xarr_['sample'] = np.array([f"indiv{s:d}" for s in xarr_['sample']], dtype=object)

    return xarr_


def read_msp_mutations(
    fname: str,
    admixpop: str,
    ancpop: List[str],
    keep: Any = None,
) -> xr.Dataset:

    ts = tskit.load(fname)

    # nodes to be traced
    node_admixed = np.array(
        [
            i.id
            for i in ts.nodes()
            if ts.population(i.population).metadata["name"] == admixpop
            and i.time == 0.0
        ]
    )
    if keep is not None:
        node_admixed = node_admixed[np.in1d(node_admixed, keep)]

    sample_id = [ ts.node(node_admixed[i]).individual for i in range(0, len(node_admixed), 2)]
    sample = np.array([f"indiv{s:d}" for s in sample_id], dtype=object)
    marker = np.array(ts.tables.sites.position)


    gt = (
        ts.genotype_matrix()[:, node_admixed]
        .reshape(marker.shape[0], sample.shape[0], 2)
    )
    #gt_dos = gt[:, node_admixed][:, ::2] + gt[:, node_admixed][:, 1::2]
    ds_gt = xr.Dataset(
        data_vars={
            "genotype":(['marker', 'sample', 'ploidy'],gt)
        },
        coords={
            "marker": marker,
            "sample": sample,
            "ploidy": [0, 1]
            }
    )


    return ds_gt
