"""Calculate immune scores given available results from HoVer-NeXt and C2R SRMA."""

import json
import os
from glob import glob
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import openslide
import pandas as pd
import zarr
from libpysal import weights
from PIL import Image
from scipy.ndimage import label
from scipy.spatial import cKDTree
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects
from tqdm.auto import tqdm

from src.constants import CA_DICT, LOOKUP, LUP_CA, PRED_KEYS, RADII
from src.data_utils import WholeSlideDataset
from src.parallel_watershed import parallel_dilate, parallel_watershed


def load_data(wsi_p: Path, params: dict):
    """Load and prepare previously generated data from HoVer-NeXt.

    Parameters
    ----------
    wsi_p: Path
        input Path from eval_main to H&E WSI
    params: dict
        Dictionary generated in main

    """
    print("loading data")
    if type(wsi_p) is str:
        wsi_p = Path(wsi_p)
    ending = wsi_p.suffix
    p = wsi_p.stem

    print(Path(params["nuclei_results"], p, "pinst_pp.zip"))
    try:
        pinst = zarr.open(Path(params["nuclei_results"], p, "pinst_pp.zip"), "r")[:]
    except FileNotFoundError:
        return None, None, None, None, None, None

    # load the dictionary
    with Path(params["nuclei_results"], p, "class_inst.json").open("r") as fp:
        pcls = json.load(fp)
    # create a centroid info array
    centroid_array = np.array(
        [[int(k), v[0], *v[1]] for k, v in pcls.items()]
    )  # instance_id, class_id, y, x

    # or alternatively create a lookup for the instance map to get a corresponding class map
    pcls_list = np.array([0] + [v[0] for v in pcls.values()])
    pcls_keys = np.array(["0", *list(pcls.keys())]).astype(int)
    lookup = np.zeros(pcls_keys.max() + 1, dtype=np.uint8)
    lookup[pcls_keys] = pcls_list
    cls_map = lookup[pinst]

    try:
        if wsi_p.endswith(".czi"):
            bounds_x = 0
            bounds_y = 0
        else:
            with openslide.open_slide(wsi_p) as sl:
                bounds_x = int(sl.properties["openslide.bounds-x"])  # 158208
                bounds_y = int(sl.properties["openslide.bounds-y"])  # 28672
    except KeyError:
        bounds_x = 0
        bounds_y = 0

    try:
        ptis = np.load(
            Path(
                params["segmentation_results"],
                p + ending,
                p + ending + "_meta_full.npz",
            ),
        )
    except FileNotFoundError:
        return None, None, None, None, None, None
    cx = ptis["x"]
    cy = ptis["y"]
    crds = np.vstack([cx, cy]).T
    ca_p = ptis["p"]
    x_interval = np.min(np.unique(cx)[1:] - np.unique(cx)[:-1])

    crds_ = crds - np.array([bounds_x, bounds_y])
    crds_adj = crds_ - (np.min(crds_[0], 0) % x_interval)
    crds_adj /= x_interval
    crds_adj = crds_adj.astype(np.int32)

    # 32 is hardcoded, CA segmentation is 16mpp, 32x0.5mpp
    msk_ = np.full(
        (*tuple(np.around(np.array(cls_map.shape) / 32).astype(np.int32)), 9),
        1,
        dtype=np.float32,
    )

    crd_flt = (crds_adj < list(reversed(msk_.shape[:2]))).all(1)
    ca_p_relu = np.maximum(0, ca_p)
    msk_[crds_adj[crd_flt, 1], crds_adj[crd_flt, 0]] = ca_p_relu[crd_flt]
    zoom_msk = cv2.resize(msk_, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    zoom_cls = np.argmax(zoom_msk, axis=2)
    # TODO: fix +4
    zoom_zero = np.zeros((zoom_cls.shape[0] + 4, zoom_cls.shape[1] + 4), dtype=np.uint8)
    zoom_zero[4:, 4:] = zoom_cls
    return pinst, cls_map, lookup, x_interval, zoom_zero, centroid_array


def gen_mask(
    mask,
    x_interval,
    cutoff_area_tum_um=500.0,
    blur_area_um=500.0,
    closing_px=10,
    mpp=0.2524,
    blur_th=0.1,
):
    """Generate IEL mask."""
    interval_um = x_interval * mpp
    cutoff_area_tum_px = np.ceil(cutoff_area_tum_um / interval_um).astype(int)
    blur_area_px = np.ceil(blur_area_um / interval_um).astype(int)

    if blur_area_px != 0:
        mask = cv2.blur(mask.astype(float), (blur_area_px, blur_area_px))
        mask = mask > blur_th

    # Set tumor location
    mask = closing(mask, disk(2))
    mask = remove_small_objects(mask, int(cutoff_area_tum_px) ** 2)
    mask = closing(mask, disk(closing_px))
    return remove_small_holes(mask, int(cutoff_area_tum_px) ** 2)


def create_seedmap(cls_map, nodes):
    """Create seedmap for watershed."""
    seeds = np.zeros_like(cls_map, dtype=np.int32)
    n_ = nodes[nodes[:, 1] != PRED_KEYS["lymphocyte"]].copy().astype(int)
    seeds[n_[:, 2], n_[:, 3]] = n_[:, 0]
    return zarr.array(seeds)


def epi_mask(cls_map, seeds, nodes, nprocs=4, epi_dil=49, con_dil=25, min_size=8192):
    """Use seedmap to create watershed based map of epithelium."""
    ws = parallel_watershed(
        np.zeros_like(cls_map, dtype=np.int32),
        np.asarray(seeds),
        mask=None,
        nprocs=nprocs,
        pad_size=64,
    )
    lkup = np.zeros(nodes[:, 0].max().astype(int) + 1).astype(np.int32)
    lkup[nodes[:, 0].astype(int)] = nodes[:, 1]
    epi_con_mask = lkup[ws]
    del ws, lkup

    dilated_ = parallel_dilate(
        (np.asarray(cls_map) == PRED_KEYS["epithelial-cell"]).astype(np.uint8),
        epi_dil,
        nprocs=nprocs,
        pad_size=128,
    )
    con_dilated_ = parallel_dilate(
        (epi_con_mask == PRED_KEYS["connective-tissue-cell"]).astype(np.uint8),
        con_dil,
        nprocs=nprocs,
        pad_size=128,
    )

    mby = (dilated_ > 0) & (epi_con_mask == PRED_KEYS["epithelial-cell"]) & ~(con_dilated_ > 0)
    return remove_small_objects(mby, min_size)


def find_iels(nodes, ie_mask):
    """Detect IELs."""
    tils_ = nodes[nodes[:, 1] == PRED_KEYS["lymphocyte"]].copy()
    iels_ = tils_[ie_mask[tils_[:, 2].astype(int), tils_[:, 3].astype(int)]].copy()

    iel_tree = cKDTree(iels_[:, 2:])
    query = iel_tree.query_ball_tree(iel_tree, 20)
    iel_ids_ = iels_[[q[0] for q in query if len(q) == 1], 0]
    return iel_ids_, nodes[np.isin(nodes[:, 0], iel_ids_), 2:]


def export_qp(iel_crds, result_dir, pt="iel"):
    """Export detected IELs as tsv for import to QuPath."""
    iel_crds = iel_crds * 2
    iel_crds = np.around(
        iel_crds,
        2,
    )  # Adjust for center crop from prediction ??? doesnt actually make sense :)
    file = result_dir + "/pred_" + pt + ".tsv"
    with Path(file).open("w") as textfile:
        textfile.write("x" + "\t" + "y" + "\t" + "name" + "\t" + "color" + "\n")
        lines = [
            str(element[1]) + "\t" + str(element[0]) + "\t" + pt + "\t" + "-256" + "\n"
            for element in iel_crds
        ]
        textfile.writelines(lines)


def get_filtered_nuc(ca_map, pinst, cent_array):
    msk_rm = remove_small_objects(np.isin(ca_map, [1, 2, 4]), 1024)
    msk_rm = (
        cv2.resize(
            msk_rm.astype(np.uint8),
            (pinst.shape[1], pinst.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    bin_filter = ~msk_rm[cent_array[:, -2].astype(int), cent_array[:, -1].astype(int)]
    return cent_array[bin_filter, 0], bin_filter


def get_tls(
    cent_array, nuc_filter, norm_map, params, p: Path, thresh=24, min_size=500, max_size=5000
):
    """Detect TLS and LNs by clustering."""
    nucs = cent_array[nuc_filter]
    tils = nucs[nucs[:, 1] == 3]
    coordinates = tils[:, 2:]
    dist = weights.DistanceBand.from_array(coordinates, threshold=thresh, silence_warnings=True)
    dist_graph = nx.from_scipy_sparse_array(dist.sparse, create_using=nx.Graph())
    cc = list(nx.connected_components(dist_graph))

    tls = [np.array(list(x)) for x in cc if (len(x) > min_size) & (len(x) < max_size)]
    lns = [np.array(list(x)) for x in cc if (len(x) >= max_size)]
    shp = norm_map.shape
    nm_rs = cv2.resize(
        np.asarray(norm_map).astype(np.uint8),
        dsize=(shp[1] // 10, shp[0] // 10),
        interpolation=cv2.INTER_NEAREST,
    )
    nm_rs = remove_small_objects(nm_rs > 0, 1e4)
    nm_rs = cv2.dilate(nm_rs.astype(np.uint8), disk(50))
    nm_fl = cv2.resize(nm_rs, (shp[1], shp[0]), interpolation=cv2.INTER_NEAREST)
    keep = []
    for i, j in enumerate(tls):
        til_crd = tils[j, 2:].astype(int)
        if not nm_fl[til_crd[:, 0], til_crd[:, 1]].any():
            keep.append(i)
    nucs_out, tls_flt, lns_out = None, None, None
    if len(tls) > 0:
        nucs_out = pd.DataFrame(tils[np.concatenate(tls)], columns=["id", "class", "x", "y"])
        nucs_out["tls_id"] = np.concatenate([np.full_like(j, i) for i, j in enumerate(tls)])
        nucs_out[["id", "tls_id"]].to_csv(
            Path(params["nuclei_results"], p, "tls_ids.csv"),
            index=False,
        )
        tls_flt = nucs_out[nucs_out["tls_id"].isin(keep)]
        tls_flt[["id", "tls_id"]].to_csv(
            Path(params["nuclei_results"], p, "tls_ids_filt.csv"),
            index=False,
        )
        export_qp(
            nucs_out[["x", "y"]].to_numpy(),
            Path(params["nuclei_results"], p),
            pt="tls",
        )

    if len(lns) > 0:
        lns_out = pd.DataFrame(tils[np.concatenate(lns)], columns=["id", "class", "x", "y"])
        lns_out["ln_id"] = np.concatenate([np.full_like(j, i) for i, j in enumerate(lns)])
        lns_out[["id", "ln_id"]].to_csv(
            Path(params["nuclei_results"], p, "ln_ids.csv"),
            index=False,
        )
    return nucs_out, tls_flt, lns_out


def extend(a):
    """Quick function to make a list from list of lists."""
    out = []
    for sublist in a:
        out.extend(sublist)
    return out


def get_muc_amount(
    ca_map, x_interval=64, ellipse_size=(16, 16), iterations=40, muc_ellipse=(16, 16)
):
    """Calculate amount of mucin from tissue seg map."""
    msk = gen_mask(ca_map == CA_DICT["Tumor"], x_interval)
    closing = cv2.morphologyEx(
        (msk * 1).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse_size),
        iterations=iterations,
    )
    muc_exp = cv2.morphologyEx(
        ((ca_map == CA_DICT["Mucin"]) * 1).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, muc_ellipse),
        iterations=(1),
    )
    lab_muc, _ = label(muc_exp)
    muc_blobs = np.unique(lab_muc[closing.astype(bool) & (lab_muc != 0)])
    pot_muc = ca_map == CA_DICT["Mucin"]
    pot_muc[~np.isin(lab_muc, muc_blobs[1:])] = 0
    return np.sum(pot_muc) * (2**7)


def get_crd_adj(wsi):
    """Calculate coordinate adjustment for MIRAX files."""
    path = Path(wsi)
    ds = WholeSlideDataset(
        path=path,
        crop_sizes_px=[256],
        crop_magnifications=[20],
        padding_factor=0.96875,
        ratio_object_thresh=0.001,  # Remove image background
        remove_background=True,  # Do not remove bb that overlap with background
        remove_oob=False,  # DO not remove images with background
    )
    try:
        bounds_x = int(ds.s.properties["openslide.bounds-x"])  # 158208
        bounds_y = int(ds.s.properties["openslide.bounds-y"])  # 28672
    except KeyError:
        bounds_x = 0
        bounds_y = 0

    return (ds.crop_reference_cxy - np.array([bounds_x, bounds_y])).min(0)


def eval_main(wsi: str, params: dict):
    """Generate scores."""
    wsi = Path(wsi)
    c = wsi.stem
    ext = wsi.suffix
    # Check if IELs have already been generated, and generate if not.
    if np.any(
        [
            not Path(params["nuclei_results"], c, i).exists()
            for i in ["filtered_inst.tsv", "pred_com.csv", "tum_inst.tsv"]
        ]
    ):
        pinst, cls_map, _, x_interval, ca_map, cent_array = load_data(wsi, params)
        if pinst is None:
            return {"id": c}

        # p = os.path.split(os.path.splitext(wsi)[0])[-1]
        big_mask = gen_mask(ca_map == CA_DICT["Normal mucosa"], x_interval)
        big_mask = (
            cv2.resize(
                big_mask.astype(np.uint8),
                (pinst.shape[1], pinst.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

        pinst[big_mask] = 0
        cls_map[big_mask] = 0

        # compress
        big_mask = zarr.array(big_mask)
        close_to_norm = big_mask[cent_array[:, -2].astype(int), cent_array[:, -1].astype(int)]
        nodes = cent_array[~close_to_norm].copy()
        #     # create IEL map
        seeds = create_seedmap(cls_map, nodes)
        # generate tumor mask
        ie_mask = epi_mask(cls_map, seeds, nodes, params["nprocs"])
        if params["merge_seg"]:
            tum_mask = cv2.resize(
                (ca_map).astype(np.uint8),
                (pinst.shape[1], pinst.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            dilated = parallel_dilate(ie_mask, 64, nprocs=params["nprocs"], pad_size=128)
            tum_mask[dilated == 0] = False
            ie_mask |= tum_mask
            if not Path(
                params["segmentation_results"],
                os.path.split(wsi)[-1],
                "tum_mask.zip",
            ).exists():
                zarr.save_array(
                    Path(
                        params["segmentation_results"],
                        os.path.split(wsi)[-1],
                        "tum_mask.zip",
                    ),
                    ie_mask,
                )
        iel_ids, iel_crd = find_iels(nodes, ie_mask)
        export_qp(iel_crd, Path(params["nuclei_results"], c), pt="iel")

        with Path(params["nuclei_results"], c, "iel_inst.tsv").open("w") as txt:
            txt.writelines([str(elem) + "\n" for elem in iel_ids])
        # save also tumor epithelial cells separately
        with Path(params["nuclei_results"], c, "tum_inst.tsv").open("w") as txt:
            txt.writelines(
                [
                    str(elem) + "\n"
                    for elem in nodes[nodes[:, 1] == PRED_KEYS["epithelial-cell"], 0].astype(int)
                ]
            )
        # write detections as csv
        pd.DataFrame(cent_array, columns=["id", "class", "x", "y"]).to_csv(
            Path(params["nuclei_results"], c, "pred_com.csv"), index=False
        )
        filtered_nuc, nuc_filter = get_filtered_nuc(ca_map, pinst, cent_array)
        with Path(params["nuclei_results"], c, "filtered_inst.tsv").open("w") as txt:
            txt.writelines([str(elem) + "\n" for elem in filtered_nuc])
        nucs_out, tls_flt, lns_out = get_tls(cent_array, nuc_filter, big_mask, params, c)
        nucs_flt = pd.DataFrame(cent_array[nuc_filter], columns=["id", "class", "x", "y"])
        tum = nodes[nodes[:, 1] == PRED_KEYS["epithelial-cell"], 0].astype(int)
        iel = iel_ids
        try:
            tls = nucs_out["id"].to_numpy().flatten().astype(int)
            tls_filt = tls_flt["tls_id"]
            n_tls = 0 if len(tls_filt) == 0 else max(tls_filt)
        except TypeError:
            tls = []
            n_tls = 0
        try:
            lns = lns_out["id"].to_numpy().flatten().astype(int)
        except TypeError:
            lns = []
    ### Generate scores
    else:
        try:
            filtered_nuc = np.loadtxt(Path(params["nuclei_results"], c, "filtered_inst.tsv"))
            nucs = pd.read_csv(Path(params["nuclei_results"], c, "pred_com.csv"))
            nucs_flt = nucs.loc[nucs["id"].astype(int).isin(filtered_nuc)].copy()
            tum = np.loadtxt(Path(params["nuclei_results"], c, "tum_inst.tsv"))
        except FileNotFoundError:
            print(c)
            return {"id": c}
        try:
            iel = np.loadtxt(Path(params["nuclei_results"], c, "iel_inst.tsv"))
        except FileNotFoundError:
            iel = []
        try:
            tls = (
                pd.read_csv(Path(params["nuclei_results"], c, "tls_ids.csv"), sep=",")["id"]
                .to_numpy()
                .flatten()
                .astype(int)
            )
            tls_filt = pd.read_csv(Path(params["nuclei_results"], c, "tls_ids_filt.csv"), sep=",")[
                "tls_id"
            ]
            n_tls = 0 if len(tls_filt) == 0 else max(tls_filt)
        except FileNotFoundError:
            tls = []
            n_tls = 0
        try:
            lns = (
                pd.read_csv(Path(params["nuclei_results"], c, "ln_ids.csv"), sep=",")["id"]
                .to_numpy()
                .flatten()
                .astype(int)
            )
        except FileNotFoundError:
            lns = []

    all_tils = nucs_flt.loc[nucs_flt["class"] == 3].copy()
    all_tum = nucs_flt.loc[nucs_flt["id"].astype(int).isin(tum)].copy()
    tils_filt = all_tils.loc[
        ~all_tils["id"].astype(int).isin(np.concatenate([iel, tls, lns]))
    ].copy()
    tum_wr = nucs_flt["id"].astype(int).isin(tum).to_numpy()
    nucs_flt.loc[nucs_flt["id"].isin(tils_filt["id"]), "class"] = 9
    nucs_immune = nucs_flt[~nucs_flt["class"].isin([2, 6])].copy()
    tree = cKDTree(nucs_immune[["x", "y"]].to_numpy())
    tum_tree = cKDTree(all_tum[["x", "y"]].to_numpy())
    front_detected = True
    try:
        a = np.load(
            Path(
                params["segmentation_results"],
                c + ext,
                c + ext + "_meta_full_1000_tbc.npz",
            ),
            allow_pickle=True,
        )["arr_0"].item()
    except FileNotFoundError:
        front_detected = False
        print("front missing")
    if front_detected:
        tbc_contour = np.concatenate(a["tbc_contour"][-1]) * 32

        adj = get_crd_adj(wsi) / 2

        tb_tree = cKDTree((tbc_contour + adj)[:, [1, 0]])
        inv_front = tb_tree.query_ball_tree(tum_tree, r=1000)  #
        inv_front_tum_set = np.array(sorted(set(extend(inv_front)))).astype(int)
        tum_filter = np.full(all_tum.to_numpy().shape[0], False)
        tum_filter[inv_front_tum_set.astype(int)] = True
        tum_tree_front = cKDTree(all_tum[["x", "y"]].to_numpy()[tum_filter])
        tum_blob = a["tumor_blob"].copy()

        adj_shp = (adj / 32)[[1, 0]].astype(int)
        tum_blob_adj = np.zeros(np.array(tum_blob.shape) + adj_shp, dtype="bool")

        # bad code to fix negative indices
        if adj_shp[0] < 0:
            if adj_shp[1] < 0:
                tum_blob_adj = tum_blob[np.abs(adj_shp[0]) :, np.abs(adj_shp[1]) :]
            else:
                tum_blob_adj[:, adj_shp[1] :] = tum_blob[np.abs(adj_shp[0]) :, :]
        elif adj_shp[1] < 0:
            tum_blob_adj[adj_shp[0] :, :] = tum_blob[:, np.abs(adj_shp[1]) :]
        else:
            tum_blob_adj[adj_shp[0] :, adj_shp[1] :] = tum_blob

        tum_blob_adj = cv2.dilate(
            tum_blob_adj.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)),
            iterations=1,
        )

        low_res = np.round(all_tum[["x", "y"]].to_numpy() / 32).astype(int)
        low_res[low_res >= tum_blob_adj.shape] = 0
        center_filter = tum_blob_adj[low_res[:, 0], low_res[:, 1]].astype(bool)
        tum_tree_no_front = cKDTree(all_tum[["x", "y"]].to_numpy()[~tum_filter & center_filter])

    rw = {"id": c}

    for i in RADII:  # 1mpp
        i_ = i * 2  # .5mpp
        w_rad = tum_tree.query_ball_tree(tree, r=i_)
        w_rad_set = np.array(sorted(set(extend(w_rad)))).astype(int)
        if front_detected:
            immune_front = tum_tree_front.query_ball_tree(tree, r=i_)
            immune_front_set = np.array(sorted(set(extend(immune_front)))).astype(int)

            immune_filter = np.full(nucs_immune.to_numpy().shape[0], fill_value=True)
            immune_filter[immune_front_set] = False

            immune_no_front = cKDTree(nucs_immune[["x", "y"]].to_numpy()[immune_filter])
            immune_center = tum_tree_no_front.query_ball_tree(immune_no_front, r=i_)
            immune_center_set = np.array(sorted(set(extend(immune_center)))).astype(int)

            front_classes, front_counts = np.unique(
                nucs_immune.to_numpy()[immune_front_set, 1],
                return_counts=True,
            )
            center_classes, center_counts = np.unique(
                nucs_immune.to_numpy()[immune_filter][immune_center_set, 1],
                return_counts=True,
            )
            front = np.zeros(len(PRED_KEYS), dtype=np.float64)
            center = np.zeros(len(PRED_KEYS), dtype=np.float64)

            for cl, cnt in zip(front_classes, front_counts):
                front[int(cl) - 1] = cnt
            for cl, cnt in zip(center_classes, center_counts):
                center[int(cl) - 1] = cnt

        else:
            front = np.full(len(PRED_KEYS), np.nan)
            center = np.full(len(PRED_KEYS), np.nan)
        full_classes, full_counts = np.unique(
            nucs_immune.to_numpy()[w_rad_set, 1],
            return_counts=True,
        )
        full = np.zeros(len(PRED_KEYS), dtype=np.float64)
        for cl, cnt in zip(full_classes, full_counts):
            full[int(cl) - 1] = cnt
        for counts, loc in zip([full, front, center], ["full", "front", "center"]):
            for cl, cnt in enumerate(counts, 1):
                if cl in [2, 6, 8]:
                    continue
                if LOOKUP[cl] == "lymphocyte":
                    rw["lymphocyte_" + loc + "_r" + str(i)] = cnt + counts[9 - 1]
                else:
                    rw[cl + "_" + loc + "_r" + str(i)] = cnt

        if i == 20:
            rw["iel_count"] = np.count_nonzero(
                np.isin(nucs_immune.to_numpy()[w_rad_set.astype(int), 0], iel)
            )
            iel_lu = np.isin(nucs_immune["id"], iel)
            n_w_iels = [iel_lu[nb].any() for nb in w_rad]
            rw["epi_w_iel_neigh"] = np.count_nonzero(n_w_iels)

    for j in [1, 3, 4, 5, 6]:
        if j != PRED_KEYS["lymphocyte"]:
            rw[LOOKUP[j] + "_total"] = np.count_nonzero(nucs_flt["class"] == j)
        else:
            rw[LOOKUP[j] + "_total"] = np.count_nonzero(nucs_flt["class"].isin([j, 9]))
    rw["filtered_lymphocyte"] = tils_filt.shape[0]
    rw["tumor_epi_full"] = np.count_nonzero(tum_wr)
    if front_detected:
        rw["tumor_epi_front"] = np.count_nonzero(tum_filter)
        rw["tumor_epi_center"] = np.count_nonzero(~tum_filter & center_filter)
    rw["n_tls"] = n_tls
    im = np.array(
        Image.open(Path(params["segmentation_results"], c + ext, c + ext + "_fullhd.png"))
    )
    im_map = LUP_CA[im[..., 1]]
    areas = {
        k + "_um2": np.count_nonzero(im_map == i) * (2**7)
        for k, i in zip(["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"], range(9))
    }

    muc_refined = get_muc_amount(im_map)
    return {**rw, **areas, "MUC_refined_um2": muc_refined}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wsi_path",
        type=str,
        help="either path to txt with wsi paths or glob pattern",
    )
    parser.add_argument(
        "--nuclei_results",
        type=str,
        help="path to tile folder",
    )
    parser.add_argument(
        "--segmentation_results",
        type=str,
        help="path to prediction folder",
    )
    parser.add_argument(
        "--merge_seg",
        action="store_true",
        help="merge segmentation results with nuclei epi-mask",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=4,
        help="number of processes to use for parallelization",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="cell_counts.csv",
        help="output file path",
    )
    parser.add_argument("--array_tasks", type=int, default=1, help="use when running as array")
    parser.add_argument("--array_id", type=int, default=0, help="specify with slurm parameter")
    args = parser.parse_args()
    params = vars(args)
    if params["wsi_path"].endswith(".txt"):
        with Path(params["wsi_path"]).open() as f:
            wsi_list = [x.rstrip("\n") for x in f]
    else:
        wsi_list = glob(params["wsi_path"])
    if params["array_tasks"] > 1:
        wsi_list = np.array(sorted(wsi_list))[params["array_id"] :: params["array_tasks"]]
        params["output_path"] = str(params["array_id"]).join(
            os.path.splitext(params["output_path"])
        )

    counts_per_case = []
    for wsi in tqdm(wsi_list):
        try:
            counts = eval_main(wsi, params)
            counts_per_case.append(counts)
        except Exception as e:
            print(e)
    pd.DataFrame(counts_per_case).to_csv(params["output_path"], index=False)
