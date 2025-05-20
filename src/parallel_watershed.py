import numpy as np
from multiprocessing import Pool
from skimage.segmentation import watershed
import gc
import cv2


def get_tiles(x, splits=4, pad_size=64):
    ts_1 = np.ceil(x.shape[0] / splits).astype(int)
    ts_1_idx = np.arange(0, x.shape[0], ts_1)
    ts_2 = np.ceil(x.shape[1] / splits).astype(int)
    ts_2_idx = np.arange(0, x.shape[1], ts_2)
    tiles = []
    crop_info = []
    for i in ts_1_idx:
        for j in ts_2_idx:
            tiles.append(x[i : i + ts_1 + pad_size, j : j + ts_2 + pad_size])
            x_start = 0 if i == 0 else pad_size // 2
            x_stop = -pad_size // 2 if i != ts_1_idx[-1] else None
            y_start = 0 if j == 0 else pad_size // 2
            y_stop = -pad_size // 2 if j != ts_2_idx[-1] else None
            crop_info.append([x_start, x_stop, y_start, y_stop])

    return tiles, crop_info


def apply_watershed(inp):
    id, ws_surface, markers, fg = inp

    bb_ws = watershed(ws_surface, markers, connectivity=2, mask=fg)
    return id, bb_ws


def parallel_watershed(image, markers, mask, nprocs=4, pad_size=16):
    print(image.shape)
    surface_tiles, crop_info = get_tiles(image, nprocs, pad_size)
    markers_tiles, _ = get_tiles(markers, nprocs, pad_size)
    if mask is None:
        foreground_tiles = [None] * len(surface_tiles)
    else:
        foreground_tiles, _ = get_tiles(mask, nprocs, pad_size)
    print(len(surface_tiles))
    p = Pool(nprocs)
    watershed_tiles = list(
        p.map(
            apply_watershed,
            zip(
                np.arange(len(surface_tiles)),
                surface_tiles,
                markers_tiles,
                foreground_tiles,
            ),
        )
    )

    ordered_tiles = [i for _, i in sorted(watershed_tiles, key=lambda x: x[0])]
    del surface_tiles, markers_tiles, foreground_tiles, watershed_tiles
    gc.collect()
    res = []
    for j in range(0, nprocs**2, nprocs):
        res.append(
            np.concatenate(
                [
                    o[c[0] : c[1], c[2] : c[3]]
                    for o, c in zip(
                        ordered_tiles[j : j + nprocs], crop_info[j : j + nprocs]
                    )
                ],
                axis=1,
            )
        )
    res = np.concatenate(res, axis=0)
    return res


def apply_dilate(inp):
    id, tile, rad = inp
    return id, cv2.dilate(
        tile, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
    )


def parallel_dilate(image, rad, nprocs=4, pad_size=16):
    surface_tiles, crop_info = get_tiles(image, nprocs, pad_size)

    p = Pool(nprocs)
    dilate_tiles = list(
        p.map(
            apply_dilate,
            zip(
                np.arange(len(surface_tiles)),
                surface_tiles,
                [rad] * len(surface_tiles),
            ),
        )
    )

    ordered_tiles = [i for _, i in sorted(dilate_tiles, key=lambda x: x[0])]
    del surface_tiles, dilate_tiles
    gc.collect()
    res = []
    for j in range(0, nprocs**2, nprocs):
        res.append(
            np.concatenate(
                [
                    o[c[0] : c[1], c[2] : c[3]]
                    for o, c in zip(
                        ordered_tiles[j : j + nprocs], crop_info[j : j + nprocs]
                    )
                ],
                axis=1,
            )
        )
    res = np.concatenate(res, axis=0)
    return res
