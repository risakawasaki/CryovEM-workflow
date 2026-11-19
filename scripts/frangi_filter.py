import argparse
import os

import mrcfile
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters


def sigma_list(smin, smax, step):
    """Expand (min, max, step) into the list of scales, inclusive of max (e.g. 4 7 1 -> 4, 5, 6, 7)."""
    return [float(s) for s in np.arange(smin, smax + step / 2, step)]


def frangi_response(image_slice, sigmas, alpha, beta, gamma=None):
    """Raw (unnormalized) Frangi vesselness response for one 2D slice."""
    return filters.frangi(image_slice.astype(np.float32), sigmas=sigmas,
                          alpha=alpha, beta=beta, gamma=gamma).astype(np.float32)


def z_aggregate(resp, window, method):
    """Combine each slice's response with its neighbours along z.

    Cristae and membranes persist across adjacent slices; pixel noise does not,
    so a median (or mean) over a small z-window suppresses the noise.
    window=1 disables this step. Edge slices use the nearest available slices.
    """
    if window <= 1:
        return resp
    size = (window, 1, 1)
    if method == "median":
        return ndi.median_filter(resp, size=size, mode="nearest")
    return ndi.uniform_filter(resp, size=size, mode="nearest")


def normalize(resp, mode):
    """Min-max scale to 0-255, either per slice (original behaviour) or over the whole volume."""
    out = np.zeros_like(resp, dtype=np.float32)
    if mode == "volume":
        lo, hi = resp.min(), resp.max()
        if hi > lo:
            out = (resp - lo) / (hi - lo) * 255
        return out
    for z in range(resp.shape[0]):
        lo, hi = resp[z].min(), resp[z].max()
        if hi > lo:
            out[z] = (resp[z] - lo) / (hi - lo) * 255
    return out


def remove_small_islands(mask, min_size):
    """Drop connected foreground components smaller than min_size pixels, per 2D (Y, X) slice."""
    if min_size <= 0:
        return mask
    out = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        lab, n = ndi.label(mask[z])
        if n == 0:
            out[z] = mask[z]
            continue
        sizes = np.bincount(lab.ravel())
        drop = np.isin(lab, np.where(sizes < min_size)[0])
        out[z] = mask[z] & ~drop
    return out


def load_volume(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mrc":
        with mrcfile.open(path, permissive=True) as mrc:
            return np.asarray(mrc.data), mrc.voxel_size
    elif ext in (".tif", ".tiff"):
        return tifffile.imread(path), None
    raise ValueError(f"Unsupported file type: {ext}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to input MRC or TIF volume [z, y, x]")
    ap.add_argument("--output", required=True, help="directory to save output")
    ap.add_argument("--sigmas", nargs=3, type=float, default=[4, 7, 1],
                    help="Frangi scales as min max step, inclusive (4 7 1 -> 4,5,6,7)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Frangi alpha (affects 3D plate-like structures only)")
    ap.add_argument("--beta", type=float, default=0.5, help="Frangi beta")
    ap.add_argument("--gamma", type=float, default=None, help="Frangi gamma; default None = set automatically per slice")
    ap.add_argument("--z-window", type=int, default=1,
                    help="odd number of slices combined along z (e.g. 3 = z-1, z, z+1); 1 disables z-continuity")
    ap.add_argument("--z-method", choices=["median", "mean"], default="median",
                    help="how responses are combined along z (median is more robust to noise)")
    ap.add_argument("--norm", choices=["slice", "volume"], default="slice",
                    help="min-max scale to 0-255 per slice (original behaviour) or once over the whole volume")
    ap.add_argument("--threshold", type=float, default=None,
                    help="threshold on the 0-255 response; foreground = response > threshold. Omit to keep the grayscale response")
    ap.add_argument("--min-size", type=int, default=0,
                    help="after thresholding, drop 2D islands smaller than this many pixels; 0 disables. Ignored without --threshold")
    args = ap.parse_args()

    if args.z_window < 1 or args.z_window % 2 == 0:
        ap.error("--z-window must be a positive odd number (1, 3, 5, ...)")

    os.makedirs(args.output, exist_ok=True)
    sigmas = sigma_list(*args.sigmas)
    print(f"Frangi scales: {sigmas}")

    img_array, voxel_size = load_volume(args.input)
    print(f"Image shape: {img_array.shape}")

    # 1. raw Frangi response per slice
    resp = np.stack([frangi_response(img_array[z], sigmas, args.alpha, args.beta, args.gamma)
                     for z in range(img_array.shape[0])])

    # 2. z-continuity: combine each slice with its neighbours before thresholding
    resp = z_aggregate(resp, args.z_window, args.z_method)

    # 3. scale to 0-255
    resp = normalize(resp, args.norm)

    # 4. optional threshold and island removal
    if args.threshold is None:
        out = resp.astype(np.uint8)
        if args.min_size > 0:
            print("--min-size is ignored because --threshold was not set")
    else:
        fg = resp > args.threshold
        fg = remove_small_islands(fg, args.min_size)
        out = fg.astype(np.uint8) * 255

    s = args.sigmas
    suffix = (f"-sigmas{s[0]:g}-{s[1]:g}-step{s[2]:g}-alpha{args.alpha}-beta{args.beta}"
              f"-gamma{args.gamma}-z{args.z_window}{args.z_method}-norm{args.norm}"
              f"-threshold{args.threshold}-minsize{args.min_size}")
    stem = os.path.splitext(os.path.basename(args.input))[0]
    base = os.path.join(args.output, f"{stem}-frangi{suffix}")

    with mrcfile.new(base + ".mrc", overwrite=True) as mrc:
        mrc.set_data(out)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size
    tifffile.imwrite(base + ".tif", out)
    print(f"Saved {base}.mrc and .tif")


if __name__ == "__main__":
    main()
