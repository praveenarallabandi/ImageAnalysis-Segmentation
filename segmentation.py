#!/usr/bin/env python3

import time
import toml
import click
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from typing import Any, List, Dict
from click import clear, echo, style, secho

from imageLibUtil import *
from watershedSegmentation import Watershed

conf: Dict[str, Any] = {}
colorChannel = 'R'
outputDir = ''
configLoc = './config.toml'


def erode(img_arr: np.array, win: int = 1) -> np.array:
    """
    erodes 2D numpy array holding a binary image
    """

    r = np.zeros(img_arr.shape)
    [yy, xx] = np.where(img_arr > 0)

    # prepare neighborhoods
    off = np.tile(range(-win, win + 1), (2 * win + 1, 1))
    x_off = off.flatten()
    y_off = off.T.flatten()

    # duplicate each neighborhood element for each index
    n = len(xx.flatten())
    x_off = np.tile(x_off, (n, 1)).flatten()
    y_off = np.tile(y_off, (n, 1)).flatten()

    # round out offset
    ind = np.sqrt(x_off ** 2 + y_off ** 2) > win
    x_off[ind] = 0
    y_off[ind] = 0

    # duplicate each index for each neighborhood element
    xx = np.tile(xx, ((2 * win + 1) ** 2))
    yy = np.tile(yy, ((2 * win + 1) ** 2))

    nx = xx + x_off
    ny = yy + y_off

    # bounds checking
    ny[ny < 0] = 0
    ny[ny > img_arr.shape[0] - 1] = img_arr.shape[0] - 1
    nx[nx < 0] = 0
    nx[nx > img_arr.shape[1] - 1] = img_arr.shape[1] - 1

    r[ny, nx] = 255

    return r


def dilate(img_arr: np.array, win: int = 1) -> np.array:
    inverted_img = np.invert(img_arr)
    eroded_inverse = erode(inverted_img, win).astype(np.uint8)
    eroded_img = np.invert(eroded_inverse)

    return eroded_img


def histogramThresholding(img_arr: np.array) -> np.array:

    hist = histogram(img_arr)

    middle = findMiddleHist(hist)

    img_copy = img_arr.copy()
    img_copy[img_copy > middle] = 255
    img_copy[img_copy < middle] = 0

    img_copy = img_copy.astype(np.uint8)

    return img_copy.reshape(img_arr.shape)


def histogramClustering(img_arr: np.array) -> np.array:
    img_hist = histogram(img_arr)
    out = k_means(img_hist, 2)

    diff = abs(out[1] - out[0])

    img_copy = img_arr.copy()
    img_copy[img_copy > diff] = 255
    img_copy[img_copy < diff] = 0

    img_copy = img_copy.astype(np.uint8)

    return img_copy.reshape(img_arr.shape)


def cannyEdgeDetection(img_arr: np.array) -> np.array:

    guass = gaussianKernel(5)

    blurred_image = convolve(img_arr, guass)

    sobel, theta = sobelFilters(blurred_image)

    suppresion = nonMaxSuppression(sobel, theta)

    threshold_image, weak, strong = threshold(suppresion)

    canny_image = hysteresis(threshold_image, weak, strong)

    return canny_image


def processImage(file: Path) -> str:
    try:
        img = getImageData(file)
        conf = toml.load(configLoc) # need to change to global variable
        img = selectChannel(img, conf["COLOR_CHANNEL"])
        
        # Edge detection
        edges = cannyEdgeDetection(img)

        # Histogram Clustering Segmentation
        segmented_clustering = histogramClustering(img)

        # Histogram Thresholding Segmentation
        segmented_thresholding = histogramThresholding(img)

        # Dilation
        dilated = dilate(segmented_thresholding)

        # Erosion
        eroded = erode(segmented_thresholding)

        # watershed segmentation
        w = Watershed()
        labels = w.apply(img)

        exportImage(edges, f"edges_{file.stem}", conf)
        exportImage(segmented_clustering, f"seg_clusting_{file.stem}", conf)
        exportImage(segmented_thresholding, f"seg_thresholding_{file.stem}", conf)
        exportImage(dilated, f"dilated_{file.stem}", conf)
        exportImage(eroded, f"eroded_{file.stem}", conf)
        exportImage(labels, f"ws_segmentation_{file.stem}", conf)
    except Exception as e:
        traceback.print_exc()
        return style(f"[ERROR] {file.stem} has an issue: {e}", fg="red")

    return style(f"{f'[INFO:{file.stem}]':15}", fg="green")


def processImages(files: List[Path]):
    """
    Batch operates on a set of images in a multiprocess pool
    """

    echo(
        style("[INFO] ", fg="green")
        + f"Processing (number of processes: {conf['NUM_OF_PROCESSES']})"
    )
    echo(style("[INFO] ", fg="green") + "compiling...")
    with Pool(conf["NUM_OF_PROCESSES"]) as p:
        with tqdm(total=len(files)) as pbar:
            for res in tqdm(p.imap(processImage, files)):
                pbar.write(res + f" finished...")
                pbar.update()


def main(config_location: str):
    clear()
    global conf
    conf = toml.load(config_location)
    base_path: Path = Path(conf["INPUT_SEG_DIR"])

    files: List = list(base_path.glob(f"*{conf['FILE_EXTENSION']}"))
    echo(
        style("[INFO] ", fg="green")
        + f"Image directory: {str(base_path)} - {len(files)} images found"
    )

    Path(conf["OUTPUT_SEG_DIR"]).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    processImages(files)
    t_delta = time.time() - t0

    print()
    secho(f"Total time: {t_delta:.2f} s", fg="green")


if __name__ == "__main__":
    main(configLoc)
