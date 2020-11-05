import numpy as np
import os
import h5py
import math
import time
import json
import argparse
import logging
import multiprocessing
import functools


# Default parameters
mean_modevalue = 0.0
stdv_modevalue = 0.0
bgfactorL = 0.0
bgfactorR = 0.0
L = 0
R = 0

minvalue = 0.0
maxlevel = 65535  # 16bit

# Default is 2P.4 parameters
aL = 160.0
aR = 160.0
bL = 95.0
bR = 110.0

valid_nr_methods = [0, 1, 2, 3]


def mode(data):
    """
    Compute the mode of a numpy array.

    Parameters
    ----------
    data: np.ndarray
        Numpy array containing data of which we will calculate the mode.
    Returns
    -------
    list
        All values which appear the most number of times
    int
        The number of times the mode values appear
    """
    counts = {}

    for x in data.flatten():
        counts[x] = counts.get(x, 0) + 1

    maxcount = max(counts.values())
    modelist = []

    for x in counts:
        if counts[x] == maxcount:
            modelist.append(x)

    return modelist, maxcount


def noise_reduce(data, ind, xindex, noise_reduction):
    """
    Noise reduce a numpy array by one of several methods, specified by
    the noise_reduction argument. The current acceptable values are:
    - 0: No noise reduction
    - 1: Reduce the data value by 2 * sigma
    - 2: Reduce the data value by sigma and also normalize it
    - 3: Normalize the data value

    Parameters
    ----------
    data: np.ndarray
        The data that will be noise reduced
    ind: int
        The index currently being noise reduced within the
        larger dewarping process
    xindex: np.array
        ?
    noise_reduction: int
        The noise reduction method to use.
    Returns
    -------
    np.ndarray
        The data, of the sme shape as the input data, having had the
        dewarping process applied to it
    """
    if 0 == noise_reduction:
        pass

    elif 1 == noise_reduction:
        nf = (
            2 * stdv_modevalue
            * (xindex[ind] - xindex[ind - 1] - 1)
        )

        data = data - nf

    elif 2 == noise_reduction:
        nf = 1 * stdv_modevalue * (xindex[ind] - xindex[ind - 1] - 1)
        data = (data / (xindex[ind] - xindex[ind - 1])) - nf

    elif 3 == noise_reduction:
        data = data / (xindex[ind] - xindex[ind - 1])

    return data


def xdewarp(imgin, FOVwidth, xtable, noise_reduction):
    """
    Dewarp a single numpy array based on the information
    specified in table and noise_reduction.

    Parameters
    ----------
    imgin: np.ndarray
        The image that wil have the dewarping process applied to it.
    FOVwidth: int
        TODO Maybe it's a bug, and those 512s should be replaced with FOVwidth.
        Field of View width, a parameter of the experiment.
    xtable: dict
        A dictionary containing a number of statistics about the data
        to be dewarped, as well as various specifications for the process.
    noise_reduction: int
        The noise reduction method to use.
    Returns
    -------
    np.ndarray
        The data, of the sme shape as the input data, having had the
        dewarping process applied to it
    """
    # input_h5_file = h5py.File(input_file, 'r')
    # imgin = input_h5_file[input_dataset][frame, :, :]

    # Grab a few commonly used values from xtable to make code cleaner
    xindexL = xtable['xindexL']
    xindexLB = xtable['xindexLB']
    xindexR = xtable['xindexR']
    xindexRB = xtable['xindexRB']

    # Prepare a blank image
    imgout = np.zeros(imgin.shape)
    imgout[:, (int(aL)):(512-(int(aR)))] = imgin[:, (int(aL)):(512-(int(aR)))]

    sum = np.zeros(imgin.shape[0], np.float)

    # Left side
    for j in range(0, int(aL)):
        sum[:] = 0.0  # reset
        if xindexL[j] >= 0:
            if xindexLB[j - 1] >= 0.0:
                s = int(math.floor(xindexLB[j - 1]))
                e = int(math.floor(xindexLB[j]))

                sum[:] = (
                    (s + 1 - xindexLB[j - 1]) * imgin[:, s]
                    + (xindexLB[j] - e) * imgin[:, e]
                )

                if (e - s) > 1:   # have a middle pixel
                    sum[:] = sum[:] + imgin[:, s+1]

                # Perform the desired noise reduction method
                sum = noise_reduce(sum, j, xindexLB, noise_reduce)

                # TODO: Check on this. Make sure it wasn't an
                # error in the orginal code
                if 0 != noise_reduction:
                    low_mask = (sum < 0)
                    sum[low_mask] = 0  # underflow?

                high_mask = (sum > maxlevel)
                sum[high_mask] = maxlevel  # saturated?  for max image==1.0

                imgout[:, j] = sum
            else:
                imgout[:, j] = imgin[:, xindexL[j]] * xtable['bgfactorL']
        else:
            imgout[:, j] = xtable['mean_modevalue'] * xtable['bgfactorL']

    # Right side
    for j in range(0, (int(aR))):
        sum[:] = 0.0
        if xindexR[j] >= 0:
            if xindexRB[j - 1] >= 0.0:
                s = int(math.floor(xindexRB[j - 1]))
                e = int(math.floor(xindexRB[j]))

                sum[:] = (
                    (s + 1 - xindexRB[j - 1]) * imgin[:, 511 - s]
                    + (xindexRB[j] - e) * imgin[:, 511 - e]
                )

                if (e-s) > 1:  # have a middle pixel
                    sum[:] = sum[:] + imgin[:, 511 - (s + 1)]

                # Perform the desired noise reduction method
                sum = noise_reduce(sum, j, xindexRB, noise_reduce)

                # TODO: Check on this. Make sure it wasn't an
                # error in the orginal code
                if 0 != noise_reduction:
                    low_mask = (sum < 0)
                    sum[low_mask] = 0  # underflow?

                high_mask = (sum > maxlevel)
                sum[high_mask] = maxlevel  # saturated?  for max image==1.0

                imgout[:, j] = sum
            else:
                imgout[:, 511 - j] = (
                    imgin[:, 511 - xindexR[j]]
                    * xtable['bgfactorR']
                )
        else:
            imgout[:, 511 - j] = xtable['mean_modevalue'] * xtable['bgfactorR']

    if FOVwidth == 512:
        img = imgout.astype(np.uint16)
    else:
        img = imgout[:, xtable['L']:512 - xtable['R']].astype(np.uint16)

    return img


def xdewarp_worker(frames, FOVwidth, xtable, noise_reduction):
    """

    """

    return [
        xdewarp(frames[i, :, :], FOVwidth, xtable, noise_reduction)
        for i in range(frames.shape[0])
    ]


def get_xindex(a, b):
    """
    Generate the xinxex arrays that will be used in the dewarping process.

    Parameters
    ----------
    a: float
        Information about the rig used for the experiment.
    b: float
        Information about the rig used for the experiment.
    Returns
    -------
    np.array
        The data ready to use in the dewarping process
    np.array
        The data ready to use in the dewarping process
    """
    xindex = np.zeros(256, np.int)
    xindexB = np.zeros(256, np.float)  # between pixels

    for j in range(0, int(a)):
        xindex[j] = (
            j - int(
                    (b)
                    * (1.0 - math.sin((j/(a * 3.0) + 1.0/6.0) * 3.14159265))
                    + 0.5
                )
        )

        # TODO: Should these also use int()?
        xindexB[j] = (
            (j + 0.5) - (
                (b) * (1.0 - math.sin(
                    ((j + 0.5)/(a * 3.0) + 1.0 / 6.0) * 3.14159265
                ))
            )
        )

    return xindex, xindexB


def create_xtable(movie, aL, aR, bL, bR, noise_reduction):
    """
    Compute a number of statistics about the images to be dewarped.

    Parameters
    ----------
    meanimg: np.ndarray
        The mean value for each pixel across the movie.
    stdvimg: np.ndarray
        The standard deviation for each pixel of the movie.
    aL: float
        Information about the rig used for the experiment.
    aR: float
        Information about the rig used for the experiment.
    bL: float
        Information about the rig used for the experiment.
    bR: float
        Information about the rig used for the experiment.
    noise_reduction: int
        The noise reduction method that will be used in dewarping.
    Returns
    -------
    dict
        Various computed information about the video.
    """
    # assume flat area aL - (512-aR)

    meanimg = np.mean(movie, axis=0)  # use avgimg to compute mode
    meanimg = meanimg/256          # less noisy to get mode from 8bit histogram
    meanimg = meanimg.astype(int)  # less noisy to get mode from 8bit histogram

    # full movie got Memory Error
    stdvimg = np.std(movie[::8], axis=0)  # use avgimg to compute mode
    stdvimg = stdvimg/256          # less noisy to get mode from 8bit histogram
    stdvimg = stdvimg.astype(int)  # less noisy to get mode from 8bit histogram

    # Left side
    xindexL, xindexLB = get_xindex(aL, bL)

    # Right side
    xindexR, xindexRB = get_xindex(aR, bR)

    modelist, mean_maxcount = mode(meanimg)
    mean_modevalue = modelist[0]
    mean_minvalue = np.min(meanimg)

    # IMPORTANT!: estmated modevalue is from 8bit img, convert back to 16bit
    mean_modevalue = mean_modevalue * 256
    mean_minvalue = mean_minvalue * 256

    modelist, stdv_maxcount = mode(stdvimg)
    stdv_modevalue = modelist[0]
    stdv_minvalue = np.min(stdvimg)

    # IMPORTANT!: estmated modevalue is from 8bit img, convert back to 16bit
    stdv_modevalue = stdv_modevalue * 256
    stdv_minvalue = stdv_minvalue * 256

    j = 0
    while xindexLB[j] < 0.0 or xindexL[j] < 0:
        j = j + 1
    else:
        bgfactorL = xindexLB[j+1] - xindexLB[j]
        L = j

    j = 0
    while xindexRB[j] < 0.0 or xindexR[j] < 0:
        j = j + 1
    else:
        bgfactorR = xindexRB[j+1] - xindexRB[j]
        R = j

    # TODO: Check on this. Make sure it wasn't an
    # error in the orginal code
    if 0 != noise_reduction:
        bgfactorL = 1
        bgfactorR = 1

    table = {
        "mean_modevalue": mean_modevalue,
        'mean_maxcount': mean_maxcount,
        "mean_minvalue": mean_minvalue,
        "stdv_modevalue": stdv_modevalue,
        "stdv_maxcount": stdv_maxcount,
        "stdv_minvalue": stdv_minvalue,
        "bgfactorL": bgfactorL,
        "bgfactorR": bgfactorR,
        "L": L,
        "R": R,
        "xindexL": xindexL,
        "xindexLB": xindexLB,
        "xindexR": xindexR,
        "xindexRB": xindexRB
    }
    logging.debug(table)

    return table


def parse_input(data):
    """
    Compute a number of statistics about the images to be dewarped.

    Parameters
    ----------
    data: json
        Data from the input json file
    Returns
    -------
    dict
        Various computed information about the video.
    """

    input_h5 = data.get('input_h5', None)
    if input_h5 is None:
        raise KeyError("input JSON does not have required field: 'input_h5'")
    if not os.path.exists(input_h5):
        raise IOError(f"{input_h5} does not exist")

    output_h5 = data.get('output_h5', None)
    if output_h5 is None:
        raise KeyError("input JSON does not have required field: 'output_h5'")

    equipment_name = data.get("equipment_name", None)
    if equipment_name is None:
        raise KeyError("input JSON does not have"
                       "required field: 'equipment_name'")

    aLstr = data.get("aL", None)
    if aLstr is None:
        raise KeyError(f"parameter aL not determined for {equipment_name}")
    aL = float(aLstr)

    aRstr = data.get("aR", None)
    if aRstr is None:
        raise KeyError(f"parameter aR not determined for {equipment_name}")
    aR = float(aRstr)

    bLstr = data.get("bL", None)
    if bLstr is None:
        raise KeyError(f"parameter bL not determined for {equipment_name}")
    bL = float(bLstr)

    bRstr = data.get("bR", None)
    if bRstr is None:
        raise KeyError(f"parameter bR not determined for {equipment_name}")
    bR = float(bRstr)

    logging.debug(str(equipment_name))
    logging.debug(f"aL: {aL}   bL: {bL}")
    logging.debug(f"aR: {aR}   bR: {bR}")

    return input_h5, output_h5, aL, aR, bL, bR


def make_output_file(output_file, output_dataset,
                     FOVwidth, movie_shape, movie_dtype):
    """

    """
    # Remove old output if it exists and create new output file
    if os.path.exists(output_file):
        os.remove(output_file)

    dewarped_file = h5py.File(output_file, 'w')
    if FOVwidth == 512:
        dewarped_file.create_dataset(
            output_dataset, shape=movie_shape, dtype=movie_dtype
        )
    else:
        out_shape = [movie_shape[0], movie_shape[1], 512 - R - L]
        dewarped_file.create_dataset(
            output_dataset, shape=out_shape, dtype=movie_dtype
        )
    dewarped_file.close()


def split_input_movie(movie, chunk_size):
    """
    Splits the movie into chunks of roughly equal size, as close to the
    specified chunk_size as possible

    Parameters
    ----------
    Returns
    -------
    """

    return np.array_split(movie, chunk_size, axis=0)


def run_dewarping(FOVwidth, noise_reduction, threads,
                  input_file, input_dataset, output_file, output_dataset):
    """
    Gets information about the movie and the specified dewarping method,
    then uses multiprocessing to dewarp each frame of the movie at once, while
    tracking the progress.

    Parameters
    ----------
    FOVwidth: int
        Field of View width, a parameter of the experiment.
    Returns
    -------
    """

    # Get statistics about the movie
    input_h5_file = h5py.File(input_file, 'r')
    movie = input_h5_file[input_dataset]

    movie_shape = movie.shape
    movie_dtype = movie.dtype
    T, y_orig, x_orig = movie.shape

    # IMPORTANT!: estmated modevalue is from 8bit img but convert back to 16bit
    xtable = create_xtable(movie, aL, aR, bL, bR, noise_reduction)

    make_output_file(output_file, output_dataset,
                     FOVwidth, movie_shape, movie_dtype)

    movie_chunks = split_input_movie(movie, 1000)
    input_h5_file.close()

    start_time = time.time()
    with multiprocessing.Pool(threads) as pool, \
         h5py.File(output_file, "a") as f:

        fn = functools.partial(
            xdewarp_worker,
            FOVwidth=FOVwidth,
            xtable=xtable,
            noise_reduction=noise_reduction
        )

        num_processed = 0
        for chunk in movie_chunks:
            for frame, dewarped_frame in enumerate(pool.starmap(fn,
                                                                movie_chunks)):
                f[output_dataset][num_processed + frame, :, :] = dewarped_frame
            num_processed = num_processed + len(chunk)

    end_time = time.time()
    logging.debug(f"Elapsed time (s): {end_time - start_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    parser.add_argument('output_json')
    parser.add_argument('--log_level', default=logging.DEBUG)
    parser.add_argument('--threads', default=8, type=int)
    parser.add_argument('--FOVwidth', default=0, type=int)
    parser.add_argument('--noise_reduction', default=0, type=int)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    if args.noise_reduction not in valid_nr_methods:
        raise(
            f"{args.noise_reduction} is not a valid noise "
            f"reduction option. Must be one of {valid_nr_methods}. "
            f"Using default value noise_reduction = 0.\n",
            UserWarning
        )
    else:
        logging.debug(f"noise_reduction: {args.noise_reduction}")

    with open(args.input_json, 'r') as json_file:
        input_data = json.load(json_file)
    input_h5, output_h5, aL, aR, bL, bR = parse_input(input_data)

    run_dewarping(
        FOVwidth=args.FOVwidth,
        noise_reduction=args.noise_reduction,
        threads=args.threads,
        input_file=input_h5,
        input_dataset="data",
        output_file=output_h5,
        output_dataset="data"
    )

    with open(args.output_json, 'w') as outfile:
        json.dump({}, outfile)
