import numpy as np
import os
import h5py
import math
import time
import allensdk.core.json_utilities as ju
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


def xdewarp(imgin, FOVwidth, xtable, noise_reduction):
    """

    """

    # Grab a few commonly used values from xtable to make code cleaner
    xindexL = xtable.xindexL
    xindexLB = xtable.xindexLB
    xindexR = xtable.xindexR
    xindexRB = xtable.xindexRB

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
                if 0 == noise_reduction:
                    pass

                elif 1 == noise_reduction:
                    nf = (
                        2 * stdv_modevalue
                        * (xindexLB[j] - xindexLB[j - 1] - 1)
                    )

                    sum = sum - nf

                elif 2 == noise_reduction:
                    nf = 1 * stdv_modevalue * (xindexLB[j] - xindexLB[j-1] - 1)
                    sum = (sum / (xindexLB[j] - xindexLB[j-1])) - nf

                elif 3 == noise_reduction:
                    sum = sum / (xindexLB[j] - xindexLB[j-1])

                # TODO: Check on this. Make sure it wasn't an
                # error in the orginal code
                if 0 != noise_reduction:
                    low_mask = (sum < 0)
                    sum[low_mask] = 0  # underflow?

                high_mask = (sum > maxlevel)
                sum[high_mask] = maxlevel  # saturated?  for max image==1.0

                imgout[:, j] = sum
            else:
                imgout[:, j] = imgin[:, xindexL[j]] * xtable.bgfactorL
        else:
            imgout[:, j] = xtable.mean_modevalue * xtable.bgfactorL

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
                if 0 == noise_reduction:
                    pass

                elif 1 == noise_reduction:
                    nf = 2 * stdv_modevalue * (xindexRB[j] - xindexRB[j-1] - 1)
                    sum = sum - nf

                elif 2 == noise_reduction:
                    nf = 1 * stdv_modevalue * (xindexRB[j] - xindexRB[j-1] - 1)
                    sum = (sum / (xindexRB[j] - xindexRB[j-1])) - nf

                elif 3 == noise_reduction:
                    sum = sum / (xindexRB[j] - xindexRB[j-1])

                else:
                    low_mask = None
                    high_mask = (sum > maxlevel)

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
                    * xtable.bgfactorR
                )
        else:
            imgout[:, 511 - j] = xtable.mean_modevalue * xtable.bgfactorR

    if FOVwidth == 512:
        img = imgout.astype(np.uint16)
    else:
        img = imgout[:, xtable.L:512 - xtable.R].astype(np.uint16)

    return img


def create_xtable(meanimg, stdvimg, aL, aR, bL, bR, noise_reduction):
    """

    """
    # assume flat area aL - (512-aR)

    xindexL = np.zeros(256, np.int)
    xindexLB = np.zeros(256, np.float)  # between pixels
    xindexR = np.zeros(256, np.int)
    xindexRB = np.zeros(256, np.float)  # between pixels

    # Left side
    for j in range(0, int(aL)):
        xindexL[j] = (
            j - int(
                    (bL)
                    * (1.0 - math.sin((j/(aL * 3.0) + 1.0/6.0) * 3.14159265))
                    + 0.5
                )  # ok
        )

        # TODO: Should these also use int()?
        xindexLB[j] = (
            (j + 0.5) - (
                (bL) * (1.0 - math.sin(
                    ((j + 0.5)/(aL * 3.0) + 1.0 / 6.0) * 3.14159265
                ))
            )  # ok
        )

    # Right side
    for j in range(0, int(aR)):
        xindexR[j] = (
            j - int(
                (bR)
                * (1.0 - math.sin((j/(aR * 3.0) + 1.0/6.0) * 3.14159265))
                + 0.5)  # default
        )

        xindexRB[j] = (
           (j + 0.5) - (
               (bR) * (1.0 - math.sin(
                   ((j+0.5)/(aR * 3.0) + 1.0 / 6.0) * 3.14159265
                ))
            )  # default
        )
        print(256 - j, j, xindexL[j], xindexLB[j], xindexR[j], xindexRB[j])

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

    print('mean_modevalue=', mean_modevalue, 'mean_maxcount', mean_maxcount)
    print('mean_minvalue=',  mean_minvalue)
    print('stdv_modevalue=', stdv_modevalue, 'stdv_maxcount', stdv_maxcount)
    print('stdv_minvalue=',  stdv_minvalue)
    print('bgfactorL=', bgfactorL)
    print('bgfactorR=', bgfactorR)
    print('L R ', L, R)

    return {
        "mean_modevalue": mean_modevalue,
        'mean_maxcount': mean_maxcount,
        "stdv_modevalue": stdv_modevalue,
        "stdv_maxcount": stdv_maxcount,
        "mean_minvalue": mean_minvalue,
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


def parse_input(data):
    """
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

    print(str(equipment_name))
    print('aL bL', aL, bL)
    print('aR bR', aR, bR)

    return input_h5, output_h5, aL, aR, bL, bR


def run_dewarping(FOVwidth, noise_reduction, threads,
                  input_file, input_dataset,
                  output_file, output_dataset):
    """

    """

    # Get statistics about the movie
    input_h5_file = h5py.File(input_file, 'r')
    movie = input_h5_file[input_dataset]
    meanimg = np.mean(movie, axis=0)  # use avgimg to compute mode
    meanimg = meanimg/256          # less noisy to get mode from 8bit histogram
    meanimg = meanimg.astype(int)  # less noisy to get mode from 8bit histogram

    # full movie got Memory Error
    stdvimg = np.std(movie[::8], axis=0)  # use avgimg to compute mode
    stdvimg = stdvimg/256          # less noisy to get mode from 8bit histogram
    stdvimg = stdvimg.astype(int)  # less noisy to get mode from 8bit histogram

    movie_shape = movie.shape
    movie_dtype = movie.dtype
    T, y_orig, x_orig = movie.shape

    # need one frame from movie for mode
    # IMPORTANT!: estmated modevalue is from 8bit img but convert back to 16bit
    xtable = create_xtable(meanimg, stdvimg, aL, aR, bL, bR, noise_reduction)

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

    start_time = time.time()

    # Separate frames into elementes of a list and dewarp each indvidually
    images = [
        input_h5_file[input_dataset][frame, :, :] for frame in range(T)
    ]

    with multiprocessing.Pool(threads) as pool:
        fn = functools.partial(
            xdewarp,
            FOVwidth=FOVwidth,
            xtable=xtable,
            noise_reduction=noise_reduction
        )
        result = pool.map_async(fn, images, chunksize=1)

        while not result.ready():
            num_finished = T - result._number_left
            frac = float(num_finished) / T
            fps = float(num_finished) / (time.time() - start_time)

            logging.debug(
                f"Finished: {num_finished}/{T}, {frac * 100}, fps: {fps}"
            )
            time.sleep(5)

    f = h5py.File(output_file, "a")
    f[output_dataset] = np.stack(result.get())

    end_time = time.time()
    logging.debug(f"Elapsed time (s): {end_time - start_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    parser.add_argument('output_json')
    parser.add_argument('--log_level', default=logging.DEBUG)
    parser.add_argument('--threads', default=4, type=int)
    parser.add_argument('--FOVwidth', default=0, type=int)
    parser.add_argument('--noise_reduction', default=0, type=int)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    if args.noise_reduction not in valid_nr_methods:
        raise(
            f"{args.noise_reduction} is not a valid noise "
            f"reduction option. Must be from {valid_nr_methods}. "
            f"Using default value noise_reduction = 0.\n",
            UserWarning
        )
    else:
        print(f"noise_reduction: {args.noise_reduction}")

    input_data = ju.read(args.input_json)
    input_h5, output_h5, aL, aR, bL, bR = parse_input(input_data)

    logging.debug("got parameters from json")

    """
        New strategy:
            read the data in under run_image,
            split into all the frames there,
            pass each frame as input to the xdewarp methods,
            also pass the new "parameters" dict,
            pool.map,
            collect output from pool.map into one video,
            return video to main
    """

    run_dewarping(
        FOVwidth=args.FOVwidth,
        noise_reduction=args.noise_reduction,
        threads=args.threads,
        input_file=input_h5,
        input_dataset="data",
        output_file=output_h5,
        output_dataset="data"
    )

    ju.write(args.output_json, {})
