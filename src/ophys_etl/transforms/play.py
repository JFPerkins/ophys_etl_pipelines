import h5py
import numpy as np
import math

from ophys_etl.transforms import sine_dewarp


def test_dewarp_regression():
    old_dewarped_video = h5py.File(
        'tests/transforms//resources/dewarping_regression_test_output.h5', 'r'
    )

    input_video = h5py.File(
        'tests/transforms//resources/dewarping_regression_test_input.h5', 'r'
    )

    xtable = create_xtable(
        movie=input_video['data'],
        aL=160.0,
        aR=150.0,
        bL=85.0,
        bR=100.0,
        noise_reduction=3
    )

    new_dewarped_video = []
    for frame in range(input_video['data'].shape[0]):
        new_dewarped_video.append(
            sine_dewarp.xdewarp(
                imgin=input_video['data'][frame, :, :],
                FOVwidth=0,
                xtable=xtable,
                noise_reduction=3
            )
        )
    new_dewarped_video = np.stack(new_dewarped_video)

    np.testing.assert_array_equal(
        old_dewarped_video['data'],
        new_dewarped_video
    )


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

    modelist, mean_maxcount = sine_dewarp.mode(meanimg)
    mean_modevalue = modelist[0]
    mean_minvalue = np.min(meanimg)

    # IMPORTANT!: estmated modevalue is from 8bit img, convert back to 16bit
    mean_modevalue = mean_modevalue * 256
    mean_minvalue = mean_minvalue * 256

    modelist, stdv_maxcount = sine_dewarp.mode(stdvimg)
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
        "xindexRB": xindexRB,
        "aL": aL,
        "aR": aR,
        "bL": bL,
        "bR": bR
    }

    return table


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


if '__main__' == __name__:
    test_dewarp_regression()
