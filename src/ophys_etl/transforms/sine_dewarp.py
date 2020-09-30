# Template from leakage_deconv.py by David
import sys
import numpy as np
import os
import h5py
import math
import time
import allensdk.core.json_utilities as ju
# import json_utilities as ju
import argparse
import logging
import multiprocessing
import functools
# import scientifica_paramaters as sp

# Read system parameters from input.json than from scientifica_paramaters.py

# --- for grid image test ---
'''
from PIL import Image
from pylab import *

# Add these to solve no X server problem:
# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
'''
# --- end for grid image test ---


LOCK = multiprocessing.Lock()


x = np.zeros(256, np.int)
xindexL = np.zeros(256, np.int)
xindexLB = np.zeros(256, np.float)  # between pixels
xindexR = np.zeros(256, np.int)
xindexRB = np.zeros(256, np.float)  # between pixels

mean_modevalue = 0.0
stdv_modevalue = 0.0
minvalue = 0.0
bgfactorL = 0.0
bgfactorR = 0.0
L = 0
R = 0
maxlevel = 65535  # 16bit

# default is 2P.4 parameters
aL = 160.0
aR = 160.0
bL = 95.0
bR = 110.0


def mode(data):
    counts = {}
    for x in data.flatten():
        counts[x] = counts.get(x, 0) + 1
    maxcount = max(counts.values())
    modelist = []
    for x in counts:
        if counts[x] == maxcount:
            modelist.append(x)
    return modelist, maxcount


def create_xtable(meanimg, stdvimg, aL, aR, bL, bR):
    # assume flat area aL - (512-aR)

    # Left side
    for j in range(0, int(aL)):
        xindexL[j] = j - int((bL) * (1.0 - math.sin((j/(aL * 3.0) + 1.0/6.0) * 3.14159265)) + 0.5)  # ok
        xindexLB[j] = (j+0.5) - ((bL) * (1.0 - math.sin(((j+0.5)/(aL * 3.0) + 1.0 / 6.0) * 3.14159265)))  # ok

    # Right side
    for j in range(0, int(aR)):
        xindexR[j] = j - int((bR) * (1.0 - math.sin((j/(aR * 3.0) + 1.0/6.0) * 3.14159265)) + 0.5)  # default
        xindexRB[j] = (j+0.5) - ((bR) * (1.0 - math.sin(((j+0.5)/(aR * 3.0) + 1.0 / 6.0) * 3.14159265)))  # default
        print(256 - j, j, xindexL[j], xindexLB[j], xindexR[j], xindexRB[j])

    # Compute global variables
    global mean_modevalue
    global stdv_modevalue
    global mean_minvalue
    global stdv_minvalue
    global bgfactorL
    global bgfactorR
    global L
    global R

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

    print('mean_modevalue=', mean_modevalue, 'mean_maxcount', mean_maxcount)
    print('mean_minvalue=',  mean_minvalue)
    print('stdv_modevalue=', stdv_modevalue, 'stdv_maxcount', stdv_maxcount)
    print('stdv_minvalue=',  stdv_minvalue)
    print('bgfactorL=', bgfactorL)
    print('bgfactorR=', bgfactorR)
    print('L R ', L, R)


def xdewarp(imgin, FOVwidth):
    # Center
    # for j in range(aL,512-aR):   #near-flat area
    imgout = np.zeros(imgin.shape)
    imgout[:, (int(aL)):(512-(int(aR)))] = imgin[:, (int(aL)):(512-(int(aR)))]

    # sum=np.zeros(512, np.float)
    sum = np.zeros(imgin.shape[0], np.float)

    # Left side
    for j in range(0, int(aL)):
        sum[:] = 0.0  # reset
        if xindexL[j] >= 0:
            if xindexLB[j-1] >= 0.0:
                s = int(math.floor(xindexLB[j-1]))
                e = int(math.floor(xindexLB[j]))
                sum[:] = (s+1 - xindexLB[j-1]) * imgin[:, s] + (xindexLB[j] - e) * imgin[:, e]
                if (e - s) > 1:   # have a middle pixel
                    sum[:] = sum[:] + imgin[:, s+1]
                mask = (sum > maxlevel)  # saturated?  for max image==1.0
                sum[mask] = maxlevel
                imgout[:, j] = sum
            else:
                imgout[:, j] = imgin[:, xindexL[j]] * bgfactorL
        else:
            imgout[:, j] = mean_modevalue * bgfactorL

    # Right side
    for j in range(0, (int(aR))):
        sum[:] = 0.0
        if xindexR[j] >= 0:
            if xindexRB[j-1] >= 0.0:
                s = int(math.floor(xindexRB[j-1]))
                e = int(math.floor(xindexRB[j]))
                sum[:] = (s+1 - xindexRB[j-1]) * imgin[:, 511 - s] + (xindexRB[j] - e) * imgin[:, 511 - e]
                if (e-s) > 1:  # have a middle pixel
                    sum[:] = sum[:] + imgin[:, 511-(s+1)]
                mask = (sum > maxlevel)  # saturated? for max image==1.0
                sum[mask] = maxlevel
                imgout[:, 511 - j] = sum
            else:
                imgout[:, 511 - j] = imgin[:, 511 - xindexR[j]] * bgfactorR
        else:
            imgout[:, 511 - j] = mean_modevalue * bgfactorR

    if FOVwidth == 512:
        img = imgout.astype(np.uint16)
        # img = imgout #test 8bit grid image
    else:
        img = imgout[:, L:512 - R].astype(np.uint16)

    return img


# noise_reduction in left and right dewarped zones by 2 * sigma
def xdewarp_nr1(imgin, FOVwidth):
    # Center
    # for j in range(aL,512-aR):   #near-flat area
    imgout = np.zeros(imgin.shape)
    imgout[:, (int(aL)):(512 - (int(aR)))] = imgin[:, (int(aL)):(512 - (int(aR)))]

    # sum=np.zeros(512, np.float)
    sum = np.zeros(imgin.shape[0], np.float)

    # Left side
    for j in range(0, int(aL)):
        sum[:] = 0.0  # reset
        if xindexL[j] >= 0:
            if xindexLB[j-1] >= 0.0:
                s = int(math.floor(xindexLB[j-1]))
                e = int(math.floor(xindexLB[j]))
                sum[:] = (s+1 - xindexLB[j-1]) * imgin[:, s] + (xindexLB[j] - e) * imgin[:, e]
                if (e-s) > 1:   # have a middle pixel
                    sum[:] = sum[:] + imgin[:, s+1]
                # imgout[:,j] = sum
                # ### reduce accumulated bg white noise
                nf = 2 * stdv_modevalue * (xindexLB[j] - xindexLB[j-1] - 1)
                sum = sum - nf
                mask = (sum < 0)  # underflow?
                sum[mask] = 0
                mask = (sum > maxlevel)  # saturated?  for max image==1.0
                sum[mask] = maxlevel
                imgout[:, j] = sum
            else:
                imgout[:, j] = imgin[:, xindexL[j]]
        else:
            imgout[:, j] = mean_modevalue

    # Right side
    for j in range(0, (int(aR))):
        sum[:] = 0.0
        if xindexR[j] >= 0:
            if xindexRB[j-1] >= 0.0:
                s = int(math.floor(xindexRB[j-1]))
                e = int(math.floor(xindexRB[j]))
                sum[:] = (s+1 - xindexRB[j-1]) * imgin[:, 511-s] + (xindexRB[j] - e) * imgin[:, 511-e]
                if (e-s) > 1:   # have a middle pixel
                    sum[:] = sum[:] + imgin[:, 511-(s+1)]
                # imgout[:,511-j] = sum
                # ### reduce estimated bg white noise
                nf = 2 * stdv_modevalue * (xindexRB[j] - xindexRB[j-1] - 1)
                sum = sum - nf
                mask = (sum < 0)  # underflow?
                sum[mask] = 0
                mask = (sum > maxlevel)  # saturated? for max image==1.0
                sum[mask] = maxlevel
                imgout[:, 511-j] = sum
            else:
                imgout[:, 511-j] = imgin[:, 511-xindexR[j]]
        else:
            imgout[:, 511-j] = mean_modevalue

    if FOVwidth == 512:
        img = imgout.astype(np.uint16)
        # img = imgout #test 8bit grid image
    else:
        img = imgout[:, L:512-R].astype(np.uint16)

    return img


# noise_reduction in left and right dewarped zones, method 2: by 1 * sigma and normalized
def xdewarp_nr2(imgin, FOVwidth):
    # Center
    # for j in range(aL,512-aR):   #near-flat area
    imgout = np.zeros(imgin.shape)
    imgout[:, (int(aL)):(512-(int(aR)))] = imgin[:, (int(aL)):(512-(int(aR)))]

    # sum=np.zeros(512, np.float)
    sum = np.zeros(imgin.shape[0], np.float)

    # Left side
    for j in range(0, int(aL)):
        sum[:] = 0.0  # reset
        if xindexL[j] >= 0:
            if xindexLB[j-1] >= 0.0:
                s = int(math.floor(xindexLB[j-1]))
                e = int(math.floor(xindexLB[j]))
                sum[:] = (s+1 - xindexLB[j-1]) * imgin[:, s] + (xindexLB[j] - e) * imgin[:, e]
                if (e-s) > 1:   # have a middle pixel
                    sum[:] = sum[:] + imgin[:, s+1]
                # imgout[:,j] = sum
                # ### reduce accumulated bg white noise
                nf = 1 * stdv_modevalue * (xindexLB[j] - xindexLB[j-1] - 1)
                sum = (sum / (xindexLB[j] - xindexLB[j-1])) - nf
                mask = (sum < 0)  # underflow?
                sum[mask] = 0
                mask = (sum > maxlevel)  # saturated?  for max image==1.0
                sum[mask] = maxlevel
                imgout[:, j] = sum
            else:
                imgout[:, j] = imgin[:, xindexL[j]]
        else:
            imgout[:, j] = mean_modevalue

    # Right side
    for j in range(0, (int(aR))):
        sum[:] = 0.0
        if xindexR[j] >= 0:
            if xindexRB[j-1] >= 0.0:
                s = int(math.floor(xindexRB[j-1]))
                e = int(math.floor(xindexRB[j]))
                sum[:] = (s+1 - xindexRB[j-1]) * imgin[:, 511-s] + (xindexRB[j] - e) * imgin[:, 511-e]
                if (e-s) > 1:   # have a middle pixel
                    sum[:] = sum[:] + imgin[:, 511-(s+1)]
                # imgout[:,511-j] = sum
                # ### reduce estimated bg white noise
                nf = 1 * stdv_modevalue * (xindexRB[j] - xindexRB[j-1] - 1)
                sum = (sum / (xindexRB[j] - xindexRB[j-1])) - nf
                mask = (sum < 0)  # underflow?
                sum[mask] = 0
                mask = (sum > maxlevel)  # saturated? for max image==1.0
                sum[mask] = maxlevel
                imgout[:, 511-j] = sum
            else:
                imgout[:, 511-j] = imgin[:, 511-xindexR[j]]
        else:
            imgout[:, 511-j] = mean_modevalue

    if FOVwidth == 512:
        img = imgout.astype(np.uint16)
        # img = imgout #test 8bit grid image
    else:
        img = imgout[:, L:512-R].astype(np.uint16)

    return img


# noise_reduction in left and right dewarped zones, method 3: normalized only
# no sigma reduction
def xdewarp_nr3(imgin, FOVwidth):
    # Center
    # for j in range(aL,512-aR):   #near-flat area
    imgout = np.zeros(imgin.shape)
    imgout[:, (int(aL)):(512-(int(aR)))] = imgin[:, (int(aL)):(512-(int(aR)))]

    # sum=np.zeros(512, np.float)
    sum = np.zeros(imgin.shape[0], np.float)

    # Left side
    for j in range(0, int(aL)):
        sum[:] = 0.0  # reset
        if xindexL[j] >= 0:
            if xindexLB[j-1] >= 0.0:
                s = int(math.floor(xindexLB[j-1]))
                e = int(math.floor(xindexLB[j]))
                sum[:] = (s + 1 - xindexLB[j-1]) * imgin[:, s] + (xindexLB[j] - e) * imgin[:, e]
                if (e-s) > 1:   # have a middle pixel
                    sum[:] = sum[:] + imgin[:, s+1]
                # imgout[:,j] = sum
                # ### reduce accumulated bg white noise
                # nf = 1* stdv_modevalue *(xindexLB[j] - xindexLB[j-1] - 1)
                # sum = (sum / (xindexRB[j] - xindexRB[j-1])) - nf
                sum = sum / (xindexLB[j] - xindexLB[j-1])
                mask = (sum < 0)  # underflow?
                sum[mask] = 0
                mask = (sum > maxlevel)  # saturated?  for max image==1.0
                sum[mask] = maxlevel
                imgout[:, j] = sum
            else:
                imgout[:, j] = imgin[:, xindexL[j]]
        else:
            imgout[:, j] = mean_modevalue

    # Right side
    for j in range(0, (int(aR))):
        sum[:] = 0.0
        if xindexR[j] >= 0:
            if xindexRB[j-1] >= 0.0:
                s = int(math.floor(xindexRB[j-1]))
                e = int(math.floor(xindexRB[j]))
                sum[:] = (s + 1 - xindexRB[j-1]) * imgin[:, 511 - s] + (xindexRB[j] - e) * imgin[:, 511 - e]
                if (e-s) > 1:   # have a middle pixel
                    sum[:] = sum[:] + imgin[:, 511-(s+1)]
                # imgout[:,511-j] = sum
                # ### reduce estimated bg white noise
                # nf = 1* stdv_modevalue *(xindexRB[j] - xindexRB[j-1] - 1)
                # sum = (sum / (xindexRB[j] - xindexRB[j-1])) - nf
                sum = sum / (xindexRB[j] - xindexRB[j-1])
                mask = (sum < 0)  # underflow?
                sum[mask] = 0
                mask = (sum > maxlevel)  # saturated? for max image==1.0
                sum[mask] = maxlevel
                imgout[:, 511 - j] = sum
            else:
                imgout[:, 511 - j] = imgin[:, 511 - xindexR[j]]
        else:
            imgout[:, 511 - j] = mean_modevalue

    if FOVwidth == 512:
        img = imgout.astype(np.uint16)
        # img = imgout #test 8bit grid image
    else:
        img = imgout[:, L:512-R].astype(np.uint16)

    return img


def parse_input(data):
    input_h5 = data.get('input_h5', None)
    if input_h5 is None:
        raise KeyError("input JSON does not have required field: 'input_h5'")
    if not os.path.exists(input_h5):
        raise IOError("%s does not exist" % input_h5)

    output_h5 = data.get('output_h5', None)
    if output_h5 is None:
        raise KeyError("input JSON does not have required field: 'output_h5'")

    equipment_name = data.get("equipment_name", None)
    if equipment_name is None:
        raise KeyError("input JSON does not have required field: 'equipment_name'")

    # set global variable
    global aL
    global aR
    global bL
    global bR

    # get paramaterss from scientifica_parameters.py 
    # equipment_params = sp.EQUIPMENT_PARAMETER_FILES.get(equipment_name, None)
    # if equipment_params is None:
    #    raise KeyError('parameters are not determined for this microscope: %s' % equipment_name)

    # aL = float(equipment_params.get('aL', None))
    # aR = float(equipment_params.get('aR', None))
    # bL = float(equipment_params.get('bL', None))
    # bR = float(equipment_params.get('bR', None))

    aLstr = data.get("aL", None)
    if aLstr is None:
        raise KeyError('parameter aL not determined for this microscope: %s' % equipment_name)
    aL = float(aLstr)

    aRstr = data.get("aR", None)
    if aRstr is None:
        raise KeyError('parameter aR not determined for this microscope: %s' % equipment_name)
    aR = float(aRstr)

    bLstr = data.get("bL", None)
    if bLstr is None:
        raise KeyError('parameter bL not determined for this microscope: %s' % equipment_name)
    bL = float(bLstr)

    bRstr = data.get("bR", None)
    if bRstr is None:
        raise KeyError('parameter bR not determined for this microscope: %s' % equipment_name)
    bR = float(bRstr)

    print(str(equipment_name))
    print('aL bL', aL, bL)
    print('aR bR', aR, bR)

    return input_h5, output_h5, aL, aR, bL, bR


def run_image(frame, FOVwidth, noise_reduction,
              input_file, input_dataset,
              output_file, output_dataset):

    LOCK.acquire()
    f = h5py.File(input_file, "r")
    image = f[input_dataset][frame, :, :]
    f.close()
    LOCK.release()

    if noise_reduction == 1:
        imgout = xdewarp_nr1(image, FOVwidth)
    elif noise_reduction == 2:
        imgout = xdewarp_nr2(image, FOVwidth)
    elif noise_reduction == 3:
        imgout = xdewarp_nr3(image, FOVwidth)
    else:
        imgout = xdewarp(image, FOVwidth)
    print("frame done ", frame)

    LOCK.acquire()
    f = h5py.File(output_file, "a")
    f[output_dataset][frame, :, :] = imgout
    f.close()
    LOCK.release()

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    parser.add_argument('output_json')
    parser.add_argument('--log_level', default=logging.DEBUG)
    # parser.add_argument('--threads', default=8, type=int)
    parser.add_argument('--threads', default=4, type=int)
    parser.add_argument('--FOVwidth', default=0, type=int)
    parser.add_argument('--noise_reduction', default=0, type=int)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    print("noise_reduction=", args.noise_reduction)

    input_data = ju.read(args.input_json)
    input_h5, output_h5, aL, aR, bL, bR = parse_input(input_data)

    logging.debug("got parameters from json")

    # -- test read only --
    # exit(0)

    # dewarp the movie
    movie_file = h5py.File(input_h5)
    movie = movie_file['data']
    # img = movie[0]
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
    movie_file.close()

    if os.path.exists(output_h5):
        os.remove(output_h5)

    # ------ grid image check only------
    '''
    print "Test grid image"
    #img = plt.imread("2p5_grid_locC_1p2x_8bit.png") # 8bit
    img = plt.imread("med_20160720_2p4_GridImages_2P.png") # 8bit
    create_xtable(img, aL, aR, bL, bR)
    maxlevel= 1.0 #grid 8bit png
    imgout=xdewarp(img, 512 )
    # to PIL
    image = Image.fromarray((imgout[:,:]*255).astype(np.uint8)) # 8bit grid png
    image.save('imgout_2p5.png') 
    exit(0)
    '''
    # ------ end grid image check only------

    # need one frame from movie for mode
    # IMPORTANT!: estmated modevalue is from 8bit img but convert back to 16bit
    create_xtable(meanimg, stdvimg, aL, aR, bL, bR)
    logging.debug("done create_xtable")

    logging.debug("create output file %s", output_h5)
    dewarped_file = h5py.File(output_h5, 'w')
    logging.debug("create data set 'data'")
    if args.FOVwidth == 512:
        dewarped_file.create_dataset('data', shape=movie_shape, dtype=movie_dtype)
    else:
        out_shape = [movie_shape[0], movie_shape[1], 512-R-L]
        dewarped_file.create_dataset('data', shape=out_shape, dtype=movie_dtype)
    dewarped_file.close()
    logging.debug("done creating output file")

    start_time = time.time()

    # T = 1000    #for testing sub movie

    fn = functools.partial(run_image, FOVwidth=args.FOVwidth, noise_reduction=args.noise_reduction,
                           input_file=input_h5, input_dataset="data",
                           output_file=output_h5, output_dataset="data")

    pool = multiprocessing.Pool(args.threads)
    result = pool.map_async(fn, range(T), chunksize=1)

    while not result.ready():
        num_finished = T - result._number_left
        frac = float(num_finished) / T
        fps = float(num_finished) / (time.time() - start_time)
        logging.debug("finished: %d/%d, %f%%, fps: %f" % (num_finished, T, frac * 100, fps))
        # time.sleep(10)
        time.sleep(2)

    # rame_list = result.get()   # unnecessary

    pool.close()
    pool.join()

    end_time = time.time()

    logging.debug("Elapsed time (s): %f", end_time-start_time)

    output_data = {}

    ju.write(args.output_json, output_data)


if __name__ == '__main__':
    main()
