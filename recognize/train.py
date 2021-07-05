import numpy as np
import cv2
import os
import csv
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from PIL import Image, ImageChops
import PIL.ImageOps
import statistics


def has_black_bg(file_path):
    img = Image.open(file_path)
    gray = img.convert('L')
    bw = np.asarray(gray).copy()
    bw[bw < 128] = 0  # Black
    bw[bw >= 128] = 255  # White
    black = 0
    white = 0
    for pixel in bw[0]:
        if pixel == 0:
            black += 1
        else:
            white += 1

    for pixel in bw[-1]:
        if pixel == 0:
            black += 1
        else:
            white += 1
    if black > white:
        return True
    else:
        return False


def convert_to_bw(file_path, black_bg):
    img = Image.open(file_path)
    gray = img.convert('L')
    bw = np.asarray(gray).copy()
    bw[bw < 128] = 0  # Black
    bw[bw >= 128] = 255  # White

    if not black_bg:
        inv = PIL.ImageOps.invert(gray)
        bw = np.asarray(inv).copy()
        bw[bw < 128] = 0  # Black
        bw[bw >= 128] = 255  # White

    else:
        bw = np.asarray(gray).copy()
        bw[bw < 128] = 0  # Black
        bw[bw >= 128] = 255  # White

    return bw


def convert_to_bw1(file_path):
    img = Image.open(file_path)
    gray = img.convert('L')
    inv = PIL.ImageOps.invert(gray)
    bw = np.asarray(inv).copy()
    bw[bw < 128] = 0  # Black
    bw[bw >= 128] = 255  # White
    return bw


def trim(bw):
    im = Image.fromarray(bw)

    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        i = im.crop(bbox)
        # i.save('../dataset/numero2/1.png')
        cropped = np.asarray(i)
        return cropped


def euler_numbers(param):
    lp = np.pad(param, ((1, 0), (1, 0)), 'constant')

    i_nw = lp[:-1, :-1]
    i_n = lp[:-1, 1:]
    i_w = lp[1:, :-1]

    is_upstream_convexity = np.logical_and(param, (param != i_n))
    is_upstream_convexity = np.logical_and(is_upstream_convexity, (param != i_nw))
    is_upstream_convexity = np.logical_and(is_upstream_convexity, (param != i_w))

    is_upstream_concavity = np.logical_and(param, (param != i_nw))
    is_upstream_concavity = np.logical_and(is_upstream_concavity, (param == i_n))
    is_upstream_concavity = np.logical_and(is_upstream_concavity, (param == i_w))

    upstream_convexity_labels = param[is_upstream_convexity]
    upstream_concavity_labels = param[is_upstream_concavity]

    total_upstream_convexities = np.bincount(upstream_convexity_labels)[
                                 1:]  # Discard the zero bin, which is the background.
    total_upstream_concavities = np.bincount(upstream_concavity_labels)[1:]
    return total_upstream_convexities - total_upstream_concavities


def number_of_holes(bw):
    # Load the image
    _, l = cv2.connectedComponents(bw)
    e = euler_numbers(l)

    # All the regions with no holes will have an Euler number of 1. Regions with one hole
    # will have an Euler number of 0. Two holes -> -1 etc.

    num_single_hole = np.sum(e == 0)
    num_two_holes = np.sum(e == -1)
    num_three_holes = np.sum(e == -2)
    num_more_holes = np.sum(e < -2)
    if num_single_hole != 0:
        return 1
    if num_two_holes != 0:
        return 2
    if num_three_holes != 0:
        return 3
    return 0


def numbe_of_angles(bw):
    coords = corner_peaks(corner_harris(bw), min_distance=1, threshold_rel=0.5)
    return len(coords)


def number_of_pixels(bw):
    count = 0
    c = 0
    for line in bw:
        for pixel in line:
            if pixel == 255:
                count += 1

    return count / (len(bw) * len(bw[0]))


def symetrie(bw):
    count11, count12, count21, count22 = [0, 0, 0, 0]
    height = len(bw)
    width = len(bw[0])
    half_height = int(len(bw) / 2)
    half_width = int(len(bw[0]) / 2)
    for i in range(half_height):
        for j in range(half_width):
            if bw[i][j] == 255:
                count11 += 1
        for j in range(half_width, width):
            if bw[i][j] == 255:
                count12 += 1

    for i in range(half_height, height):
        for j in range(half_width):
            if bw[i][j] == 255:
                count21 += 1
        for j in range(half_width, width):
            if bw[i][j] == 255:
                count22 += 1

    h = (count11 + count12) / (count21 + count22)
    v = (count11 + count21) / (count12 + count22)

    return h, v


def heightByWidth(bw):
    return len(bw) / len(bw[0])


def VTD_HTD(bw):
    vtd = []
    htd = []

    h = len(bw)
    w = len(bw[0])
    # calculate htd

    for i in range(h):
        htd.append(0)
        for j in range(w - 1):
            if (bw[i][j] == 0 and bw[i][j + 1] == 255):
                htd[i] += 1

    # calculate vtd

    for i in range(w):
        vtd.append(0)
        for j in range(h - 1):
            if (bw[j][i] == 0 and bw[j + 1][i] == 255):
                vtd[i] += 1
    return statistics.mean(vtd), statistics.stdev(vtd), statistics.mean(htd), statistics.stdev(htd)


def bottom_line_density(bw, black_bg):
    h = len(bw)
    w = len(bw[h - 1])
    count = 0
    if black_bg:
        for pixel in bw[h - 2]:
            if pixel != 0:
                count += 1
    else:
        for pixel in bw[h - 2]:
            if pixel == 0:
                count += 1
    return count / w


def hu_moments(bw):
    HuMoments = cv2.HuMoments(cv2.moments(bw)).flatten()
    return HuMoments


def largeur(bw, black_bg):
    if black_bg:
        pass


def dataset(images_path, black_bg=False):
    dataset = []
    classes = os.listdir(images_path)
    for label in classes:
        images = os.listdir(images_path + '/' + label)
        for image in images:
            print(images_path + '/' + label + '/' + image)
            black_bg = has_black_bg(images_path + '/' + label + '/' + image)
            cropped = convert_to_bw(images_path + '/' + label + '/' + image, black_bg)

            n = number_of_holes(cropped)
            mean_vtd, stddev_vtd, mean_htd, stddev_htd = VTD_HTD(cropped)
            cropped = trim(cropped)
            horizental_symetry, vertical_symetry = symetrie(cropped)
            bld = bottom_line_density(cropped, black_bg)

            v = list()

            v.append(n)
            v.append(horizental_symetry)
            v.append(vertical_symetry)
            v.append(mean_vtd)
            v.append(stddev_vtd)
            v.append(mean_htd)
            v.append(stddev_htd)
            v.append(bld)
            v.append(label)
            # v.append(label + '/' + image)
            dataset.append(v)

    with open('attributes_modified.csv', 'w', newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["number_of_holes", "horizental_symetry", "vertical_symetry", "mean_vtd", "stddev_vtd", "mean_htd",
             "stddev_htd", "bottom_line_density", "Class"])

        writer.writerows(dataset)
    return dataset


imagesPath = '../dataset/numero'

#dataset(imagesPath, black_bg=False)
