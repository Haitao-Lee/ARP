# coding = utf-8
import numpy as np
import os
import struct
# from pathlib import Path
from config import args
from tqdm import tqdm


def get_file_names(data_dir, origin_lut, origin_peak):
    lut_names = []
    peak_X_names = []
    peak_Y_names = []
    for root, folders, _ in os.walk(data_dir):
        for i in folders:
            for _, _, files in os.walk(root + '/' + i + '/' + origin_lut):
                for k in range(len(files)):
                    lut_names.append(root + '/' + i + '/' + origin_lut +
                                     str('/lut%d' % (k + 1)))
            for _, _, files in os.walk(root + '/' + i + '/' + origin_peak):
                for k in range(len(files)):
                    if k < 36:
                        peak_X_names.append(root + '/' + i + '/' +
                                            origin_peak + str('/X Peak%d' %
                                                              (k + 1)))
                    else:
                        peak_Y_names.append(root + '/' + i + '/' +
                                            origin_peak + str('/Y Peak%d' %
                                                              (k + 1 - 36)))
    return lut_names, peak_X_names, peak_Y_names


def load_origin_luts(names):
    luts = []
    for i in tqdm(
            range(len(names)),
            desc="\033[31mLoading luts:\033[0m",
    ):
        name = names[i]
        raw_lut = open(name, 'rb')
        size = os.path.getsize(name)
        tmp = []
        for j in range(size // 4):
            data = raw_lut.read(4)
            value = struct.unpack('>I', data)
            tmp.append(value[0])
        lut = np.array(tmp).reshape([1024, 1024])
        luts.append(lut)
    luts = np.array(luts)
    return luts


def load_origin_peaks(x_names, y_names):
    peaks_x = []
    peaks_y = []
    for i in tqdm(
            range(len(x_names)),
            desc="\033[31mLoading peaks:\033[0m",
    ):
        x_name = x_names[i]
        y_name = y_names[i]
        peak_x_f = open(str(x_name), 'rb')
        peak_y_f = open(str(y_name), 'rb')
        size_x = os.path.getsize(str(x_name))
        size_y = os.path.getsize(str(y_name))
        assert size_x == size_y, f'size_x: {size_x}; size_y: {size_y}'
        peak_xs = []
        peak_ys = []
        # tmp_peaks = []
        for i in range(size_x // 4):
            data_x = peak_x_f.read(4)
            num_x = struct.unpack('>I', data_x)
            peak_xs.append(num_x[0])
            data_y = peak_y_f.read(4)
            num_y = struct.unpack('>I', data_y)
            peak_ys.append(num_y[0])
            # tmp_peaks.append(np.array([num_x[0], num_y[0]]))
        # tmp_peaks = np.array(tmp_peaks)
        peak_xs = np.array(peak_xs).reshape([64, 64])
        peak_ys = np.array(peak_ys).reshape([64, 64])
        peaks_x.append(peak_xs)
        peaks_y.append(peak_ys)
    peaks_x = np.array(peaks_x)
    peaks_y = np.array(peaks_y)
    return peaks_x, peaks_y


def load_training_data():
    luts = []
    peaks = []
    for path in os.listdir(args.reg_luts_npy + "/training/"):
        luts.append(np.load(args.reg_luts_npy + "/training/" + path))
    for path in os.listdir(args.reg_peaks_npy + "/training/"):
        peaks.append(np.load(args.reg_peaks_npy + "/training/" + path))
    return luts, peaks


def load_validation_data():
    luts = []
    peaks = []
    for path in os.listdir(args.reg_luts_npy + "/validation/"):
        luts.append(np.load(args.reg_luts_npy + "/validation/" + path))
    for path in os.listdir(args.reg_peaks_npy + "/validation/"):
        peaks.append(np.load(args.reg_peaks_npy + "/validation/" + path))
    return luts, peaks