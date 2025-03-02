# coding = utf-8
import matplotlib.pyplot as plt
import data_input
import numpy as np
# import data_transform
import data_set
import os
from config import args
import random


def main():
    luts = []
    peaks = []
    for path in os.listdir(args.exp_train_lut):
        luts.append(np.load(args.exp_train_lut + '/' + path))
    for path in os.listdir(args.exp_train_peak):
        peaks.append(np.load(args.exp_train_peak + '/' + path))
    mean = data_set.mean_cordinate[0,:]
    mean.resize([256, 2])
    # x = data_set.mean_cordiate[0, 1::2]
    # y = data_set.mean_cordiate[0, 0::2]
    for i in range(len(luts)):
        #i = i + 600
        _, axes = plt.subplots(nrows=1, ncols=2)
        axes[1].matshow(luts[i],  alpha=1)
        #axes[1].plot(peaks[i][:,1],peaks[i][:,0], "yo")
        tmp = [x for x in range(256)]
        random.shuffle(tmp)
        peaks_tmp = peaks[i]
        peaks_ran = np.zeros(peaks_tmp.shape)
        for k in range(256):
            peaks_ran[k,:] = peaks_tmp[tmp[k], :]
        
        axes[1].plot(peaks[i][0::4,1], peaks[i][0::4,0],"ro")
        axes[1].plot(peaks[i][1::4,1], peaks[i][1::4,0],"yo")
        axes[1].plot(peaks[i][2::4,1], peaks[i][2::4,0],"wo")
        axes[1].plot(peaks[i][3::4,1], peaks[i][3::4,0],"bo")
        
        # axes[1].plot(peaks_ran[0::4,1], peaks_ran[0::4,0],"ro")
        # axes[1].plot(peaks_ran[1::4,1], peaks_ran[1::4,0],"yo")
        # axes[1].plot(peaks_ran[2::4,1], peaks_ran[2::4,0],"wo")
        # axes[1].plot(peaks_ran[3::4,1], peaks_ran[3::4,0],"bo")
        # axes[1].set_xlim([0, 255])
        # axes[1].set_ylim([0, 255])
        # tmp_lut = np.zeros(luts[i].shape)
        # for j in range(peaks[i].shape[0]):
        #     tmp_lut[min(max(0, peaks[i][j][0]), 255), min(max(0, peaks[i][j][1]), 255)] = 0.5*luts[i].max()
        axes[1].set_title("groundtruth%d"%i)
        # reg_peaks = peaks[i]
        axes[0].matshow(luts[i],  alpha=1)
        # axes[0].plot(reg_peaks[:,1], reg_peaks[:,0],"ro")
        axes[0].plot(mean[0::4,1], mean[0::4,0],"ro")
        axes[0].plot(mean[1::4,1], mean[1::4,0],"yo")
        axes[0].plot(mean[2::4,1], mean[2::4,0],"wo")
        axes[0].plot(mean[3::4,1], mean[3::4,0],"bo")
        axes[0].plot(mean[::16,1], mean[::16,0],"g*")
        # axes[0].set_xlim([0, 255])
        # axes[0].set_ylim([0, 255])
        axes[0].set_title("mean")
        plt.show()
    return 0


if __name__ == '__main__':
    main()
