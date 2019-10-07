
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.io import loadmat
import math
import matplotlib
import matplotlib.cm as cm
from copy import copy
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import sys
from pathlib import Path
import collections
import os
import errno


def normalizeAnatomy(anatomy):
    sc_sd1 = 3.5  # Global +- *std range
    sc_sd = 2.0  # Local Slice +- *std range
    zero4b = 7.0
    max_int = np.amax(anatomy)
    min_int = np.amin(anatomy)
    max_anat_value = np.amax(anatomy)
    min_anat_value = np.amin(anatomy)
    xw, yw, zw = anatomy.shape
    AnatR = anatomy - min_int
    Ar = np.reshape(AnatR, (zw*yw*xw, 1))
    max_median = np.median(Ar)
    max_std = np.std(Ar)
    max_intensity = max_median + sc_sd1*max_std
    mask = AnatR > max_intensity
    AnatR[mask] = max_intensity
    slope = 256/max_intensity
    AnatR_down = AnatR * slope / 2
    for sl in range(0, zw):
        sli = np.reshape(AnatR_down[:, :, sl], (1, xw*yw))
        sli_v = np.array([])
        # mask close to zero out of stat (how close ? less than 7)
        # for px in range(0, xw*yw):
        #     # print(sli[0, px])
        #     if (sli[0, px] >= zero4b):
        #         sli_v = np.append(sli_v, sli[0, px])
        mask = sli >= zero4b
        sli_v = sli[mask]
        # print(sli_v.shape)
        max_sli = np.amax(sli_v)
        min_sli = np.amin(sli_v)
        mean_sli = np.mean(sli_v)
        std_sli = np.std(sli_v)
        max_stat_sli = mean_sli + sc_sd*std_sli
        min_stat_sli = mean_sli - sc_sd*std_sli
        act_max_sli = max(max_sli, max_stat_sli)
        act_min_sli = min(min_sli, min_stat_sli)
        delta_u = act_max_sli - mean_sli
        delta_l = mean_sli - act_min_sli
        new_m = round((128-zero4b)/2)
        slope_u = (128-new_m)/delta_u
        slope_l = (new_m-zero4b)/delta_l
        # normalizing individual slice
        for i in range(0, xw):
            for j in range(0, yw):
                if ((AnatR_down[i, j, sl] >= mean_sli) and (AnatR_down[i, j, sl] <= act_max_sli)):
                    AnatR_down[i, j, sl] = slope_u * \
                        (AnatR_down[i, j, sl]-mean_sli)+new_m
                elif ((AnatR_down[i, j, sl] >= act_min_sli)and(AnatR_down[i, j, sl] < mean_sli)):
                    AnatR_down[i, j, sl] = slope_l * \
                        (AnatR_down[i, j, sl]-act_min_sli)+zero4b
                elif ((AnatR_down[i, j, sl] < act_min_sli)and(AnatR_down[i, j, sl] > 2)):
                    AnatR_down[i, j, sl] = zero4b
                elif (AnatR_down[i, j, sl] > act_max_sli):
                    AnatR_down[i, j, sl] = slope_u * \
                        (act_max_sli - mean_sli) + new_m
                if (AnatR_down[i, j, sl] <= 2):
                    AnatR_down[i, j, sl] = 1
                elif (AnatR_down[i, j, sl] >= 127):
                    AnatR_down[i, j, sl] = 127
            # j
        # i
    # sl
    Anat = np.round(AnatR_down)
    return Anat, xw, yw, zw


def maskSpmT(spmTmapPath, threshold):
    spmT_data = nibabel.load(spmTmapPath)
    spmT_nochannel = spmT_data.get_data()[:, :, :, 0]
    noNanMask = np.isnan(spmT_nochannel)
    spmT_nochannel[noNanMask] = 0.0
    thresholdMask = spmT_nochannel <= threshold
    spmT_nochannel[thresholdMask] = 0.0
    spmT_max = np.amax(spmT_nochannel)
    return spmT_nochannel, spmT_max


def plotOverlay(Anat, spmT_nochannel, threshold, spmT_max):
    my_norm = matplotlib.colors.Normalize(
        clip=False)
    my_cmap = copy(cm.get_cmap('inferno'))
    my_cmap.set_under('w', alpha=0)
    my_cmap.set_bad('w', alpha=0)

    fig, axs = plt.subplots(
        nrows=6, ncols=4, figsize=(4, 6), facecolor=(0, 0, 0))

    # fig.dpi = 1000

    for i in range(int(6*4)):
        row = int(np.floor(i/4))
        col = int(i % 4)
        ax = axs[row, col]
        ax.imshow(np.rot90(Anat[:, :, i]),
                  cmap='gray', interpolation='nearest')
        tmap = ax.imshow(np.rot90(spmT_nochannel[:, :, i]),
                         cmap=my_cmap,
                         norm=my_norm,
                         vmin=threshold,
                         vmax=spmT_max,
                         alpha=0.5,
                         interpolation='nearest')
        ax.axis('off')
        ax.set_aspect('equal')
        if (row == 0 and col == 3):
            axins = inset_axes(parent_axes=ax, height="200%",
                               width="5%", loc='upper right', borderpad=0)
            axins.yaxis.set_ticks_position('left')
            axins.tick_params(colors='w', labelsize=5.0)
            cbar = fig.colorbar(tmap, cax=axins)
            cbar.set_alpha(1)
            cbar.draw_all()
    return fig


def main(t1ImgPath, spmTmapPath, threshold):
    # print(t1ImgPath, spmTmapPath, threshold)
    t1w_data = nibabel.load(t1ImgPath)
    anatomy = t1w_data.get_data()[:, :, :, 0]
    Anat, xw, yw, zw = normalizeAnatomy(anatomy)
    spmT_nochannel, spmT_max = maskSpmT(spmTmapPath, threshold)
    # 1) CHECK DIM
    # 2) BASE ON DIM - MONTAGE ROW AND COL -> pass to plotOverlay
    # 3) In plotOverlay save each overlayed slice with unique name into folder {parent}/PNG_RESULT_PY/ (os.makedirs this)
    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory

    spmFolder = Path(spmTmapPath).parent
    fig = plotOverlay(Anat, spmT_nochannel, threshold, spmT_max)

    outputFileName = '{parent}/montage.png'.format(
        parent=spmFolder)
    # print(outputFileName)
    fig.savefig(Path(outputFileName), bbox_inches='tight', pad_inches=0,
                dpi=1000, facecolor=fig.get_facecolor())


if __name__ == '__main__':
    arg_names = ['pyFile', 't1ImgPath', 'spmTmapPath', 'threshold']
    Arg_list = collections.namedtuple('Arg_list', arg_names)
    # print(Arg_list)
    args = dict(zip(arg_names, sys.argv))
    args = Arg_list(*(args.get(arg, None) for arg in arg_names))

    t1ImgPath = str(args[1])
    spmTmapPath = str(args[2])
    threshold = float(args[3])

    main(t1ImgPath, spmTmapPath, threshold)
