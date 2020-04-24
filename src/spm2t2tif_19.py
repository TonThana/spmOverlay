
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
from functools import reduce

PT_p_factor = 0.001


def factors(n):
    return tuple(reduce(list.__add__,
                        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
# (1, 24, 2, 12, 3, 8, 4, 6)


def normalizeAnatomy(anatomy):
    sc_sd1 = 3.5  # Global +- *std range
    sc_sd = 2.0  # Local Slice +- *std range
    zero4b = 7.0
    # max_int = np.amax(anatomy)
    min_int = np.amin(anatomy)
    # max_anat_value = np.amax(anatomy)
    # min_anat_value = np.amin(anatomy)
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
    print(spmT_data.get_data().shape)

    spmT_orig = spmT_data.get_data()

    if (len(spmT_orig.shape) == 3):
        spmT_nochannel = spmT_orig
    else:
        spmT_nochannel = spmT_orig[:, :, :, 0]

    print("threshold", threshold)
    noNanMask = np.isnan(spmT_nochannel)
    spmT_nochannel[noNanMask] = 0.0

    spmT_max = np.amax(spmT_nochannel)
    print("spmT_max", spmT_max)
    if (threshold == None):
        threshold = PT_p_factor * spmT_max

    thresholdMask = spmT_nochannel <= threshold
    spmT_nochannel[thresholdMask] = 0.0

    return spmT_nochannel, spmT_max


def plotOverlay(Anat, spmT_nochannel, threshold, spmT_max, rownum, colnum, outputFolder):
    if (threshold == None):
        # for pt use log scale
        my_norm = matplotlib.colors.LogNorm(clip=False)
        threshold = spmT_max * PT_p_factor
        colormapName = 'jet'
    else:
        my_norm = matplotlib.colors.Normalize(
            clip=False)
        colormapName = 'spring'

    # CHANGE COLORMAP HERE  https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

    my_cmap = copy(cm.get_cmap(colormapName))
    my_cmap.set_under('w', alpha=0)
    my_cmap.set_bad('w', alpha=0)

    fig, axs = plt.subplots(
        nrows=rownum, ncols=colnum, figsize=(colnum, rownum), facecolor=(0, 0, 0))

    # threshold_removeFS = str(threshold).replace(".", "_")
    xw = None
    yw = None
    for i in range(int(rownum*colnum)):
        # print(xw, yw)
        if (i == 19):
            row = int(np.floor(i/colnum))
            col = int(i % colnum)
            ax = axs[row, col]
            ax.imshow(np.zeros(shape=(xw, yw)),
                      cmap='gray', interpolation='nearest')
            ax.axis('off')
            ax.set_aspect('equal')
            continue

        if (i == 0):
            xw = Anat.shape[0]
            yw = Anat.shape[1]

        row = int(np.floor(i/colnum))
        col = int(i % colnum)
        ax = axs[row, col]
        ax.imshow(np.rot90(Anat[:, :, i]),
                  cmap='gray', interpolation='nearest')
        tmap = ax.imshow(np.rot90(spmT_nochannel[:, :, i]),
                         cmap=my_cmap,
                         norm=my_norm,
                         vmin=threshold,
                         vmax=spmT_max,
                         alpha=0.85,
                         interpolation='nearest')
        ax.axis('off')
        ax.set_aspect('equal')

        # Saving indiv slice still broken (contaminated with 0S somthing...)

        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # print('{outputFolder}/overlay_slice{sli}_th{threshold}.png'.format(
        #     sli=i, threshold=threshold_removeFS, outputFolder=outputFolder))

        # fig.savefig('{outputFolder}/overlay_slice{sli}_th{threshold}.png'.format(
        #     sli=i, threshold=threshold_removeFS, outputFolder=outputFolder), bbox_inches=extent, dpi=800, facecolor=fig.get_facecolor(),  pad_inches=0)

        if (row == 0 and col == colnum-1):
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
    spm_x, spm_y, spm_z = spmT_nochannel.shape
    # 1) CHECK DIM
    if (spm_x != xw or spm_y != yw or spm_z != zw):
        raise Exception('Dimension of overlay subject and template mismatch')

    # 2) BASE ON DIM - MONTAGE ROW AND COL -> pass to plotOverlay
    # factors_list = factors(zw)
    # rownum = factors_list[-1]
    # colnum = factors_list[-2]
    # print("ROWNUM={}".format(rownum))
    # print("COLNUM={}".format(colnum))
    # print("ROW * COL", int(rownum*colnum))
    # assert (int(rownum * colnum) ==
    #         zw), "something went wrong with slice number factorisation"

    rownum = 5
    colnum = 4

    # 3) In plotOverlay save each overlayed slice with unique name into folder {parent}/PNG_RESULT_PY/ (os.makedirs this)
    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory

    spmFolder = Path(spmTmapPath).parent
    print("***OUTPUT FOLDER: ", spmFolder)
    outputFolder = "{parent}/PNG_RESULT_PY".format(parent=spmFolder)

    try:
        os.makedirs(outputFolder)
    except FileExistsError:
        print('{parent} FOLDER already exist'.format(parent=outputFolder))

    fig = plotOverlay(Anat, spmT_nochannel, threshold,
                      spmT_max, rownum, colnum, outputFolder)

    outputFileName = '{parent}/montage.png'.format(
        parent=outputFolder)
    # print(outputFileName)
    fig.savefig(outputFileName, bbox_inches='tight', pad_inches=0,
                dpi=1000, facecolor=fig.get_facecolor())


if __name__ == '__main__':
    arg_names = ['pyFile', 't1ImgPath', 'spmTmapPath', 'threshold']
    Arg_list = collections.namedtuple('Arg_list', arg_names)
    # print(Arg_list)
    args = dict(zip(arg_names, sys.argv))
    args = Arg_list(*(args.get(arg, None) for arg in arg_names))

    t1ImgPath = str(args[1])
    spmTmapPath = str(args[2])
    print(Path(spmTmapPath).name)
    if (Path(spmTmapPath).name == "rPT.img"):
        threshold = None  # leave blank first
    else:
        threshold = float(args[3])
    # print(Path(spmTmapPath))

    main(t1ImgPath, spmTmapPath, threshold)
