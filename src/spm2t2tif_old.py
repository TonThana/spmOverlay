# %%

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.io import loadmat
import math
import matplotlib
import matplotlib.cm as cm
from copy import copy
import matplotlib.pyplot as plt
import nibabel
import numpy as np

# %%

# Process data

T1WImgPath = 'TestData/SPM2T2TIF/WN/601375499_301_AX_T1W_REF_2D_20190925/WUTTHIPHAT_NAEWKLANG_5508793_20190925_601375499_301_AX_T1W_REF_2D.img'

T1WHdrPath = 'TestData/SPM2T2TIF/WN/601375499_301_AX_T1W_REF_2D_20190925/WUTTHIPHAT_NAEWKLANG_5508793_20190925_601375499_301_AX_T1W_REF_2D.hdr'

t1w_data = nibabel.load(T1WImgPath)

anatomy = t1w_data.get_data()[:, :, :, 0]
max_anat_value = np.amax(anatomy)
min_anat_value = np.amin(anatomy)
##
xw, yw, zw = anatomy.shape

# %%
# follow matlab code
# Argument input
sc_sd1 = 3.5  # Global +- *std range
sc_sd = 2.0  # Local Slice +- *std range
#
max_int = np.amax(anatomy)
min_int = np.amin(anatomy)
AnatR = anatomy - min_int
Ar = np.reshape(AnatR, (zw*yw*xw, 1))
max_median = np.median(Ar)
max_std = np.std(Ar)
max_intensity = max_median + sc_sd1*max_std

mask = AnatR > max_intensity
AnatR[mask] = max_intensity

# %%
zero4b = 7.0
slope = 256/max_intensity
AnatR_down = AnatR * slope / 2  # 0 -> 128
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
                AnatR_down[i, j, sl] = slope_u*(act_max_sli - mean_sli) + new_m
            if (AnatR_down[i, j, sl] <= 2):
                AnatR_down[i, j, sl] = 1
            elif (AnatR_down[i, j, sl] >= 127):
                AnatR_down[i, j, sl] = 127
        # j
    # i
# sl

# bright-ish Anat
Anat = np.round(AnatR_down)
#plt.imshow(Anat[:,:,10], cmap='gray')

# %%

# Load spmT map and filter nan and value less than threshold
spmTPath = 'TestData/SPM2T2TIF/WN/601375499_501_fMRI_Rt_Hand_Tapping_20190925/rspmT_0003.img'

spmT_data = nibabel.load(spmTPath)
# (512,512,24,1)
##

spmT_nochannel = spmT_data.get_data()[:, :, :, 0]
# (512,512,24)

threshold = 7.15

noNanMask = np.isnan(spmT_nochannel)
spmT_nochannel[noNanMask] = 0.0
thresholdMask = spmT_nochannel <= threshold
spmT_nochannel[thresholdMask] = 0.0

spmT_max = np.amax(spmT_nochannel)
spmT_min = np.amin(spmT_nochannel)
print(spmT_max, spmT_min)

# %%
my_norm = matplotlib.colors.Normalize(
    clip=False)
my_cmap = copy(cm.get_cmap('inferno'))
my_cmap.set_under('w', alpha=0)
my_cmap.set_bad('w', alpha=0)

#

# %%


fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(4, 6), facecolor=(0, 0, 0))

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

#plt.subplots_adjust(wspace=0, hspace=0)

fig.savefig('example.png', bbox_inches='tight', pad_inches=0,
            dpi=1000, facecolor=fig.get_facecolor())
# plt.show()
# %%
