#!/data/pnl/kcho/anaconda3/bin/python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nb
from scipy import ndimage
import functools


class Enigma:
    def __init__(self):
        self.enigma_dir = Path('/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
        self.enigma_fa_loc = self.enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
        self.enigma_fa_data = nb.load(str(self.enigma_fa_loc)).get_data()
        self.enigma_skeleton_mask_loc = self.enigma_dir / \
            'ENIGMA_DTI_FA_skeleton_mask.nii.gz'
        self.enigma_skeleton_data = nb.load(
            str(self.enigma_skeleton_mask_loc)).get_data()


class FigureSettings:
    def __init__(self):
        self.ncols = 5
        self.nrows = 4
        self.nslice = self.ncols * self.nrows
        self.size_w = 4
        self.size_h = 4
        self.slice_gap = 3
        self.dpi = 200

    def get_cbar_horizontal_info(self):
        self.cbar_x = 0.25
        self.cbar_y = 0.03
        self.cbar_height = 0.03
        self.cbar_width = 0.15

    def add_cbars_horizontal(self):
        self.cbar_x_steps = 0.2

        for num, image_data in enumerate(self.image_data_list):
            print(self.cbar_x)
            # x, y, width, height
            axbar = self.fig.add_axes([
                self.cbar_x,
                self.cbar_y, 
                self.cbar_width, 
                self.cbar_height])

            cb = self.fig.colorbar(
                    self.imshow_list[num],
                    axbar,
                    orientation='horizontal',
                    ticks=[0.05, 0.95])

            cb.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
            cb.outline.set_edgecolor('white')
            cb.ax.set_title(
                    self.cbar_titles[num],
                    fontweight='bold', color='white')
            cb.ax.yaxis.set_label_position('left')
            self.cbar_x += self.cbar_x_steps
    
class FigureNifti:
    def get_slice(self, data, z_num):
        return np.flipud(data[:, :, z_num].T)

    def transparent_mask(self, mask_data):
        mask_data = np.where(mask_data < 1, np.nan, mask_data)
        return mask_data

    def transparent_zero(self, data):
        data = np.where(data == 0, np.nan, data)
        return data

    def transparent_out_the_skeleton(self, data):
        data = np.where(self.enigma_skeleton_data == 1, data, np.nan)
        return data

    def get_center(self, data):
        # Get the center of data
        self.center_of_data = np.array(
            ndimage.measurements.center_of_mass(data)).astype(int)

        # Get the center slice number
        self.z_slice_center = self.center_of_data[-1]
        self.get_slice_nums()

    def get_slice_nums(self):
        self.slice_nums = np.arange(
            self.z_slice_center - (self.nslice * self.slice_gap),
            self.z_slice_center + (self.nslice * self.slice_gap),
            self.slice_gap)[::2]

    def annotate_with_z(self):
        for num, ax in enumerate(np.ravel(self.axes)):
            ax.annotate(
                f'z = {self.slice_nums[num]}',
                (0.01, 0.1),
                xycoords='axes fraction',
                color='white')

    def get_overlap_between_maps(self):
        self.overlap_map = np.where(
                (self.image_data_list[0] > 0) * (self.image_data_list[1] > 0),
                self.image_data_list[0], 
                np.nan)

        self.image_data_list.append(self.overlap_map)

class TbssFigure(Enigma, FigureSettings, FigureNifti):
    def __init__(self, image_files, out_img_file, **kwargs):
        Enigma.__init__(self)
        FigureSettings.__init__(self)

        self.get_center(self.enigma_fa_data)
        self.enigma_skeleton_data = self.transparent_mask(
            self.enigma_skeleton_data)

        self.fig, self.axes = plt.subplots(
            ncols=self.ncols,
            nrows=self.nrows,
            figsize=(self.size_w * self.ncols,
                     self.size_h * self.nrows),
            dpi=self.dpi)

        self.image_files = image_files
        self.image_data_list = [nb.load(x).get_data() for x
                                in self.image_files]
        self.out_img_file = out_img_file
        print('hoho')
        plt.style.use('dark_background')

    def images_mask_out_the_skeleton(self):
        new_images = []
        for image in self.image_data_list:
            new_image = self.transparent_out_the_skeleton(image)
            new_images.append(new_image)
        self.image_data_list = new_images

    def images_mask_out_the_zero(self):
        new_images = []
        for image in self.image_data_list:
            new_image = self.transparent_zero(image)
            new_images.append(new_image)
        self.image_data_list = new_images

    def image_mask_out_the_skeleton(self):
        self.image_data = self.transparent_out_the_skeleton(self.image_data)

    def loop_through_axes_draw_bg(self):
        for num, ax in enumerate(np.ravel(self.axes)):
            z_num = self.slice_nums[num]
            enigma_fa_d = self.get_slice(self.enigma_fa_data, z_num)
            enigma_skeleton_d = self.get_slice(
                    self.enigma_skeleton_data,
                    z_num)

            # background FA map
            img = ax.imshow(
                    enigma_fa_d,
                    cmap='gray')

            # background skeleton
            img = ax.imshow(
                    enigma_skeleton_d,
                    interpolation=None, cmap='ocean')
            ax.axis('off')

    def loop_through_axes_draw_images(self):
        for num, ax in enumerate(np.ravel(self.axes)):
            z_num = self.slice_nums[num]

            # background FA map
            self.imshow_list = []
            for num, image in enumerate(self.image_data_list):
                image_d = self.get_slice(image, z_num)
                img = ax.imshow(image_d, cmap=self.cmap_list[num])
                self.imshow_list.append(img)


# def get_slice(data, slice_nums, num):
    # return np.flipud(data[:, :, slice_nums[num]].T)


# def get_data(fa_filled, fat_filled, fw_filled):
    # fa_data = nb.load(fa_filled).get_data()
    # fat_data = nb.load(fat_filled).get_data()
    # fw_data = nb.load(fw_filled).get_data()

    # return fa_data, fat_data, fw_data

# def lupus_get_figures(fa_filled, fat_filled, outfile):
    # print('lupus_get_figures')

    # fa_data = nb.load(fa_filled).get_data()
    # fat_data = nb.load(fat_filled).get_data()

    # enigma_dir = Path('/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
    # enigma_fa_loc = enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
    # enigma_fa_data = nb.load(str(enigma_fa_loc)).get_data()

    # enigma_skeleton_mask_loc = enigma_dir / \
        # 'ENIGMA_DTI_FA_skeleton_mask.nii.gz'

    # enigma_skeleton_data = nb.load(str(enigma_skeleton_mask_loc)).get_data()

    # ncols = 5
    # nrows = 4
    # size_w = 4
    # size_h = 4
    # slice_gap = 3

    # # Get the center of data
    # center_of_data = np.array(
        # ndimage.measurements.center_of_mass(
            # enigma_fa_data)).astype(int)

    # # Get the center slice number
    # z_slice_center = center_of_data[-1]

    # # Get the slice numbers in array
    # nslice = ncols * nrows
    # slice_nums = np.arange(z_slice_center-(nslice * slice_gap),
                           # z_slice_center+(nslice * slice_gap),
                           # slice_gap)[::2]

    # # if corrpMap.corrp_data_filled exist
    # fa_data[fa_data == 0] = np.nan
    # fat_data[fat_data == 0] = np.nan

    # enigma_skeleton_data = np.where(
        # enigma_skeleton_data < 1,
        # np.nan,
        # enigma_skeleton_data)

    # # Make fig and axes
    # fig, axes = plt.subplots(ncols=ncols,
                             # nrows=nrows,
                             # figsize=(size_w * ncols,
                                      # size_h * nrows),
                             # dpi=200)

    # # For each axis
    # for num, ax in enumerate(np.ravel(axes)):
        # print(num)
        # enigma_fa_d = get_slice(enigma_fa_data, slice_nums, num)
        # enigma_skeleton_d = get_slice(enigma_skeleton_data, slice_nums, num)
        # fa_d = get_slice(fa_data, slice_nums, num)
        # fat_d = get_slice(fat_data, slice_nums, num)

        # fa_fat_overlap = np.where(
                # (fa_d > 0) * (fat_d > 0),
                # fa_d, np.nan)

        # # background FA map
        # img = ax.imshow(enigma_fa_d, cmap='gray')

        # # background skeleton
        # img = ax.imshow(enigma_skeleton_d, interpolation=None, cmap='ocean')

        # # FA data
        # fa_img = ax.imshow(
                # fa_d,
                # cmap='winter',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.7)

        # # FAt data
        # fat_img = ax.imshow(
                # fat_d,
                # cmap='autumn',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.6)

        # # overlap
        # overlap = ax.imshow(
                # fa_fat_overlap,
                # cmap='Reds',
                # interpolation=None,
                # vmin=0,
                # vmax=1)

        # ax.axis('off')
        # ax.annotate('z = {}'.format(slice_nums[num]),
                    # (0.01, 0.1),
                    # xycoords='axes fraction',
                    # color='white')

    # fig.subplots_adjust(hspace=0, wspace=0)

    # cbar_top = 0.03
    # cbar_height = 0.03
    # cbar_width = 0.15

    # # x, y, width, height
    # axbar = fig.add_axes([0.25, cbar_top, cbar_width, cbar_height])
    # cb_1 = fig.colorbar(fa_img, axbar, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_1.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # # cbytick_obj = plt.getp(cb_1.ax, 'yticklabels')
    # cb_1.outline.set_edgecolor('white')
    # # cb_1.ax.set_ylabel('FA reduction', fontweight='bold', color='white')
    # cb_1.ax.set_title('FA reduction', fontweight='bold', color='white')
    # cb_1.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj, color='white')

    # axbar_2 = fig.add_axes([0.45, cbar_top, cbar_width, cbar_height])
    # cb_2 = fig.colorbar(fat_img, axbar_2, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_2.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # # cbytick_obj_2 = plt.getp(cb_2.ax, 'yticklabels')
    # cb_2.outline.set_edgecolor('white')
    # # cb_2.ax.yaxis.set_tick_params(color='white')
    # # cb_2.ax.set_ylabel('FAt reduction', fontweight='bold', color='white')
    # cb_2.ax.set_title('FAt reduction', fontweight='bold', color='white')
    # cb_2.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj_2, color='white')

    # axbar_3 = fig.add_axes([0.65, cbar_top, cbar_width, cbar_height])
    # cb_3 = fig.colorbar(overlap, axbar_3, orientation='horizontal')
    # # cbytick_obj_3 = plt.getp(cb_3.ax, 'yticklabels')
    # cb_3.outline.set_edgecolor('white')
    # # cb_3.ax.yaxis.set_tick_params(color='white')
    # # cb_3.ax.set_ylabel(
        # # 'Reduction in both FA and FAt',
        # # fontweight='bold', color='white')
    # cb_3.ax.set_title(
        # 'Reduction in both FA and FAt',
        # fontweight='bold', color='white')
    # cb_3.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj_3, color='white')

    # plt.style.use('dark_background')

    # # fig = fig
    # fig.suptitle('Reduced FA and FAt in SLE compared to HCs', y=0.90, fontsize=20)
    # fig.savefig(outfile, dpi=200, bbox_inches='tight')
    # # axes = axes

# def lupus_get_figures_FW(fw_filled, outfile):
    # print('lupus_get_figures')

    # fw_data = nb.load(fw_filled).get_data()

    # enigma_dir = Path('/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
    # enigma_fa_loc = enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
    # enigma_fa_data = nb.load(str(enigma_fa_loc)).get_data()

    # enigma_skeleton_mask_loc = enigma_dir / \
        # 'ENIGMA_DTI_FA_skeleton_mask.nii.gz'

    # enigma_skeleton_data = nb.load(str(enigma_skeleton_mask_loc)).get_data()

    # ncols = 5
    # nrows = 4
    # size_w = 4
    # size_h = 4
    # slice_gap = 3

    # # Get the center of data
    # center_of_data = np.array(
        # ndimage.measurements.center_of_mass(
            # enigma_fa_data)).astype(int)

    # # Get the center slice number
    # z_slice_center = center_of_data[-1]

    # # Get the slice numbers in array
    # nslice = ncols * nrows
    # slice_nums = np.arange(z_slice_center-(nslice * slice_gap),
                           # z_slice_center+(nslice * slice_gap),
                           # slice_gap)[::2]

    # # if corrpMap.corrp_data_filled exist
    # fw_data[fw_data == 0] = np.nan

    # enigma_skeleton_data = np.where(
        # enigma_skeleton_data < 1,
        # np.nan,
        # enigma_skeleton_data)

    # # Make fig and axes
    # fig, axes = plt.subplots(ncols=ncols,
                             # nrows=nrows,
                             # figsize=(size_w * ncols,
                                      # size_h * nrows),
                             # dpi=200)

    # # For each axis
    # for num, ax in enumerate(np.ravel(axes)):
        # print(num)
        # enigma_fa_d = get_slice(enigma_fa_data, slice_nums, num)
        # enigma_skeleton_d = get_slice(enigma_skeleton_data, slice_nums, num)
        # fw_d = get_slice(fw_data, slice_nums, num)

        # # background FA map
        # img = ax.imshow(enigma_fa_d, cmap='gray')

        # # background skeleton
        # img = ax.imshow(enigma_skeleton_d, interpolation=None, cmap='ocean')

        # # overlap
        # fw_img = ax.imshow(
                # fw_d,
                # cmap='winter',
                # interpolation=None,
                # vmin=0,
                # vmax=1)

        # ax.axis('off')
        # ax.annotate('z = {}'.format(slice_nums[num]),
                    # (0.01, 0.1),
                    # xycoords='axes fraction',
                    # color='white')

    # fig.subplots_adjust(hspace=0, wspace=0)

    # cbar_top = 0.03
    # cbar_height = 0.03
    # cbar_width = 0.15

    # # x, y, width, height
    # axbar = fig.add_axes([0.4, cbar_top, cbar_width, cbar_height])
    # cb_1 = fig.colorbar(fw_img, axbar, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_1.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # cb_1.outline.set_edgecolor('white')
    # cb_1.ax.set_title('FW increase', fontweight='bold', color='white')
    # cb_1.ax.yaxis.set_label_position('left')

    # plt.style.use('dark_background')

    # fig.suptitle('Increased FW in SLE compared to HCs', y=0.90, fontsize=20)
    # fig.savefig(outfile, dpi=200, bbox_inches='tight')

# def lupus_get_figures_NPSLE_nonNPSLE(fa_filled, fat_filled, outfile):
    # print('lupus_get_figures')

    # fa_data = nb.load(fa_filled).get_data()
    # fat_data = nb.load(fat_filled).get_data()

    # enigma_dir = Path('/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
    # enigma_fa_loc = enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
    # enigma_fa_data = nb.load(str(enigma_fa_loc)).get_data()

    # enigma_skeleton_mask_loc = enigma_dir / \
        # 'ENIGMA_DTI_FA_skeleton_mask.nii.gz'

    # enigma_skeleton_data = nb.load(str(enigma_skeleton_mask_loc)).get_data()

    # ncols = 5
    # nrows = 4
    # size_w = 4
    # size_h = 4
    # slice_gap = 3

    # # Get the center of data
    # center_of_data = np.array(
        # ndimage.measurements.center_of_mass(
            # enigma_fa_data)).astype(int)

    # # Get the center slice number
    # z_slice_center = center_of_data[-1]

    # # Get the slice numbers in array
    # nslice = ncols * nrows
    # slice_nums = np.arange(z_slice_center-(nslice * slice_gap),
                           # z_slice_center+(nslice * slice_gap),
                           # slice_gap)[::2]

    # # if corrpMap.corrp_data_filled exist
    # fa_data[fa_data == 0] = np.nan
    # fat_data[fat_data == 0] = np.nan

    # enigma_skeleton_data = np.where(
        # enigma_skeleton_data < 1,
        # np.nan,
        # enigma_skeleton_data)

    # # Make fig and axes
    # fig, axes = plt.subplots(ncols=ncols,
                             # nrows=nrows,
                             # figsize=(size_w * ncols,
                                      # size_h * nrows),
                             # dpi=200)

    # # For each axis
    # for num, ax in enumerate(np.ravel(axes)):
        # print(num)
        # enigma_fa_d = get_slice(enigma_fa_data, slice_nums, num)
        # enigma_skeleton_d = get_slice(enigma_skeleton_data, slice_nums, num)
        # fa_d = get_slice(fa_data, slice_nums, num)
        # fat_d = get_slice(fat_data, slice_nums, num)

        # fa_fat_overlap = np.where(
                # (fa_d > 0) * (fat_d > 0),
                # fa_d, np.nan)

        # # background FA map
        # img = ax.imshow(enigma_fa_d, cmap='gray')

        # # background skeleton
        # img = ax.imshow(enigma_skeleton_d, interpolation=None, cmap='ocean')

        # # FAt data
        # fat_img = ax.imshow(
                # fat_d,
                # cmap='Blues_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.6)

        # # FA data
        # fa_img = ax.imshow(
                # fa_d,
                # cmap='autumn_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.7)

        # # overlap
        # overlap = ax.imshow(
                # fa_fat_overlap,
                # cmap='Purples_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.2)

        # ax.axis('off')
        # ax.annotate('z = {}'.format(slice_nums[num]),
                    # (0.01, 0.1),
                    # xycoords='axes fraction',
                    # color='white')

    # fig.subplots_adjust(hspace=0, wspace=0)

    # cbar_top = 0.03
    # cbar_height = 0.03
    # cbar_width = 0.15

    # # x, y, width, height
    # axbar = fig.add_axes([0.25, cbar_top, cbar_width, cbar_height])
    # cb_1 = fig.colorbar(fa_img, axbar, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_1.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # # cbytick_obj = plt.getp(cb_1.ax, 'yticklabels')
    # cb_1.outline.set_edgecolor('white')
    # # cb_1.ax.set_ylabel('FA reduction', fontweight='bold', color='white')
    # cb_1.ax.set_title('NPSLE vs HC\n(36.4% of all skeleton voxels)', fontweight='bold', color='white')
    # cb_1.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj, color='white')

    # axbar_2 = fig.add_axes([0.45, cbar_top, cbar_width, cbar_height])
    # cb_2 = fig.colorbar(fat_img, axbar_2, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_2.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # # cbytick_obj_2 = plt.getp(cb_2.ax, 'yticklabels')
    # cb_2.outline.set_edgecolor('white')
    # # cb_2.ax.yaxis.set_tick_params(color='white')
    # # cb_2.ax.set_ylabel('FAt reduction', fontweight='bold', color='white')
    # cb_2.ax.set_title('nonNPSLE vs HC\n(30.4% of all skeleton voxels)', fontweight='bold', color='white')
    # cb_2.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj_2, color='white')

    # axbar_3 = fig.add_axes([0.65, cbar_top, cbar_width, cbar_height])
    # cb_3 = fig.colorbar(overlap, axbar_3, orientation='horizontal')
    # # cbytick_obj_3 = plt.getp(cb_3.ax, 'yticklabels')
    # cb_3.outline.set_edgecolor('white')
    # # cb_3.ax.yaxis.set_tick_params(color='white')
    # # cb_3.ax.set_ylabel(
        # # 'Reduction in both FA and FAt',
        # # fontweight='bold', color='white')
    # cb_3.ax.set_title(
        # 'reduced in both NPSLE and nonNPSLE',
        # fontweight='bold', color='white')
    # cb_3.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj_3, color='white')

    # plt.style.use('dark_background')

    # # fig = fig
    # fig.suptitle('FA reduction in NPSLE and nonNPSLE, each group compared to HCs', y=0.90, fontsize=20)
    # fig.savefig(outfile, dpi=200, bbox_inches='tight')
    # # axes = axes

# def lupus_get_figures_NPSLE_nonNPSLE_fat(fa_filled, fat_filled, outfile):
    # print('lupus_get_figures')

    # fa_data = nb.load(fa_filled).get_data()
    # fat_data = nb.load(fat_filled).get_data()

    # enigma_dir = Path('/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
    # enigma_fa_loc = enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
    # enigma_fa_data = nb.load(str(enigma_fa_loc)).get_data()

    # enigma_skeleton_mask_loc = enigma_dir / \
        # 'ENIGMA_DTI_FA_skeleton_mask.nii.gz'

    # enigma_skeleton_data = nb.load(str(enigma_skeleton_mask_loc)).get_data()

    # ncols = 5
    # nrows = 4
    # size_w = 4
    # size_h = 4
    # slice_gap = 3

    # # Get the center of data
    # center_of_data = np.array(
        # ndimage.measurements.center_of_mass(
            # enigma_fa_data)).astype(int)

    # # Get the center slice number
    # z_slice_center = center_of_data[-1]

    # # Get the slice numbers in array
    # nslice = ncols * nrows
    # slice_nums = np.arange(z_slice_center-(nslice * slice_gap),
                           # z_slice_center+(nslice * slice_gap),
                           # slice_gap)[::2]

    # # if corrpMap.corrp_data_filled exist
    # fa_data[fa_data == 0] = np.nan
    # fat_data[fat_data == 0] = np.nan

    # enigma_skeleton_data = np.where(
        # enigma_skeleton_data < 1,
        # np.nan,
        # enigma_skeleton_data)

    # # Make fig and axes
    # fig, axes = plt.subplots(ncols=ncols,
                             # nrows=nrows,
                             # figsize=(size_w * ncols,
                                      # size_h * nrows),
                             # dpi=200)

    # # For each axis
    # for num, ax in enumerate(np.ravel(axes)):
        # print(num)
        # enigma_fa_d = get_slice(enigma_fa_data, slice_nums, num)
        # enigma_skeleton_d = get_slice(enigma_skeleton_data, slice_nums, num)
        # fa_d = get_slice(fa_data, slice_nums, num)
        # fat_d = get_slice(fat_data, slice_nums, num)

        # fa_fat_overlap = np.where(
                # (fa_d > 0) * (fat_d > 0),
                # fa_d, np.nan)

        # # background FA map
        # img = ax.imshow(enigma_fa_d, cmap='gray')

        # # background skeleton
        # img = ax.imshow(enigma_skeleton_d, interpolation=None, cmap='ocean')

        # # FAt data
        # fat_img = ax.imshow(
                # fat_d,
                # cmap='Blues_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.6)

        # # FA data
        # fa_img = ax.imshow(
                # fa_d,
                # cmap='autumn_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.7)

        # # overlap
        # overlap = ax.imshow(
                # fa_fat_overlap,
                # cmap='Purples_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.2)

        # ax.axis('off')
        # ax.annotate('z = {}'.format(slice_nums[num]),
                    # (0.01, 0.1),
                    # xycoords='axes fraction',
                    # color='white')

    # fig.subplots_adjust(hspace=0, wspace=0)

    # cbar_top = 0.03
    # cbar_height = 0.03
    # cbar_width = 0.15

    # # x, y, width, height
    # axbar = fig.add_axes([0.25, cbar_top, cbar_width, cbar_height])
    # cb_1 = fig.colorbar(fa_img, axbar, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_1.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # # cbytick_obj = plt.getp(cb_1.ax, 'yticklabels')
    # cb_1.outline.set_edgecolor('white')
    # # cb_1.ax.set_ylabel('FA reduction', fontweight='bold', color='white')
    # cb_1.ax.set_title('NPSLE vs HC\n(29.7% of all skeleton voxels)', fontweight='bold', color='white')
    # cb_1.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj, color='white')

    # axbar_2 = fig.add_axes([0.45, cbar_top, cbar_width, cbar_height])
    # cb_2 = fig.colorbar(fat_img, axbar_2, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_2.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # # cbytick_obj_2 = plt.getp(cb_2.ax, 'yticklabels')
    # cb_2.outline.set_edgecolor('white')
    # # cb_2.ax.yaxis.set_tick_params(color='white')
    # # cb_2.ax.set_ylabel('FAt reduction', fontweight='bold', color='white')
    # cb_2.ax.set_title('nonNPSLE vs HC\n(15.3% of all skeleton voxels)', fontweight='bold', color='white')
    # cb_2.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj_2, color='white')

    # axbar_3 = fig.add_axes([0.65, cbar_top, cbar_width, cbar_height])
    # cb_3 = fig.colorbar(overlap, axbar_3, orientation='horizontal')
    # # cbytick_obj_3 = plt.getp(cb_3.ax, 'yticklabels')
    # cb_3.outline.set_edgecolor('white')
    # # cb_3.ax.yaxis.set_tick_params(color='white')
    # # cb_3.ax.set_ylabel(
        # # 'Reduction in both FA and FAt',
        # # fontweight='bold', color='white')
    # cb_3.ax.set_title(
        # 'reduced in both NPSLE and nonNPSLE',
        # fontweight='bold', color='white')
    # cb_3.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj_3, color='white')

    # plt.style.use('dark_background')

    # # fig = fig
    # fig.suptitle('FAt reduction in NPSLE and nonNPSLE, each group compared to HCs', y=0.90, fontsize=20)
    # fig.savefig(outfile, dpi=200, bbox_inches='tight')


# def lupus_get_figures_NPSLE_nonNPSLE_fw(fa_filled, fat_filled, outfile):
    # print('lupus_get_figures')

    # fa_data = nb.load(fa_filled).get_data()
    # fat_data = nb.load(fat_filled).get_data()

    # enigma_dir = Path('/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
    # enigma_fa_loc = enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
    # enigma_fa_data = nb.load(str(enigma_fa_loc)).get_data()

    # enigma_skeleton_mask_loc = enigma_dir / \
        # 'ENIGMA_DTI_FA_skeleton_mask.nii.gz'

    # enigma_skeleton_data = nb.load(str(enigma_skeleton_mask_loc)).get_data()

    # ncols = 5
    # nrows = 4
    # size_w = 4
    # size_h = 4
    # slice_gap = 3

    # # Get the center of data
    # center_of_data = np.array(
        # ndimage.measurements.center_of_mass(
            # enigma_fa_data)).astype(int)

    # # Get the center slice number
    # z_slice_center = center_of_data[-1]

    # # Get the slice numbers in array
    # nslice = ncols * nrows
    # slice_nums = np.arange(z_slice_center-(nslice * slice_gap),
                           # z_slice_center+(nslice * slice_gap),
                           # slice_gap)[::2]

    # # if corrpMap.corrp_data_filled exist
    # fa_data[fa_data == 0] = np.nan
    # fat_data[fat_data == 0] = np.nan

    # enigma_skeleton_data = np.where(
        # enigma_skeleton_data < 1,
        # np.nan,
        # enigma_skeleton_data)

    # # Make fig and axes
    # fig, axes = plt.subplots(ncols=ncols,
                             # nrows=nrows,
                             # figsize=(size_w * ncols,
                                      # size_h * nrows),
                             # dpi=200)

    # # For each axis
    # for num, ax in enumerate(np.ravel(axes)):
        # print(num)
        # enigma_fa_d = get_slice(enigma_fa_data, slice_nums, num)
        # enigma_skeleton_d = get_slice(enigma_skeleton_data, slice_nums, num)
        # fa_d = get_slice(fa_data, slice_nums, num)
        # fat_d = get_slice(fat_data, slice_nums, num)

        # fa_fat_overlap = np.where(
                # (fa_d > 0) * (fat_d > 0),
                # fa_d, np.nan)

        # # background FA map
        # img = ax.imshow(enigma_fa_d, cmap='gray')

        # # background skeleton
        # img = ax.imshow(enigma_skeleton_d, interpolation=None, cmap='ocean')

        # # FAt data
        # fat_img = ax.imshow(
                # fat_d,
                # cmap='Blues_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.6)

        # # FA data
        # fa_img = ax.imshow(
                # fa_d,
                # cmap='autumn_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.7)

        # # overlap
        # overlap = ax.imshow(
                # fa_fat_overlap,
                # cmap='Purples_r',
                # interpolation=None,
                # vmin=0,
                # vmax=1)#, alpha=0.2)

        # ax.axis('off')
        # ax.annotate('z = {}'.format(slice_nums[num]),
                    # (0.01, 0.1),
                    # xycoords='axes fraction',
                    # color='white')

    # fig.subplots_adjust(hspace=0, wspace=0)

    # cbar_top = 0.03
    # cbar_height = 0.03
    # cbar_width = 0.15

    # # x, y, width, height
    # axbar = fig.add_axes([0.25, cbar_top, cbar_width, cbar_height])
    # cb_1 = fig.colorbar(fa_img, axbar, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_1.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # # cbytick_obj = plt.getp(cb_1.ax, 'yticklabels')
    # cb_1.outline.set_edgecolor('white')
    # # cb_1.ax.set_ylabel('FA reduction', fontweight='bold', color='white')
    # cb_1.ax.set_title('NPSLE vs HC\n(5.7% of all skeleton voxels)', fontweight='bold', color='white')
    # cb_1.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj, color='white')

    # axbar_2 = fig.add_axes([0.45, cbar_top, cbar_width, cbar_height])
    # cb_2 = fig.colorbar(fat_img, axbar_2, orientation='horizontal',
                        # ticks=[0.05, 0.95])
    # cb_2.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
    # # cbytick_obj_2 = plt.getp(cb_2.ax, 'yticklabels')
    # cb_2.outline.set_edgecolor('white')
    # # cb_2.ax.yaxis.set_tick_params(color='white')
    # # cb_2.ax.set_ylabel('FAt reduction', fontweight='bold', color='white')
    # cb_2.ax.set_title('nonNPSLE vs HC\n(4.4% of all skeleton voxels)', fontweight='bold', color='white')
    # cb_2.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj_2, color='white')

    # axbar_3 = fig.add_axes([0.65, cbar_top, cbar_width, cbar_height])
    # cb_3 = fig.colorbar(overlap, axbar_3, orientation='horizontal')
    # # cbytick_obj_3 = plt.getp(cb_3.ax, 'yticklabels')
    # cb_3.outline.set_edgecolor('white')
    # # cb_3.ax.yaxis.set_tick_params(color='white')
    # # cb_3.ax.set_ylabel(
        # # 'Reduction in both FA and FAt',
        # # fontweight='bold', color='white')
    # cb_3.ax.set_title(
        # 'increased in both NPSLE and nonNPSLE',
        # fontweight='bold', color='white')
    # cb_3.ax.yaxis.set_label_position('left')
    # # plt.setp(cbytick_obj_3, color='white')

    # plt.style.use('dark_background')

    # # fig = fig
    # fig.suptitle('FW increase in NPSLE and nonNPSLE, each group compared to HCs', y=0.90, fontsize=20)
    # fig.savefig(outfile, dpi=200, bbox_inches='tight')

def prac():
    fa = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats/HC_vs_SLE/tbss_FA_tfce_corrp_tstat1_filled.nii.gz'
    fat = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats/HC_vs_SLE/tbss_FAt_tfce_corrp_tstat1_filled.nii.gz'
    fw = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats/HC_vs_SLE/tbss_FW_tfce_corrp_tstat2_filled.nii.gz'

    tbssFigure = TbssFigure([fa, fat], 'prac.png')
    print(tbssFigure.image_files)
    tbssFigure.images_mask_out_the_zero()
    tbssFigure.loop_through_axes_draw_bg()
    tbssFigure.annotate_with_z()
    tbssFigure.get_overlap_between_maps()
    tbssFigure.cmap_list = ['autumn', 'cool', 'Reds']
    tbssFigure.loop_through_axes_draw_images()

    tbssFigure.cbar_titles = ['ha', 'ho', 'overlap']
    tbssFigure.get_cbar_horizontal_info()
    tbssFigure.add_cbars_horizontal()

    tbssFigure.fig.savefig('prac.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    prac()

    # SLE vs HC
    # lupus_get_figures_FW(fw, 'FW_out.png')
    # lupus_get_figures(fa, fat, 'FA_FAt_out.png')

    # NPSLE & nonNPSLE
    # fa_NPSLE = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats_HC_NPSLE/tbss_FA_tfce_corrp_tstat1_filled.nii.gz'
    # fa_nonNPSLE = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats_HC_nonNPSLE/tbss_FA_tfce_corrp_tstat1_filled.nii.gz'
    # lupus_get_figures_NPSLE_nonNPSLE(
            # fa_NPSLE,
            # fa_nonNPSLE,
            # '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats/HC_vs_SLE/NPSLE_nonNPSLE_FA.png')


    # fat_NPSLE = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats_HC_NPSLE/tbss_FAt_tfce_corrp_tstat1_filled.nii.gz'
    # fat_nonNPSLE = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats_HC_nonNPSLE/tbss_FAt_tfce_corrp_tstat1_filled.nii.gz'
    # lupus_get_figures_NPSLE_nonNPSLE_fat(
            # fat_NPSLE,
            # fat_nonNPSLE,
            # '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats/HC_vs_SLE/NPSLE_nonNPSLE_FAt.png')

    # fw_NPSLE = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats_HC_NPSLE/tbss_FW_tfce_corrp_tstat2_filled.nii.gz'
    # fw_nonNPSLE = '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats_HC_nonNPSLE/tbss_FW_tfce_corrp_tstat2_filled.nii.gz'
    # lupus_get_figures_NPSLE_nonNPSLE_fw(
            # fw_NPSLE,
            # fw_nonNPSLE,
            # '/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats/HC_vs_SLE/NPSLE_nonNPSLE_FW.png')
