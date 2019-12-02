#!/data/pnl/kcho/anaconda3/bin/python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nb
from scipy import ndimage
import functools


class Enigma:
    # @functools.cached_property
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
            # x, y, width, height
            cbar_title = self.cbar_titles[num]
            axbar = self.fig.add_axes([
                self.cbar_x,
                self.cbar_y,
                self.cbar_width,
                self.cbar_height])

            if cbar_title.lower() != 'overlap':
                cb = self.fig.colorbar(
                        self.imshow_list[num],
                        axbar,
                        orientation='horizontal',
                        ticks=[0, 1])

                cb.ax.set_xticklabels(['P = 0.05', 'P < 0.01'], color='white')
            else:
                cb = self.fig.colorbar(
                        self.imshow_list[num],
                        axbar,
                        orientation='horizontal',
                        boundaries=[0.999, 1],
                        ticks=[])

            cb.outline.set_edgecolor('white')
            cb.ax.set_title(
                    cbar_title,
                    fontsize=15, fontweight='bold', color='white')
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
        # slices below z=40 have been zeroed in order to move the
        # center_of_data upwards for better visualization
        data_zeroed_below = data.copy()
        data_zeroed_below[:, :, :40] = 0
        self.center_of_data = np.array(
            ndimage.measurements.center_of_mass(data_zeroed_below)).astype(int)

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
                1-self.image_data_list[0],
                np.nan)
        # self.overlap_map = np.where(
                # (self.image_data_list[0] > 0) * (self.image_data_list[1] > 0),
                # 1,
                # np.nan)

        self.image_data_list.append(self.overlap_map)



class TbssFigure(Enigma, FigureSettings, FigureNifti):
    def __init__(self, **kwargs):
        Enigma.__init__(self)
        FigureSettings.__init__(self)

        self.get_cbar_horizontal_info()
        print('Given parameters')
        for key, value in kwargs.items():
            print(f"\t{key} : {value}")
            setattr(self, key, value)

        self.get_center(self.enigma_fa_data)
        self.enigma_skeleton_data = self.transparent_mask(
            self.enigma_skeleton_data)

        self.fig, self.axes = plt.subplots(
            ncols=self.ncols,
            nrows=self.nrows,
            figsize=(self.size_w * self.ncols,
                     self.size_h * self.nrows),
            dpi=self.dpi)

        self.fig.subplots_adjust(hspace=0, wspace=0)

        # self.image_files = image_files
        self.image_data_list = [nb.load(x).get_data() for x
                                in self.image_files]
        plt.style.use('dark_background')


    def create_figure_one_map(self):
        self.images_mask_out_the_zero()
        self.loop_through_axes_draw_bg()
        self.annotate_with_z()

        # self.cmap_list = ['Blues_r', 'autumn_r', 'Purples_r']
        self.loop_through_axes_draw_images()
        # self.get_overlap_between_maps()
        # self.loop_through_axes_draw_overlap()

        self.add_cbars_horizontal()

        self.fig.suptitle(self.title, y=0.9, fontsize=25)

        # self.fig.subplots_adjust(top=1.2)
        self.fig.savefig(self.output_file, dpi=200)#, bbox_inches='tight')
        # self.fig.savefig(self.output_file, dpi=200, bbox_inches='tight')

    def create_figure_two_maps_and_overlap(self):
        self.images_mask_out_the_zero()
        self.loop_through_axes_draw_bg()
        self.annotate_with_z()

        # self.cmap_list = ['Blues_r', 'autumn_r', 'Purples_r']
        self.loop_through_axes_draw_images()
        self.get_overlap_between_maps()
        self.loop_through_axes_draw_overlap()

        # self.cbar_titles = ['FA', 'FAt', 'overlap']
        self.get_cbar_horizontal_info()
        self.add_cbars_horizontal()

        self.fig.suptitle(self.title, y=0.9, fontsize=25)

        # self.fig.subplots_adjust(top=1.2)
        self.fig.savefig(self.output_file, dpi=200)#, bbox_inches='tight')
        # self.fig.savefig(self.output_file, dpi=200, bbox_inches='tight')

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
            for img_num, image in enumerate(self.image_data_list):
                alpha = self.alpha_list[img_num]
                image_d = self.get_slice(image, z_num)
                img = ax.imshow(
                        image_d,
                        cmap=self.cmap_list[img_num],
                        vmin=0,
                        vmax=1, alpha=alpha)
                self.imshow_list.append(img)

    def loop_through_axes_draw_overlap(self):
        image = self.image_data_list[-1]
        alpha = self.alpha_list[-1]
        for num, ax in enumerate(np.ravel(self.axes)):
            z_num = self.slice_nums[num]

            # background FA map
            image_d = self.get_slice(image, z_num)
            img = ax.imshow(
                    image_d,
                    cmap=self.overlap_cmap,
                    vmin=0,
                    vmax=1, alpha=alpha)
            self.imshow_list.append(img)


def prac():
    data_dir = Path('/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/stats/HC_vs_SLE')
    fa = str(data_dir / 'tbss_FA_tfce_corrp_tstat1_filled.nii.gz')
    fat = str(data_dir / 'tbss_FAt_tfce_corrp_tstat1_filled.nii.gz')
    fw = str(data_dir / 'tbss_FW_tfce_corrp_tstat2_filled.nii.gz')

    fa_color_1 = 'Blues_r'
    fa_color_2 = 'autumn'
    fa_color_overlap = 'summer'

    # # SLE
    # tbssFigure = TbssFigure(
        # image_files=[fa, fat],
        # output_file='SLE_FA_FAt.png',
        # cmap_list=[fa_color_1, fa_color_2],
        # overlap_cmap=fa_color_overlap,
        # cbar_titles=[
            # 'Reduced FA (34.5%)',
            # 'Reduced FAt (26.5%)',
            # 'Overlap'],
        # alpha_list=[1, 1, 0.8],
        # title='Significant changes in FA and FAt in HC vs SLE')
    # tbssFigure.create_figure_two_maps_and_overlap()

    # fw_color = 'Blues_r'
    # fat_color = 'autumn'

    # # SLE FAt vs FW
    # tbssFigure = TbssFigure(
        # image_files=[fat, fw],
        # output_file='SLE_FAt_FW.png',
        # cmap_list=[fat_color, fw_color],
        # overlap_cmap=fa_color_overlap,
        # cbar_titles=['Reduced FAt', 'Increased FW', 'Overlap'],
        # alpha_list=[1, 1, 0.8],
        # title='Significant changes in FAt and FW in HC vs SLE')
    # tbssFigure.create_figure_two_maps_and_overlap()

    # fw_color = 'Blues_r'
    # # SLE FW only
    # tbssFigure = TbssFigure(
            # image_files=[fw],
            # output_file='SLE_FW.png',
            # cmap_list=[fw_color],
            # cbar_titles=['Increased FW'],
            # alpha_list=[0.8],
            # title='Significant changes in FW in HC vs SLE',
            # cbar_x=0.35, cbar_width=0.3)
    # tbssFigure.create_figure_one_map()

    # data_dir = Path('/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/')
    # fa_npsle = str(data_dir / 'stats_HC_NPSLE/tbss_FA_tfce_corrp_tstat1_filled.nii.gz')
    # fa_nonnpsle = str(data_dir / 'stats_HC_nonNPSLE/tbss_FA_tfce_corrp_tstat1_filled.nii.gz')

    # npsle_color = 'Reds_r'
    # non_npsle_color = 'PuBu_r'
    # overlap_color = 'summer'

    # tbssFigure = TbssFigure(
        # image_files=[fa_npsle, fa_nonnpsle],
        # output_file='NPSLE_nonNPSLE_FA.png',
        # cmap_list=[npsle_color, non_npsle_color],
        # overlap_cmap=overlap_color,
        # cbar_titles=[
            # 'NPSLE : Reduced FA',
            # 'non-NPSLE : Reduced FA',
            # 'Overlap'],
        # alpha_list=[1, 1, 0.8],
        # title='Significant changes in FA in NPSLE and nonNPSLE, '
              # 'each compared to HCs')
    # tbssFigure.create_figure_two_maps_and_overlap()


    # data_dir = Path('/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/')
    # fat_npsle = str(data_dir / 'stats_HC_NPSLE/tbss_FAt_tfce_corrp_tstat1_filled.nii.gz')
    # fat_nonnpsle = str(data_dir / 'stats_HC_nonNPSLE/tbss_FAt_tfce_corrp_tstat1_filled.nii.gz')

    # npsle_color = 'Reds_r'
    # non_npsle_color = 'PuBu_r'
    # overlap_color = 'summer'

    # tbssFigure = TbssFigure(
        # image_files=[fat_npsle, fat_nonnpsle],
        # output_file='NPSLE_nonNPSLE_FAt.png',
        # cmap_list=[npsle_color, non_npsle_color],
        # overlap_cmap=overlap_color,
        # cbar_titles=[
            # 'NPSLE : Reduced FAt',
            # 'non-NPSLE : Reduced FAt',
            # 'Overlap'],
        # alpha_list=[1, 1, 0.8],
        # title='Significant changes in FAt in NPSLE and nonNPSLE, '
              # 'each compared to HCs')
    # tbssFigure.create_figure_two_maps_and_overlap()

    data_dir = Path('/data/pnl/kcho/Lupus/TBSS/Tashrif/tbss/')
    fw_npsle = str(data_dir / 'stats_HC_NPSLE/tbss_FW_tfce_corrp_tstat2_filled.nii.gz')
    fw_nonnpsle = str(data_dir / 'stats_HC_nonNPSLE/tbss_FW_tfce_corrp_tstat2_filled.nii.gz')

    npsle_color = 'Blues_r'
    non_npsle_color = 'Purples_r'
    overlap_color = 'Greys_r'

    tbssFigure = TbssFigure(
        image_files=[fw_npsle, fw_nonnpsle],
        output_file='NPSLE_nonNPSLE_FW.png',
        cmap_list=[npsle_color, non_npsle_color],
        overlap_cmap=overlap_color,
        cbar_titles=[
            'Increased FW in NPSLE',
            'Increased FW in non-NPSLE',
            'Overlap'],
        alpha_list=[1, 1, 0.8],
        title='Significant changes in FW in NPSLE and nonNPSLE, '
              'each compared to HCs')
    tbssFigure.create_figure_two_maps_and_overlap()

if __name__ == '__main__':
    prac()
