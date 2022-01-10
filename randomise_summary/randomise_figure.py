from pnl_randomise_utils import re, os
from pnl_randomise_utils import get_nifti_data, print_head

from typing import Union, List, Tuple

# figures
import sys
sys.path.append('/data/pnl/kcho/PNLBWH/devel/nifti-snapshot')
import nifti_snapshot.nifti_snapshot as nifti_snapshot

import matplotlib.pyplot as plt


class CorrpMapFigure(object):
    '''Figure related attributes of CorrpMap object'''

    def get_figure(self, **kwargs):
        """Get corrpMap figure"""
        self.cbar_title = f'{self.modality} {self.contrast_text}'

        # same slice
        if 'figure_same_slice' in kwargs:
            same_slice = kwargs.get('figure_same_slice')

        if hasattr(self, 'tbss_fill_out'): # for tbss fill option
            self.out_image_loc = re.sub(
                '.nii.gz', '.png', str(self.tbss_fill_out))
            self.title = f'{self.modality} {self.contrast_text}\n' \
                         f'{self.tbss_fill_out}'


            # vmin and vmax list given
            self.tbssFigure = nifti_snapshot.TbssFigure(
                image_files=[self.tbss_fill_out],
                fa_bg=self.fa_bg_loc,
                skeleton_bg=self.skel_mask_loc,
                output_file=self.out_image_loc,
                cmap_list=['autumn'],
                cbar_titles=[self.cbar_title],
                alpha_list=[1],
                title=self.title,
                tbss_filled=True,
                same_slice=same_slice)

            # below is self.tbssFigure.create_figure_one_map()
            # self.tbssFigure.images_mask_out_the_zero()

            # self.tbssFigure.loop_through_axes_draw_bg()
            self.tbssFigure.loop_through_axes_draw_bg_tbss()
            self.tbssFigure.annotate_with_z()
            self.tbssFigure.loop_through_axes_draw_images()
            self.tbssFigure.cbar_x = 0.25
            self.tbssFigure.cbar_width = 0.5

            
            self.tbssFigure.add_cbars_horizontal_tbss_filled()
            self.tbssFigure.fig.suptitle(
                self.tbssFigure.title, y=0.92, fontsize=25)
            self.tbssFigure.fig.savefig(self.tbssFigure.output_file, dpi=200)

        else:
            self.out_image_loc = re.sub('.nii.gz', '.png', str(self.location))
            self.title = f'{self.modality} {self.contrast_text}\n' \
                         f'{self.location}'
            self.tbssFigure = nifti_snapshot.TbssFigure(
                image_files=[str(self.location)],
                fa_bg=self.fa_bg_loc,
                skeleton_bg=self.skel_mask_loc,
                output_file=self.out_image_loc,
                cmap_list=['autumn'],
                cbar_titles=[self.cbar_title],
                alpha_list=[1],
                cbar_ticks=[0.95, 1],
                title=self.title,
                same_slice=same_slice)


            # below is self.tbssFigure.create_figure_one_map()
            self.tbssFigure.images_mask_out_the_zero()
            self.tbssFigure.images_mask_by_threshold(0.95)

            # self.tbssFigure.loop_through_axes_draw_bg()
            self.tbssFigure.loop_through_axes_draw_bg_tbss()
            self.tbssFigure.annotate_with_z()
            self.tbssFigure.loop_through_axes_draw_images_corrp_map(0.95)
            self.tbssFigure.cbar_x = 0.25
            self.tbssFigure.cbar_width = 0.5
            self.tbssFigure.add_cbars_horizontal()

            self.tbssFigure.fig.suptitle(
                self.tbssFigure.title, y=0.92, fontsize=25)
            self.tbssFigure.fig.savefig(self.tbssFigure.output_file, dpi=200)


    def get_figure_enigma(self, **kwargs):
        # TODO replace this function with nifti_snapshot
        # TODO add skeleton check functions to the randomise_summary
        """Fig and axes attribute to CorrpMap"""

        # if study template is not ENIGMA
        if 'mean_fa' in kwargs:
            mean_fa_loc = kwargs.get('mean_fa')
            print(f'background image : {mean_fa_loc}')
            self.enigma_fa_data = get_nifti_data(mean_fa_loc)

            mean_fa_skel_loc = re.sub('.nii.gz', '_skeleton.nii.gz',
                                      mean_fa_loc)
            print(f'background skeleton image: {mean_fa_skel_loc}')
            self.enigma_skeleton_data = get_nifti_data(mean_fa_skel_loc)
        else:
            # self.enigma_fa_data = get_nifti_data(self.enigma_fa_loc)
            self.enigma_fa_data = get_nifti_data(self.fa_bg_loc)
            self.enigma_skeleton_data = get_nifti_data(
                    self.skel_mask_loc)

        # figure settings
        self.ncols = 5
        self.nrows = 4
        size_w = 4
        size_h = 4

        # When study template is used, slice_gap=3 is too wide)
        if self.data_shape[-1] < 100:
            slice_gap = 2
        else:
            slice_gap = 3

        # Get the center of data
        center_of_data = np.array(
            ndimage.measurements.center_of_mass(
                self.enigma_fa_data)).astype(int)
        # Get the center slice number
        z_slice_center = center_of_data[-1]

        # Get the slice numbers in array
        nslice = self.ncols * self.nrows
        slice_nums = np.arange(z_slice_center-(nslice * slice_gap),
                               z_slice_center+(nslice * slice_gap),
                               slice_gap)[::2]

        # if corrpMap.corrp_data_filled exist
        if hasattr(self, 'corrp_data_filled'):
            data = np.where(self.corrp_data_filled == 0,
                            np.nan,
                            self.corrp_data_filled)

        elif hasattr(self, 'type'):
            if self.type in ['average', 'std', 'bin_sum', 'bin_sum_diff']:
                # for skeleton std data plot
                data = np.where(self.corrp_data == 0,
                                np.nan,
                                self.corrp_data)
        else:
            # Make voxels with their intensities lower than data_vmin
            # transparent
            data = np.where(self.corrp_data < self.threshold,
                            np.nan,
                            self.corrp_data)

        # TODO put below to above
        if hasattr(self, 'vmin'):
            vmin = self.vmin
        else:
            vmin = self.threshold

        if hasattr(self, 'vmax'):
            if self.vmax == 'free':
                vmax = self.corrp_data.max()
            else:
                vmax = self.vmax
        else:
            vmax = 1

        self.tbssFigure = nifti_snapshot.TbssFigure(
            template=self.template,
            image_data_list=[data],
            output_file=self.out_image_loc,
            cmap_list=['autumn'],
            cbar_titles=[self.cbar_title],
            alpha_list=[1],
            title=self.title)

        # below is self.tbssFigure.create_figure_one_map()
        self.tbssFigure.images_mask_out_the_zero()
        self.tbssFigure.images_mask_by_threshold(0.95)
        # self.tbssFigure.loop_through_axes_draw_bg()
        self.tbssFigure.loop_through_axes_draw_bg_tbss()
        self.tbssFigure.annotate_with_z()
        self.tbssFigure.loop_through_axes_draw_images_corrp_map(0.95)
        self.tbssFigure.cbar_x = 0.25
        self.tbssFigure.cbar_width = 0.5
        self.tbssFigure.add_cbars_horizontal()

        # self.fig = self.tbssFigure.fig
        self.tbssFigure.fig.suptitle(
            self.tbssFigure.title, y=0.92, fontsize=25)
        self.tbssFigure.fig.savefig(self.tbssFigure.output_file, dpi=200)

    def tbss_fill(self):
        if self.template == 'enigma':
            command = f'tbss_fill  \
                    {self.location} \
                    {self.threshold} \
                    {self.mean_fa} {self.tbss_fill_out}'
                    # {self.enigma_fa_loc} {self.tbss_fill_out}'
        else:
            command = f'tbss_fill  \
                    {self.location} \
                    {self.threshold} \
                    {self.fa_bg_loc} {self.tbss_fill_out}'

        print(re.sub('\s+', ' ', command))
        print(os.popen(command).read())


def create_figure(args: object, corrp_map_classes: List[object]) -> None:
    if args.figure or args.tbss_fill:
        print_head('Saving figures')
        for corrpMap in corrp_map_classes:
            if corrpMap.significant is True:
                # tbss_fill if tbss_fill=True
                if args.tbss_fill:
                    print_head(f'Estimating tbss_fill for {corrpMap.location}')
                    # run tbss_fill
                    corrpMap.tbss_fill_out = re.sub(
                        '.nii.gz', '_filled.nii.gz',
                        str(corrpMap.location))
                    corrpMap.tbss_fill()
                    corrpMap.get_figure(figure_same_slice=args.figure_same_slice)
                    plt.close()

                if args.figure:
                    corrpMap.get_figure(figure_same_slice=args.figure_same_slice)
                    plt.close()
