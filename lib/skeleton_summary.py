#!/data/pnl/kcho/anaconda3/bin/python

# table and array
import numpy as np
import pandas as pd

# os tools
import re
from pathlib import Path

# import print option
# from kchopy.kcho_utils import print_df

# figure
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# stats
from itertools import combinations
from stats import anova, ttest

from pnl_randomise_utils import get_nifti_data, get_nifti_img_data

def get_average_for_each_volume(data, mask):
    """Get average of values in the mask for each volume in 4d matrix"""
    cluster_averages = {}
    # Get average of values in the `significant_cluster_data` map
    # for each skeleton volume
    for vol_num in np.arange(data.shape[3]):
        vol_data = data[:, :, :, vol_num]
        average = vol_data[mask == 1].mean()
        cluster_averages[vol_num] = average
    return cluster_averages


class MergedSkeleton:
    """TBSS all_modality_skeleton map object"""
    def __init__(self, merged_skeleton_loc, mask_loc):
        """initialize mergedSkeleton object"""
        self.merged_skeleton_loc = merged_skeleton_loc
        self.skel_mask_loc = mask_loc

        # load merged skeleton nifti
        print(f"Reading {merged_skeleton_loc}")
        self.merged_skeleton_img, self.merged_skeleton_data = \
                get_nifti_img_data(merged_skeleton_loc)
        print(f"Completed reading {merged_skeleton_loc}")

        # load mask as boolean array
        self.mask_data = get_nifti_data(mask_loc) == 1

        # binarize merged skeleton map
        print(f"Estimating sum of binarized skeleton maps for all subject")
        self.merged_skeleton_data_bin_sum = np.sum(
            np.where(self.merged_skeleton_data == 0, 0, 1),
            axis=3)

        print(f"Estimating mean of binarized skeleton maps for all subject")
        self.merged_skeleton_data_bin_mean = np.mean(
            np.where(self.merged_skeleton_data == 0, 0, 1),
            axis=3)

    def update_with_corrpMap(self, corrpMap):
        """Add modality, cluster_averages_df, df"""
        self.mask_data = ''
        self.merged_skeleton_data_bin_sum = ''
        self.merged_skeleton_data_bin_mean = ''

        # significant cluster mask
        self.sig_mask = np.where(
            corrpMap.corrp_data >= corrpMap.threshold, 1, 0)

        self.cluster_averages = get_average_for_each_volume(
            self.merged_skeleton_data, self.sig_mask)

        # get a map with significant voxels
        self.modality = corrpMap.modality

        self.cluster_averages_df = pd.DataFrame.from_dict(
            self.cluster_averages,
            orient='index',
            columns=[f'{corrpMap.modality} values in the significant '
                     f'cluster {corrpMap.name}']
        )

        # get a list of groups for each  volume
        group_list = corrpMap.matrix_df[corrpMap.group_cols].astype(
            'int').to_string(header=False, index=False).split('\n')

        # data-frame for each subject
        self.df = pd.DataFrame({
            'subject': corrpMap.matrix_df.index,
            'mean': list(self.cluster_averages.values()),
            'group': group_list
            })

    def skeleton_level_summary(self):
        """Summarize all skeleton

        Information of the 4d merged skeleton files.

        Attributes:
            merged_skeleton_mean_map: 3d arr, mean of the skeleton maps
            merged_skeleton_std_map: 3d arr, std of the skeleton maps
            merged_skeleton_mean:
                float, nonzero mean of merged_skeleton_mean_map
            merged_skeleton_std:
                float, nonzero std of merged_skeleton_mean_map
        """
        # skeleton mean and std maps
        print(f"Estimating average of all skeleton maps")
        self.merged_skeleton_mean_map = np.mean(
            self.merged_skeleton_data, axis=3)
        print(f"Estimating standard deviation of all skeleton maps")
        self.merged_skeleton_std_map = np.std(
                self.merged_skeleton_data, axis=3)

        # skeleton mean and std values
        print(f"Get a mean value skeleton for all subject = single value")
        self.merged_skeleton_mean = self.merged_skeleton_mean_map[
                np.nonzero(self.merged_skeleton_mean_map)].mean()
        print(f"Get a std skeleton for all subject = single value")
        self.merged_skeleton_std = self.merged_skeleton_std_map[
                np.nonzero(self.merged_skeleton_std_map)].mean()

        # assign 1 for voxels where all subject have skeleton
        # assign 0 for voxels where only some subjects have skeleton
        print("Get alteration map")
        self.skeleton_alteration_map = np.where(
            (self.merged_skeleton_data_bin_mean != 0) &
            (self.merged_skeleton_data_bin_mean != 1),
            1, 0)

    def subject_level_summary(self):
        """Summarize subject skeletons

        Attributes:
            subject_nonzero_means: list, mean of non-zero skeleton
            subject_nonzero_stds: list, std of non-zero skeleton
        """
        # Non-zero mean values in each subject skeleton
        self.subject_nonzero_means = []
        self.subject_nonzero_means_left = []
        self.subject_nonzero_means_right = []
        self.subject_nonzero_stds = []
        self.subject_nonzero_voxel_count = []

        # loop through each subject array
        # TODO: change here to array computation
        print('loop through each subject array')
        for vol_num in np.arange(self.merged_skeleton_data.shape[-1]):
            vol_data = self.merged_skeleton_data[:, :, :, vol_num]
            left_vol_data = self.merged_skeleton_data[90:, :, :, vol_num]
            right_vol_data = self.merged_skeleton_data[:90, :, :, vol_num]

            non_zero_mean = vol_data[np.nonzero(vol_data)].mean()
            non_zero_mean_left = left_vol_data[np.nonzero(left_vol_data)].mean()
            non_zero_mean_right = right_vol_data[np.nonzero(right_vol_data)].mean()
            non_zero_std = vol_data[np.nonzero(vol_data)].std()
            non_zero_voxel_count = len(np.where(vol_data != 0)[0])

            self.subject_nonzero_means.append(non_zero_mean)
            self.subject_nonzero_means_left.append(non_zero_mean_left)
            self.subject_nonzero_means_right.append(non_zero_mean_right)
            self.subject_nonzero_stds.append(non_zero_std)
            self.subject_nonzero_voxel_count.append(non_zero_voxel_count)
        print('loop through each subject array - finished')

    def subject_level_summary_with_warp(self, warp_dir, caselist):
        """Summarize subject skeletons

        Attributes:
            subject_nonzero_means: list, mean of non-zero skeleton
            subject_nonzero_stds: list, std of non-zero skeleton
        """
        # zero in the skeleton
        self.subject_zero_skeleton_values = []

        with open(caselist, 'r') as f:
            cases = [x.strip() for x in f.readlines()]
        # loop through each subject array
        for vol_num in np.arange(self.merged_skeleton_data.shape[-1]):
            vol_data = self.merged_skeleton_data[:, :, :, vol_num]
            subject_id = cases[vol_num]
            warp_data_loc = list(Path(warp_dir).glob(f'*{subject_id}*'))[0]
            warp_data = get_nifti_data(warp_data_loc)
            # zero where the mask is not zero
            zero_in_the_skeleton_coord = np.where(
                (self.mask_data == 1) & (vol_data == 0)
                )

            self.subject_zero_skeleton_values.append(
                    warp_data[zero_in_the_skeleton_coord])

    def subject_level_summary_with_mask(self, mask, threshold):
        """Summarize subject skeletons

        Attributes:
            subject_nonzero_means: list, mean of non-zero skeleton
            subject_nonzero_stds: list, std of non-zero skeleton
        """

        mask_data = get_nifti_data(mask)
        mask_data = np.where(mask_data > threshold, 1, 0)

        # Non-zero mean values in each subject skeleton
        self.subject_masked_means = []
        self.subject_masked_means_left = []
        self.subject_masked_means_right = []
        self.subject_masked_stds = []
        self.subject_nonzero_voxel_count = []

        # loop through each subject array
        for vol_num in np.arange(self.merged_skeleton_data.shape[-1]):
            vol_data = self.merged_skeleton_data[:, :, :, vol_num] * mask_data
            left_vol_data = vol_data[90:, :, :]
            right_vol_data = vol_data[:90, :, :]

            non_zero_mean = vol_data[np.nonzero(vol_data)].mean()
            non_zero_mean_left = left_vol_data[np.nonzero(left_vol_data)].mean()
            non_zero_mean_right = right_vol_data[np.nonzero(right_vol_data)].mean()
            non_zero_std = vol_data[np.nonzero(vol_data)].std()
            non_zero_voxel_count = len(np.where(vol_data != 0)[0])

            self.subject_masked_means.append(non_zero_mean)
            self.subject_masked_means_left.append(non_zero_mean_left)
            self.subject_masked_means_right.append(non_zero_mean_right)
            self.subject_masked_stds.append(non_zero_std)
            self.subject_nonzero_voxel_count.append(non_zero_voxel_count)

class SkeletonDir:
    """TBSS skeleton directory object"""
    def __init__(self, skeleton_dir):
        """Load list of nifti files in skeleton_dir and make df"""
        self.skeleton_dir = Path(skeleton_dir)
        self.skeleton_files = list(self.skeleton_dir.glob('*nii.gz'))

        self.df = pd.DataFrame({
            'files': self.skeleton_files
        })

    def summary(self):
        """Summarize skeleton"""
        # list of all skeleton nifti files in numpy arrays
        arrays = [get_nifti_data(x) for x in self.skeleton_files]

        # merge skeleton files
        self.merged_skeleton_data = np.stack(arrays, axis=3)

        self.means = [x[np.nonzero(x)].mean() for x in arrays]
        self.df['mean'] = self.means
        self.stds = [x[np.nonzero(x)].std() for x in arrays]
        self.df['std'] = self.stds

        self.mean = self.merged_skeleton_data[
            np.nonzero(self.merged_skeleton_data)].mean()
        self.std = self.merged_skeleton_data[
            np.nonzero(self.merged_skeleton_data)].std()

        self.merged_data_df = pd.DataFrame({
            'merged mean': [self.mean],
            'merged std': [self.std]
        })

    def merge_demo_df(self, demo_df, merge_on='subject'):
        self.df['subject'] = self.df['files'].apply(
            lambda x: x.name).str.split('_').str[0]
        self.df = pd.merge(self.df, demo_df, on=merge_on, how='left')

    def get_group_figure(self):
        """Group average figure of skeleton

        - skeleton group average as ahline
        - skeleton subject average as ahline
        - tests between subject averages between groups
        """
        plt.style.use('default')

        self.g = sns.catplot(
                x='group',
                y='mean',
                hue='group',
                hue_order=self.df.group.unique(),
                data=self.df)

        if self.df['mean'].mean() < 0.005:
            self.g.ax.set_ylim(
                self.df['mean'].min() - (self.df['mean'].std()/3),
                self.df['mean'].max() - (self.df['mean'].std()/3)
                )

        self.g.fig.set_size_inches(8, 4)
        self.g.fig.set_dpi(150)
        self.g.ax.set_ylabel(f'{self.modality}')
        self.g.ax.set_xlabel('Group')
        self.g.ax.set_title(
            f'Average {self.modality} in skeleton for all subjects',
            fontweight='bold')

        # tick labels to have number of groups
        def get_ticklabels(tmp_df, group):
            row_count_for_group = len(tmp_df[tmp_df.group == group])
            return f'{group} ({row_count_for_group})'

        self.g.ax.set_xticklabels([get_ticklabels(self.df, x) for x in
                                   self.df.group.unique()])
        # group_list = corrpMap.group_labels
        # print(group_list)

        # average line
        line_width = 0.3
        gb = self.df.groupby('group')
        for num, group in enumerate(self.df.group.unique()):
            table = gb.get_group(group)
            average = table['mean'].mean()
            self.g.ax.plot([num-line_width, num+line_width],
                           [average, average])

        # Add stat information to the graph
        height = 0.9
        two_groups_perm = list(combinations(self.df.group.unique(), 2))

        # if two groups
        if len(self.df.group.unique()) == 2:
            height_step = 0.8 / len(two_groups_perm)
        else:
            height_step = 0.8 / (len(two_groups_perm) + 1)

        # two group comparisons
        # TODO: add ANCOVA
        for g1, g2 in two_groups_perm:
            gb = self.df.groupby('group')
            g1_means = gb.get_group(g1)['mean']
            g2_means = gb.get_group(g2)['mean']

            # t, p = ss.ttest_ind(g1_means, g2_means)
            t, p, dof = ttest(g1_means, g2_means)

            if p < 0.05:
                text = f'{g1} vs {g2}\nT ({int(dof)}) = {t:.2f}, P = {p:.2f}*'
            else:
                text = f'{g1} vs {g2}\nT ({int(dof)}) = {t:.2f}, P = {p:.2f}'

            self.g.ax.text(1, height, text,
                           transform=self.g.ax.transAxes)
            height -= height_step

        # ANCOVA if there are more than two groups
        if len(self.df.group.unique()) > 2:
            anova_df = anova(self.df, 'mean ~ group')
            f_val = anova_df.loc['group', 'F']
            dof = anova_df.loc['group', 'df']
            p = anova_df.loc['group', 'PR(>F)']
            if p < 0.05:
                text = f'ANOVA\nF ({int(dof)}) = '\
                       f'{f_val:.2f}, P = {p:.2f}*'
            else:
                text = f'ANOVA\nF ({int(dof)}) = '\
                       f'{f_val:.2f}, P = {p:.2f}'
            self.g.ax.text(1, height, text,
                           transform=self.g.ax.transAxes)

    def get_subject_skeleton_volume_figure(self):
        """Subject skeleton volume figure """

        self.g_skel_vol = sns.catplot(
                x='subject',
                y='skeleton_volume',
                data=self.df)

        self.g_skel_vol.fig.set_size_inches(8, 4)
        self.g_skel_vol.fig.set_dpi(150)
        self.g_skel_vol.ax.set_ylabel(f'{self.modality} skeleton volume')

        if len(self.df) > 100:
            for tick in self.g_skel_vol.ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
        self.g_skel_vol.ax.set_xlabel('Subject')
        self.g_skel_vol.ax.set_title(
            f'Volume of {self.modality} skeleton for all subjects',
            fontweight='bold')

        # highlight the subject who are off the most common value
        most_common_volume = self.df['skeleton_volume'].value_counts().idxmax()
        for index, row in self.df.iterrows():
            if row.skeleton_volume != most_common_volume:
                self.g_skel_vol.ax.text(
                    index,
                    row.skeleton_volume,
                    row.subject, ha='center', va='center')

    def get_subject_zero_skeleton_figure(self):
        plt.style.use('default')
        """Subject skeleton volume figure """
        tmp_df = self.df.copy()

        # tmp_df.at[0, 'zero_skeleton'] = [1,2,3,4]
        for index, row in tmp_df.iterrows():
            try:
                tmp_df.loc[index, 'mean'] = np.mean(row['zero_skeleton'])
                tmp_df.loc[index, 'min'] = np.min(row['zero_skeleton'])
                tmp_df.loc[index, 'max'] = np.max(row['zero_skeleton'])
                tmp_df.loc[index, 'std'] = np.std(row['zero_skeleton'])
                tmp_df.loc[index, 'count'] = len(row['zero_skeleton'])
            except:
                tmp_df.loc[index, 'mean'] = 0
                tmp_df.loc[index, 'min'] = 0
                tmp_df.loc[index, 'max'] = 0
                tmp_df.loc[index, 'std'] = 0
                tmp_df.loc[index, 'count'] = 0

        # select only the subjects with zero voxels
        zv_subj_nums = tmp_df[tmp_df['count'] != 0].index

        tmp_df = pd.melt(
            tmp_df, id_vars='subject',
            var_name='zero skeleton info',
            value_vars=['mean', 'min', 'max', 'std', 'count'],
            value_name='value').reset_index()

        self.g_skel_zero = sns.catplot(
                x='subject',
                y='value',
                row='zero skeleton info',
                sharey=False,
                data=tmp_df)

        self.g_skel_zero.fig.set_size_inches(8, 8)
        self.g_skel_zero.fig.set_dpi(150)
        for ax in np.ravel(self.g_skel_zero.axes):
            var = ax.get_title().split(' = ')[1]
            ax.set_title('')
            ax.set_ylabel(var)

            # ax.set_ylabel(f'{self.modality} zero-skeleton mean')
            if len(self.df) > 50:
                ax.set_xticks(zv_subj_nums)
                ax.set_xticklabels(zv_subj_nums)
                # for tick in ax.xaxis.get_major_ticks():
                    # tick.label.set_fontsize(8)

            # highlight the subject with zero voxels in the skeleton
            for subj_num in zv_subj_nums:
                ax.axvline(subj_num, color='r', alpha=0.3, ls='--')

        ax.set_xlabel('Subject')

        self.g_skel_zero.fig.suptitle(
            f'Summary of values in the warped {self.modality} images\n' 
            f'at the zero voxels in the skeletonized {self.modality} map for '
            'all subjects',
            fontweight='bold', y=1.05)

        # # highlight the subject who are off the most common value
        # most_common_volume = self.df['skeleton_volume'].value_counts().idxmax()
        # for index, row in self.df.iterrows():
            # if row.skeleton_volume != most_common_volume:
                # self.g_skel_zero.ax.text(
                    # index,
                    # row.skeleton_volume,
                    # row.subject, ha='center', va='center')



class SkeletonDirSig(SkeletonDir):
    """TBSS skeleton directory object with significant region"""

    def summary(self):
        """Summarize skeleton"""
        # list of all skeleton nifti files in numpy arrays
        arrays = [get_nifti_data(x) for x in self.skeleton_files]

        # merge skeleton files
        self.merged_skeleton_data = np.stack(arrays, axis=3)
        self.mask_4d = np.broadcast_to(
            self.sig_mask,
            self.merged_skeleton_data.shape)

        self.means = [x[self.mask == 1].mean() for x in arrays]
        self.df['mean'] = self.means

        self.stds = [x[self.mask == 1].std() for x in arrays]
        self.df['std'] = self.stds

        self.mean = self.merged_skeleton_data[
            self.mask_4d == 1].mean()
        self.std = self.merged_skeleton_data[
            self.mask_4d == 1].std()

        self.merged_data_df = pd.DataFrame({
            'merged mean': [self.mean],
            'merged std': [self.std]
        })



def skeleton_summary(merged_4d_file, skeleton_mask, tbss_all_loc, **kwargs):
    """ Make summary from corrpMap, using its merged_skeleton"""
    caselist = str(Path(tbss_all_loc) / 'log/caselist.txt')


    # update with design contrast and matrix
    mergedSkeleton = MergedSkeleton(merged_4d_file, skelton_mask)

    if 'directory' in kwargs:
        location = kwargs.get('directory')
        contrast = Path(location) / 'design.con'
        matrix = Path(location) / 'design.mat'
        mergedSkeleton.get_matrix_info()
        mergedSkeleton.get_contrast_info()


    mergedSkeleton.skeleton_level_summary()
    mergedSkeleton.subject_level_summary()

    # set modality of the merged skeleton file
    mergedSkeleton.modality = Path(merged_4d_file).name.split('_')[1]
    warp_dir = str(Path(tbss_all_loc) / mergedSkeleton.modality / 'warped')

    # create a dataframe that has
    # - mean of values in the non zero skeleton for each subject
    # - standard deviation of values in the nonzero skeleton for each subject
    print('Creating figures to summarize information from the skeletons')
    print(f'Using information in {warp_dir} to '
          'extract values in warped maps')

    with open(caselist, 'r') as f:
        cases = [x.strip() for x in f.readlines()]

    mergedSkeleton.subject_level_summary_with_warp(warp_dir, caselist)
    mergedSkeleton.df = pd.DataFrame({
        'subject': cases,
        'mean': mergedSkeleton.subject_nonzero_means,
        'skeleton_volume': mergedSkeleton.subject_nonzero_voxel_count,
        'skeleton non_zero_std': mergedSkeleton.subject_nonzero_stds,
        'zero_skeleton': mergedSkeleton.subject_zero_skeleton_values,
        })
    SkeletonDir.get_subject_zero_skeleton_figure(mergedSkeleton)
    out_image_loc = re.sub('.nii.gz',
                           '_skeleton_zero_mean_in_warp.png',
                           merged_4d_file)

    mergedSkeleton.g_skel_zero.savefig(
            out_image_loc,
            facecolor='white', dpi=200)
    plt.close()

    # Figure that shows
    # - skeleton group average as ahline
    # - skeleton subject average as scatter dots
    # - tests between subject averages between groups
    plt.style.use('default')
    SkeletonDir.get_group_figure(mergedSkeleton)
    out_image_loc = re.sub('.nii.gz',
                           '_skeleton_average_for_all_subjects.png',
                           str(merged_4d_file))
    mergedSkeleton.g.savefig(out_image_loc, facecolor='white', dpi=200)
    plt.close()
    print('\t- Average for the skeleton in each subjects')

    SkeletonDir.get_subject_skeleton_volume_figure(mergedSkeleton)
    out_image_loc = re.sub('.nii.gz',
                           '_skeleton_volume_for_all_subjects.png',
                           str(merged_4d_file))
    mergedSkeleton.g_skel_vol.savefig(out_image_loc, facecolor='white', dpi=200)
    plt.close()
    print('\t- Volume for the skeleton in each subjects')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
The most simple way to use the script is
  cd stats
  ls
    all_FA_skeleton.nii.gz
    tbss_FA_tfce_corrp_tstat1.nii.gz
    tbss_FA_tfce_corrp_tstat2.nii.gz
    design.mat
    design.con
  randomise_summary.py
        ''', epilog="Kevin Cho Thursday, August 22, 2019")

    argparser.add_argument(
        "--dir", "-d",
        type=str,
        help='design directory')

    argparser.add_argument(
        "--contrast", "-con",
        type=str,
        help='design contrast')

    argparser.add_argument(
        "--matrix", "-mat",
        type=str,
        help='design matrix')

    argparser.add_argument(
        "--merged_4d_file", "-i",
        type=str,
        help='Merged 4d file')

    argparser.add_argument(
        "--skeleton_mask", "-m",
        type=str,
        help='Merged 4d file')

    argparser.add_argument(
        "--warp_dir", "-w",
        type=str,
        help='warp dir')

    argparser.add_argument(
        "--caselist", "-c",
        type=str,
        help='caselist')

    argparser.add_argument(
        "--tbss_all_loc", "-tal",
        type=str,
        help='tbss_all location')

    args = argparser.parse_args()

    if args.dir:
        skeleton_summary(args.merged_4d_file,
                         args.skeleton_mask,
                         args.tbss_all_loc,
                         directory=args.dir)
    else:
        skeleton_summary(args.merged_4d_file, 
                         args.skeleton_mask,
                         args.tbss_all_loc,
                         matrix=args.matrix, 
                         contrast=args.contrast)
