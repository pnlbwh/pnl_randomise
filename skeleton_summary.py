# nifti
import nibabel as nb

# table and array
import numpy as np
import pandas as pd

# os tools
from pathlib import Path

# import print option
# from kchopy.kcho_utils import print_df

# figure
import seaborn as sns
# import matplotlib.pyplot as plt

# stats
from itertools import combinations
import scipy.stats as ss

from kchopy.kcho_utils import print_head
from kchostats.ancova import anova, ttest

class MergedSkeleton:
    """TBSS all_modality_skeleton map object"""
    def __init__(self, merged_skeleton_loc):
        """Read in merged skeleton nifti file"""
        self.merged_skeleton_loc = merged_skeleton_loc
        self.merged_skeleton_img = nb.load(str(self.merged_skeleton_loc))
        print(f"Reading {merged_skeleton_loc}")
        self.merged_skeleton_data = self.merged_skeleton_img.get_data()

        # ENIGMA
        self.enigma_dir = Path('/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
        self.enigma_fa_loc = self.enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
        self.enigma_skeleton_mask_loc = self.enigma_dir / \
            'ENIGMA_DTI_FA_skeleton_mask.nii.gz'

        # binarize merged skeleton map
        self.merged_skeleton_data_bin = np.where(
            self.merged_skeleton_data == 0, 0, 1)
        self.merged_skeleton_data_bin_sum = np.sum(
            self.merged_skeleton_data_bin, axis=3)
        self.merged_skeleton_data_bin_mean = np.mean(
            self.merged_skeleton_data_bin, axis=3)

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
        self.merged_skeleton_mean_map = np.mean(
            self.merged_skeleton_data, axis=3)
        self.merged_skeleton_std_map = np.std(
                self.merged_skeleton_data, axis=3)

        # skeleton mean and std values
        self.merged_skeleton_mean = self.merged_skeleton_mean_map[
                np.nonzero(self.merged_skeleton_mean_map)].mean()
        self.merged_skeleton_std = self.merged_skeleton_std_map[
                np.nonzero(self.merged_skeleton_std_map)].mean()

        # assign 1 for voxels where all subject have skeleton
        # assign 0 for voxels where only some subjects have skeleton
        self.skeleton_alteration_map = np.where(
            (self.merged_skeleton_data_bin_mean != 0) &
            (self.merged_skeleton_data_bin_mean != 1),
            1, 0)

        # TODO: diff map between ENIGMA skeleton mask
        target_data = nb.load(str(self.enigma_skeleton_mask_loc)).get_data()

    def subject_level_summary(self):
        """Summarize subject skeletons

        Attributes:
            subject_nonzero_means: list, mean of non-zero skeleton
            subject_nonzero_stds: list, std of non-zero skeleton
        """
        # Non-zero mean values in each subject skeleton
        self.subject_nonzero_means = []
        self.subject_nonzero_stds = []
        # loop through each subject array
        for vol_num in np.arange(self.merged_skeleton_data.shape[-1]):
            vol_data = self.merged_skeleton_data[:, :, :, vol_num]
            non_zero_mean = vol_data[np.nonzero(vol_data)].mean()
            non_zero_std = vol_data[np.nonzero(vol_data)].std()
            self.subject_nonzero_means.append(non_zero_mean)
            self.subject_nonzero_stds.append(non_zero_std)


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
        arrays = [nb.load(str(x)).get_data() for x in self.skeleton_files]

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
        self.g = sns.catplot(x='group', y='mean', hue='group', data=self.df)
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

        # average line
        line_width = 0.3
        for num, (group, table) in enumerate(self.df.groupby('group')):
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
