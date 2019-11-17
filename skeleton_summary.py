# nifti
import nibabel as nb

# table and array
import numpy as np
import pandas as pd

# os tools
from pathlib import Path

# import print option
from fsl_randomise_utils import print_df

# figure
import seaborn as sns
import matplotlib.pyplot as plt

# stats
from itertools import combinations
import scipy.stats as ss


class MergedSkeleton:
    """TBSS all_modality_skeleton map object"""
    def __init__(self, merged_skeleton_loc):
        """Read in merged skeleton nifti file"""
        self.merged_skeleton_loc = merged_skeleton_loc
        self.merged_skeleton_img = nb.load(str(self.merged_skeleton_loc))
        self.merged_skeleton_data = self.merged_skeleton_img.get_data()

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

        self.means = [x[np.nonzero(x)].mean() for x in arrays]
        self.df['mean'] = self.means
        self.stds = [x[np.nonzero(x)].std() for x in arrays]
        self.df['std'] = self.stds

        # merge skeleton files
        self.merged_data = np.stack(arrays, axis=3)

        self.mean = self.merged_data[np.nonzero(self.merged_data)].mean()
        self.std = self.merged_data[np.nonzero(self.merged_data)].std()

        self.merged_data_df = pd.DataFrame({
            'merged mean':[self.mean],
            'merged std':[self.std]
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
        self.g.ax.set_title(f'Average {self.modality} in ANTS-TBSS skeletons',
                       fontweight='bold')

        # tick labels to have number of groups
        get_ticklabels = lambda tmp_df, x: \
                f'{x} ({len(tmp_df[tmp_df.group==x])})'
        self.g.ax.set_xticklabels([get_ticklabels(self.df, x) for x in \
                             self.df.group.unique()])

        # average line
        line_width=0.3
        for num, (group, table) in enumerate(self.df.groupby('group')):
            average = table['mean'].mean()
            self.g.ax.plot([num-line_width, num+line_width], 
                           [average, average])
            
        # Add stat information to the graph
        height=0.8
        two_groups_perm = list(combinations(self.df.group.unique(), 2))
        for g1, g2 in two_groups_perm:
            gb = self.df.groupby('group')
            g1_means = gb.get_group(g1)['mean']
            g2_means = gb.get_group(g2)['mean']
            
            t,p = ss.ttest_ind(g1_means, g2_means)
            
            if p < 0.05:
                text = f'{g1} vs {g2}\nP ({t:.2f}) = {p:.2f}*'
            else:
                text = f'{g1} vs {g2}\nP ({t:.2f}) = {p:.2f}'

            self.g.ax.text(1, height, text, 
                      transform=self.g.ax.transAxes)
            height -= 0.3

