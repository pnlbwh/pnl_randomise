#!/data/pnl/kcho/anaconda3/bin/python

from pathlib import Path
import tempfile

# figures
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage

# Imaging
import nibabel as nb
import argparse
import pandas as pd
import re
import numpy as np
from os import environ
import os
import sys

# utils
from fsl_randomise_utils import print_df, print_head
from skeleton_summary import MergedSkeleton
from itertools import product
import inquirer

mpl.use('Agg')
pd.set_option('mode.chained_assignment', None)

'''
TODO:
    - `-i` and without `-i` to use similar flow.
    - Test
    - Save summary outputs in pdf, csv or excel?
    - TODO add interaction information (group info to the get_contrast_info_english)
    - Check whether StringIO.io is used
    - Estimate how much voxels overlap with different atlases with varying 
      thresholds.
    - Move useful functions to kcho_utils.
    - Parallelize
'''


class RandomiseRun:
    """Randomise output class

    Used to contain information about FSL randomise run.

    Key arguments:
    location -- str or Path object of a randomise output location.
              Preferably with a 'design.con' and 'design.mat' inside.
              (default:'.')
    contrast_file -- design contrast file used as the input for randomise.
                     (default:'design.con')
    matrix_file -- design matrix file used as the input for randomise.
                   (default:'design.mat')

    Below argument has been removed from RandomiseRun:
    threshold -- float, 1-p value used to threhold for significance.
                 (default:0.95)
    """

    def __init__(self,
                 location='.',
                 contrast_file='design.con',
                 matrix_file='design.mat'):
        # define inputs
        self.location = Path(location)

        # if matrix_file argument was not given, it would have
        # been set as 'design.mat' --> make it into a full path
        if matrix_file == 'design.mat':
            self.matrix_file = self.location / matrix_file
        else:
            self.matrix_file = Path(matrix_file)

        # if contrast_file argument was not given, it would have
        # been set as 'design.con' --> make it into a full path
        if contrast_file == 'design.con':
            self.contrast_file = self.location / contrast_file
        else:
            self.contrast_file = Path(contrast_file)

    def get_contrast_info(self):
        """Read design contrast file into a numpy array

        self.contrast_array : numpy array of the contrast file excluding the
                           headers
        """

        with open(self.contrast_file, 'r') as f:
            lines = f.readlines()
            headers = [x for x in lines
                       if x.startswith('/')]

        last_header_line_number = lines.index(headers[-1]) + 1
        self.contrast_array = np.loadtxt(self.contrast_file,
                                         skiprows=last_header_line_number)

    def get_contrast_info_english(self):
        """Read design contrast file into a numpy array

        self.contrast_lines : attributes that states what each row in the
                           contrast array represents
       TODO:
           select group columns
        """

        # if all lines are group comparisons -->
        # simple group comparisons or interaction effect
        if (self.contrast_array.sum(axis=1) == 0).all():
            # TODO : do below at the array level?
            df_tmp = pd.DataFrame(self.contrast_array)
            self.contrast_lines = []
            for contrast_num, row in df_tmp.iterrows():
                # name of the column with value of 1
                pos_col_num = row[row == 1].index.values[0]
                neg_col_num = row[row == -1].index.values[0]

                # if interaction : they have zeros in the group column
                # TODO : make this more efficient later
                half_cols = (self.contrast_array.shape[0] / 2) + 1
                if pos_col_num not in list(range(int(half_cols+1))):
                    if pos_col_num < neg_col_num:
                        text = 'Negative Interaction'
                    else:
                        text = 'Positive Interaction'

                else:
                    # Change order of columns according to their column numbers
                    if pos_col_num < neg_col_num:
                        text = f'Group {pos_col_num+1} > Group {neg_col_num+1}'
                    else:
                        text = f'Group {neg_col_num+1} < Group {pos_col_num+1}'
                self.contrast_lines.append(text)
        # TODO add interaction information
        # if group column is zero
        # 0   0   1    -1

        # if all rows sum to 1 --> the correlation contrast
        # TODO: there is a positibility of having 0.5 0.5?
        elif (np.absolute(self.contrast_array.sum(axis=1)) == 1).all():
            # TODO : do below at the array level?
            df_tmp = pd.DataFrame(self.contrast_array)
            self.contrast_lines = []
            for contrast_num, row in df_tmp.iterrows():
                # name of the column with value of 1
                col_num = row[row != 0].index.values[0]

                # Change order of columns according to their column numbers
                if row.loc[col_num] == 1:
                    text = f'Positively correlated with col {col_num+1}'
                else:
                    text = f'Negatively correlated with col {col_num+1}'
                self.contrast_lines.append(text)

        print(self.contrast_lines)

    def get_matrix_info(self):
        """Read design matrix file into a numpy array and summarize
        'matrix_header' : headers in the matrix file
        'matrix_array' : numpy array matrix part of the matrix file
        'matrix_df' : pandas dataframe of matrix array

        TODO:
            add function defining which column is group
        """
        with open(self.matrix_file, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            matrix_lines = [x for x in lines if x.startswith('/')]

        self.matrix_header = [
            x for x in matrix_lines if
            not x.startswith('/NumWaves') and
            not x.startswith('/NumContrasts') and
            not x.startswith('/NumPoints') and
            not x.startswith('/PPheights') and
            not x.startswith('/Matrix')]

        self.matrix_header = '\n'.join(
            [x[1:].strip() for x in self.matrix_header])

        the_line_with_matrix = lines.index('/Matrix') + 1
        self.matrix_array = np.loadtxt(self.matrix_file,
                                       skiprows=the_line_with_matrix)

        # matrix array into pandas dataframe
        self.matrix_df = pd.DataFrame(self.matrix_array)
        # rename columns to have 'col ' in each
        self.matrix_df.columns = [f'col {x}' for x in
                                  self.matrix_df.columns]
        # summarize matrix
        self.matrix_info = self.matrix_df.describe()
        self.matrix_info = self.matrix_info.loc[
                ['mean', 'std', 'min', 'max'], :]
        self.matrix_info = self.matrix_info.round(decimals=2)

        # For each column of the matrix, add counts of unique values to
        # self.matrix_info
        for col in self.matrix_df.columns:
            # create a dataframe that contains unique values as the index
            unique_values = self.matrix_df[col].value_counts().sort_index()
            # If there are less unique values than the half the number of
            # all data  for the column
            if len(unique_values) < len(self.matrix_df) / 2:
                # unique values as an extra row
                self.matrix_info.loc['unique values', col] = np.array2string(
                    unique_values.index)[1:-1]
                # count of each unique value as an extra row
                self.matrix_info.loc['count', col] = np.array2string(
                    unique_values.values)[1:-1]
            # If there are 5 or more unique values in the column, leave the
            else:
                # 'unique' and 'count' as 'continuous values'
                self.matrix_info.loc['unique values', col] = \
                        'continuous values'
                self.matrix_info.loc['count', col] = 'continuous values'

        # define which column represent group column
        # Among columns where their min value is 0 and max value is 1,
        min_0_max_1_col = [x for x in self.matrix_df.columns
                           if self.matrix_df[x].isin([0, 1]).all()]

        if 'col 0' not in min_0_max_1_col:
            print("Group Column is not in the first column")
            print("Setting self.group_cols=['no group col']")
            self.group_cols = ['no group col']

        else:
            # if sum of each row equal to 1, these columns would highly likely
            # be group columns
            if (self.matrix_df[min_0_max_1_col].sum(axis=1) == 1).all():
                self.group_cols = min_0_max_1_col
            # If not, remove a column from the list of columns at the end, and
            # test whether each row sums to 1
            elif (self.matrix_df[min_0_max_1_col[:-1]].sum(axis=1) == 1).all():
                self.group_cols = min_0_max_1_col[:-1]
            elif (self.matrix_df[min_0_max_1_col[:-2]].sum(axis=1) == 1).all():
                self.group_cols = min_0_max_1_col[:-2]
            elif (self.matrix_df[min_0_max_1_col[:-3]].sum(axis=1) == 1).all():
                self.group_cols = min_0_max_1_col[:-3]
            else:
                self.group_cols = min_0_max_1_col[0]

            # 'unique' and 'count' columns of group columns
            for group_num, col in enumerate(self.group_cols, 1):
                # unique values as an extra row
                self.matrix_info.loc['unique', col] = f"Group {group_num}"
                # count of each unique value as an extra row
                self.matrix_info.loc['count', col] = \
                    (self.matrix_df[col] == 1).sum()

    def get_corrp_files_glob_string(self, glob_string):
        """Find corrp files and return a list of Path objects
        """
        corrp_ps = list(self.location.glob(glob_string))
        # remove corrp files that are produced in the parallel randomise
        self.corrp_ps = [str(x) for x in corrp_ps if 'SEED' not in x.name]

        if len(self.corrp_ps) == 0:
            print(f'There is no corrected p-maps in {self.location}')

    def get_corrp_files(self):
        """Find corrp files and return a list of Path objects
        """
        corrp_ps = list(self.location.glob('*corrp*.nii.gz'))
        # remove corrp files that are produced in the parallel randomise
        self.corrp_ps = [str(x) for x in corrp_ps if 'SEED' not in x.name]

        if len(self.corrp_ps) == 0:
            print(f'There is no corrected p-maps in {self.location}')

    def print_matrix_info(self):
        print_head('Matrix summary')
        print(self.location)
        print(self.location / self.contrast_file)
        print(self.location / self.matrix_file)
        print(f'total number of data point : {len(self.matrix_df)}')
        print(f'Group columns are : ' + ', '.join(self.group_cols))
        print_df(self.matrix_info)


class CorrpMap(RandomiseRun):
    """Multiple comparison corrected randomise output class

    Used to contain information about corrected p maps of randomise outputs

    Key arguments:
    loc -- str or Path object, location for the corrp map.
    threshold -- float, 1-p threhold for significance.
    """

    def __init__(self, location, threshold):
        self.location = Path(location)
        self.name = self.location.name

        # if modality is included in its name
        try:
            self.modality = re.search(
                '.*(FW|FA|MD|RD|AD|MD|FAt|FAc|FAk|MK|MKc|MKk|MDt|RDt|ADt|MDt)_',
                self.location.name).group(1)
        except:
            self.modality = ''

        # find all merged file
        self.get_merged_skeleton_file()
        self.threshold = threshold
        self.test_kind = re.search(r'(\w)stat\d+.nii.gz', self.name).group(1)
        self.stat_num = re.search(r'(\d+).nii.gz', self.name).group(1)

        # Below variables are to estimate number of significant voxels in each
        # hemisphere
        self.fsl_dir = Path(environ['FSLDIR'])
        self.fsl_data_dir = self.fsl_dir / 'data'
        self.HO_dir = self.fsl_data_dir / 'atlases' / 'HarvardOxford'
        self.HO_sub_thr0_1mm = self.HO_dir / \
            'HarvardOxford-sub-maxprob-thr0-1mm.nii.gz'

        # enigma FA map - background settings
        self.enigma_dir = Path('/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
        self.enigma_fa_loc = self.enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
        self.enigma_skeleton_mask_loc = self.enigma_dir / \
            'ENIGMA_DTI_FA_skeleton_mask.nii.gz'

        self.check_significance()
        if self.significant:
            self.get_significant_info()
            self.get_significant_overlap_with_HO()
        self.make_df()

    def get_merged_skeleton_file(self):
        """Search for the matching merged skeleton files

        Searches for the merged skeleton files based on the detected modality.
        """
        # list of directories and serach patterns
        list_search_directories = [
                self.location.parent,
                self.location.parent.parent
            ]
        list_of_patters = [
            f'*all*_{self.modality}[_.]*nii.gz',
            f'*{self.modality}*merged*.nii.gz'
            ]
        # get combinations of the two lists
        list_of_dir_pat = list(product(
            list_search_directories,
            list_of_patters))

        # search files
        matching_files = []
        for s_dir, pat in list_of_dir_pat:
            try:
                mf = list(Path(s_dir).glob(pat))
                matching_files += mf
            except:
                pass

        matching_files = list(set(matching_files))
        # check matching_files list
        if len(matching_files) == 1:
            self.merged_4d_file = matching_files[0]
        elif len(matching_files) > 1:
            questions = [
                inquirer.List(
                    'merged_file',
                    message="There are more than one matching merged 4d "
                            f"skeleton for {self.location.name}. "
                            "Which is the correct merged file?",
                    choices=matching_files,
                    )
                ]
            answer = inquirer.prompt(questions)
            self.merged_4d_file = answer['merged_file']
        else:
            self.merged_4d_file = 'missing'

    def check_significance(self):
        """Check whether there is any significant voxel"""

        # read corrp images
        img = nb.load(str(self.location))
        data = img.get_data()

        # add data resolution attribute
        self.data_shape = data.shape

        # max p-value
        self.voxel_max_p = data.max()

        # Discrepancy between numpy and FSL
        if len(data[(data < 0.95) & (data >= 0.9495)]) != 0:
            self.threshold = self.threshold - 0.00001
            print('There are voxels with p value between 0.9495 and 0.05. '
                  'These numbers are rounded up in FSL to 0.95. Threfore '
                  'to match to the FSL outputs, changing the threshold to '
                  '(threshold - 0.00001)')

        # any voxels significant?
        if (data >= self.threshold).any():
            self.significant = True
            self.corrp_data = data

        else:
            self.significant = False

    def get_significant_info(self):
        """Get information of significant voxels"""
        # total number of voxels in the skeleton
        self.vox_num_total = np.count_nonzero(self.corrp_data)

        # number of significant voxels: greater or equal to 0.95 by default
        self.significant_voxel_num = \
            np.count_nonzero(self.corrp_data >= self.threshold)

        # number of significant voxels / number of all voxels
        self.significant_voxel_percentage = \
            (self.significant_voxel_num / np.count_nonzero(self.corrp_data)) \
            * 100

        # summary of significant voxels
        sig_vox_array = self.corrp_data[self.corrp_data >= self.threshold]
        self.significant_voxel_mean = 1 - sig_vox_array.mean()
        self.significant_voxel_std = sig_vox_array.std()
        self.significant_voxel_max = 1 - sig_vox_array.max()

    def get_significant_overlap_with_HO(self):
        """Get overlap information of significant voxels with Harvard Oxford"""
        # Number of voxels in each hemisphere
        # Harvard-Oxford left and right hemisphere white matter masks
        # TODO: what to do if template space is not same as the HO space?
        HO_data = nb.load(str(self.HO_sub_thr0_1mm)).get_data()

        # Few voxels from ENIGMA template skeleton spreads into the area
        # defined as a gray matter by Harvard Oxford atlas
        try:
            left_mask_array = np.where((HO_data == 1) +
                                       (HO_data == 2), 1, 0)
            left_skeleton_array = self.corrp_data * left_mask_array
            right_mask_array = np.where((HO_data == 12) +
                                        (HO_data == 13), 1, 0)
            right_skeleton_array = self.corrp_data * right_mask_array

            # count significant voxels in each hemispheres
            self.significant_voxel_left_num = np.sum(
                left_skeleton_array >= self.threshold)
            self.significant_voxel_right_num = np.sum(
                right_skeleton_array >= self.threshold)

            if np.count_nonzero(left_skeleton_array) == 0:
                self.significant_voxel_left_percent = 0
            else:
                self.significant_voxel_left_percent = (
                    self.significant_voxel_left_num /
                    np.count_nonzero(left_skeleton_array)) * 100

            if np.count_nonzero(right_skeleton_array) == 0:
                self.significant_voxel_right_percent = 0
            else:
                self.significant_voxel_right_percent = (
                    self.significant_voxel_right_num /
                    np.count_nonzero(right_skeleton_array)) * 100
        except:
            print('** This study has a specific template. The number of '
                  'significant voxels in the left and right hemisphere '
                  'will not be estimated')
            self.significant_voxel_right_percent = 'unknown'
            self.significant_voxel_left_percent = 'unknown'

    def make_df(self):
        """Make summary pandas df of each corrp maps"""
        if self.significant:
            self.df = pd.DataFrame({
                'file name': [self.name],
                'Test': self.test_kind,
                'Modality': self.modality,
                'Stat num': self.stat_num,
                'Significance': self.significant,
                'Sig Max': self.voxel_max_p,
                'Sig Mean': self.significant_voxel_mean,
                'Sig Std': self.significant_voxel_std,
                '% significant voxels': self.significant_voxel_percentage,
                '% left': self.significant_voxel_left_percent,
                '% right': self.significant_voxel_right_percent
            })

            # round up columns that stars with percent
            for percent_col in [x for x in self.df.columns
                                if x.startswith('%')]:
                try:
                    self.df[percent_col] = self.df[percent_col].round(
                        decimals=1)
                except:
                    pass
        else:
            self.df = pd.DataFrame({
                'file name': [self.name],
                'Test': self.test_kind,
                'Modality': self.modality,
                'Stat num': self.stat_num,
                'Significance': self.significant,
                'Sig Max': self.voxel_max_p,
            })

    def update_with_contrast(self):
        '''Update CorrpMap class when there the contrast file is available
        (when self.contrast_array is available)

        Keyword argument
        contrast_array : numpy array, of the design contrast file. Created
                         by loading only the contrast lines text using
                         np.loadtxt.
        '''
        # Contrast map
        line_num = int(self.stat_num)-1

        self.contrast_line = self.contrast_array[line_num, :]
        self.contrast_text = self.contrast_lines[line_num]

        self.df['contrast'] = np.array2string(self.contrast_line,
                                              precision=2)[1:-1]
        try:
            self.df['contrast_text'] = self.contrast_lines[line_num]
        except:
            self.df['contrast_text'] = '-'

        # if f-test
        if self.test_kind == 'f':
            try:
                with open(str(self.location.parent / 'design.fts'), 'r') as f:
                    lines = f.readlines()
                    design_fts_line = [x for x in lines
                                       if re.search(r'^\d', x)][0]
                self.df['contrast'] = '* ' + design_fts_line
            except:
                self.df['contrast'] = 'f-test'

        # Reorder self.df to have file name and the contrast on the left
        self.df = self.df[['file name', 'contrast', 'contrast_text'] +
                          [x for x in self.df.columns if x not in
                              ['file name',
                               'contrast',
                               'contrast_text']]]
        return self.df

    def get_significant_cluster(self):
        """Get binary array of significant cluster"""
        self.significant_cluster_data = np.where(
            self.corrp_data >= self.threshold, 1, 0)

    def update_with_4d_data(self):
        """get mean values for skeleton files in the significant voxels

        Args:
            skeleton_files: list of Path objects, skeleton file locations.

        Retrun:
            df: pandas dataframe of
                'corrp_file', 'skeleton_file', 'average'
        TODO:
            - save significant voxels
            - parallelize
            - think about using all_modality_merged images?
        """
        merged_4d_data = nb.load(str(self.merged_4d_file)).get_data()
        significant_cluster_data = np.where(
            self.corrp_data >= self.threshold, 1, 0)

        self.cluster_averages = {}
        for vol_num in np.arange(merged_4d_data.shape[3]):
            average = merged_4d_data[:,:,:,vol_num]\
                    [significant_cluster_data == 1].mean()

            self.cluster_averages[vol_num] = average

        self.cluster_averages_df = pd.DataFrame.from_dict(
            self.cluster_averages,
            orient='index', 
            columns=[f'{self.modality} values in the significant '\
                     f'cluster {self.name}']
        )
        

    def get_atlas_query(self):
        """Return pandas dataframe summary of atlas_query outputs"""
        # threshold corrp file according to the threshold
        thresholded_map =  tempfile.NamedTemporaryFile(suffix='tmp.nii.gz')
        command = f'fslmaths {self.location} \
                -thr {self.threshold} -bin \
                {thresholded_map.name}'
        # TODO change below to kcho_util run
        os.popen(command).read()

        # run atlas query from FSL
        # label
        command = f'atlasquery \
                -m {thresholded_map.name} \
                -a "JHU ICBM-DTI-81 White-Matter Labels"'
        text_label = os.popen(command).read()
        # tract
        command = f'atlasquery \
                -m {thresholded_map.name} \
                -a "JHU White-Matter Tractography Atlas"'
        text_tract = os.popen(command).read()

        # Make pandas dataframe
        df_query_label = pd.read_csv(pd.compat.StringIO(text_label), 
                                     sep=':', 
                                     names=['Structure', 'Percentage'])
        df_query_label['atlas'] = 'Labels'
        df_query_tract = pd.read_csv(pd.compat.StringIO(text_tract), 
                                     sep=':', 
                                     names=['Structure', 'Percentage'])
        df_query_tract['atlas'] = 'Tracts'
        df_query = pd.concat([df_query_label, df_query_tract])

        df_query['file_name'] = self.name
        df_query = df_query[['file_name', 'Structure', 'Percentage', 'atlas']]

        df_query = df_query.sort_values(
            ['file_name', 'atlas', 'Percentage'], 
            ascending=False)
    
        # Remove texts bound by parentheses
        df_query['Structure'] = df_query['Structure'].apply(
            lambda x: re.sub('\(.*\)', '', x))

        # Adding 'Side' column
        df_query['Side'] = df_query['Structure'].str.extract('(L|R)$')

        # Remove side information from Structure column
        df_query['Structure'] = df_query['Structure'].str.replace(
            '(L|R)$', '').str.strip()
        df_query.loc[df_query['Side'].isnull(), 'Side'] = 'M'

        # Side column to wide format - 
        self.df_query = pd.pivot_table(
            index=['file_name', 'Structure', 'atlas'],
            columns='Side',
            values='Percentage', 
            data=df_query).reset_index()

        # TODO: change here later
        self.df_query = self.df_query.groupby('atlas').get_group('Labels')


    def get_figure_enigma(self, **kwargs):
        """Fig and axes attribute to CorrpMap"""

        # if study template is not ENIGMA
        if 'mean_fa' in kwargs:
            mean_fa_loc = kwargs.get('mean_fa')
            print(f'background image : {mean_fa_loc}')
            self.enigma_fa_data = nb.load(mean_fa_loc).get_data()

            mean_fa_skel_loc = re.sub('.nii.gz', '_skeleton.nii.gz',
                                      mean_fa_loc)
            print(f'background skeleton image: {mean_fa_skel_loc}')
            self.enigma_skeleton_data = nb.load(mean_fa_skel_loc).get_data()
        else:
            self.enigma_fa_data = nb.load(
                str(self.enigma_fa_loc)).get_data()
            self.enigma_skeleton_data = nb.load(
                str(self.enigma_skeleton_mask_loc)).get_data()

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

        elif hasattr(self, 'main_data_vmax'):
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

        self.enigma_skeleton_data = np.where(
            self.enigma_skeleton_data < 1,
            np.nan,
            self.enigma_skeleton_data)

        # Make fig and axes
        fig, axes = plt.subplots(ncols=self.ncols,
                                 nrows=self.nrows,
                                 figsize=(size_w * self.ncols,
                                          size_h * self.nrows),
                                 dpi=200)

        # For each axis
        for num, ax in enumerate(np.ravel(axes)):
            # background FA map
            img = ax.imshow(
                np.flipud(self.enigma_fa_data[:, :, slice_nums[num]].T),
                cmap='gray')

            # background skeleton
            img = ax.imshow(
                np.flipud(self.enigma_skeleton_data[:, :, slice_nums[num]].T),
                interpolation=None,
                cmap='ocean')

            # main data
            if hasattr(self, 'corrp_data_filled'):
                # tbss_fill FA maps
                img = ax.imshow(np.flipud(data[:, :, slice_nums[num]].T),
                                cmap='autumn',
                                interpolation=None,
                                vmin=0,
                                vmax=1)
            elif hasattr(self, 'main_data_vmax'):
                # for skeleton std data plot
                if self.main_data_vmax == 'free':
                    img = ax.imshow(np.flipud(data[:, :, slice_nums[num]].T),
                                    interpolation=None,
                                    cmap='cool',
                                    vmin=0)
            else:
                # stat maps
                img = ax.imshow(np.flipud(data[:, :, slice_nums[num]].T),
                                interpolation=None,
                                cmap='autumn',
                                vmin=self.threshold,
                                vmax=1)
            ax.axis('off')
            ax.annotate('z = {}'.format(slice_nums[num]),
                        (0.01, 0.1),
                        xycoords='axes fraction',
                        color='white')

        fig.subplots_adjust(hspace=0, wspace=0)

        axbar = fig.add_axes([0.9, 0.2, 0.03, 0.6])
        cb = fig.colorbar(img, axbar)
        # Set y tick label color
        cbytick_obj = plt.getp(cb.ax, 'yticklabels')

        plt.setp(cbytick_obj, color='white')
        cb.outline.set_edgecolor('white')
        cb.ax.yaxis.set_tick_params(color='white')

        self.fig = fig
        self.axes = axes

    def tbss_fill(self, outfile):
        command = f'tbss_fill  \
                {self.location} \
                {self.threshold} \
                {self.enigma_fa_loc} {outfile}'
        os.popen(command).read()


def skeleton_summary(corrpMap):
    """Make summary from corrpMap, using its merged_skeleton

    TODO:
        - make it run only once for each merged skeleton file
    """
    mergedSkeleton = MergedSkeleton(str(corrpMap.merged_4d_file))
    mergedSkeleton.skeleton_level_summary()

    mergedSkeleton.enigma_fa_loc = corrpMap.enigma_fa_loc
    mergedSkeleton.enigma_skeleton_mask_loc = corrpMap.enigma_skeleton_mask_loc
    mergedSkeleton.data_shape = corrpMap.data_shape
    mergedSkeleton.threshold = 0.01

    # Save mean and std maps
    img = nb.load(str(corrpMap.merged_4d_file))
    # nb.Nifti1Image(mergedSkeleton.merged_skeleton_mean_map,
                   # affine=img.affine).to_filename('skeleton_mean.nii.gz')
    # nb.Nifti1Image(mergedSkeleton.merged_skeleton_mean_map,
                   # affine=img.affine).to_filename('skeleton_std.nii.gz')
    nb.Nifti1Image(
        mergedSkeleton.skeleton_alteration_map,
        affine=img.affine).to_filename('skeleton_enigma_diff_map.nii.gz')

    # plot average map through `get_figure_enigma` function
    mergedSkeleton.corrp_data = mergedSkeleton.merged_skeleton_mean_map
    CorrpMap.get_figure_enigma(mergedSkeleton)
    # dark figure background
    plt.style.use('dark_background')
    # title
    mergedSkeleton.fig.suptitle(
        f'All skeleton average map\n{corrpMap.merged_4d_file}',
        y=0.95,
        fontsize=20)
    out_image_loc = re.sub('.nii.gz', '_average.png',
                           str(corrpMap.merged_4d_file))
    print(out_image_loc)
    mergedSkeleton.fig.savefig(out_image_loc, dpi=200)

    # plot std map through `get_figure_enigma` function
    mergedSkeleton.corrp_data = mergedSkeleton.merged_skeleton_std_map
    mergedSkeleton.main_data_vmax = 'free'
    CorrpMap.get_figure_enigma(mergedSkeleton)
    # dark figure background
    plt.style.use('dark_background')
    # title
    mergedSkeleton.fig.suptitle(
        f'All skeleton standard deviation map\n{corrpMap.merged_4d_file}',
        y=0.95,
        fontsize=20)
    out_image_loc = re.sub('.nii.gz', '_std.png',
                           str(corrpMap.merged_4d_file))
    print(out_image_loc)
    mergedSkeleton.fig.savefig(out_image_loc, dpi=200)

    # plot binnarized sum map through `get_figure_enigma` function
    mergedSkeleton.corrp_data = mergedSkeleton.merged_skeleton_data_bin_sum
    mergedSkeleton.main_data_vmax = 'free'
    CorrpMap.get_figure_enigma(mergedSkeleton)
    # dark figure background
    plt.style.use('dark_background')
    # title
    mergedSkeleton.fig.suptitle(
        f'Sum of binarized skeleton maps\n{corrpMap.merged_4d_file}',
        y=0.95,
        fontsize=20)
    out_image_loc = re.sub('.nii.gz', '_bin_sum.png',
                           str(corrpMap.merged_4d_file))
    print(out_image_loc)
    mergedSkeleton.fig.savefig(out_image_loc, dpi=200)

    # plot diff map between the binary sum map vs ENIGMA template
    # through `get_figure_enigma` function

    # enlarge the alteration map
    mergedSkeleton.skeleton_alteration_map = ndimage.binary_dilation(
            mergedSkeleton.skeleton_alteration_map,
            iterations=7).astype(mergedSkeleton.skeleton_alteration_map.dtype)

    mergedSkeleton.corrp_data = mergedSkeleton.skeleton_alteration_map
    mergedSkeleton.main_data_vmax = 'free'
    CorrpMap.get_figure_enigma(mergedSkeleton)
    # dark figure background
    plt.style.use('dark_background')
    # title
    mergedSkeleton.fig.suptitle(
        f'Difference betwenn the sum of binarized skeleton maps\n'
        f'and ENIGMA template\n{corrpMap.merged_4d_file}',
        y=0.95,
        fontsize=20)
    out_image_loc = re.sub('.nii.gz', '_bin_sum_diff_to_enigma.png',
                           str(corrpMap.merged_4d_file))
    print(out_image_loc)
    mergedSkeleton.fig.savefig(out_image_loc, dpi=200)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        randomise_summary.py --dir /example/randomise/output/dir/
        ''', epilog="Kevin Cho Thursday, August 22, 2019")

    argparser.add_argument("--directory", "-d",
                           type=str,
                           help='Specify randomise out dir. This this option \
                                 is given, design.mat and design.con within \
                                 the directory are read by default.',
                           default=os.getcwd())

    argparser.add_argument("--input", "-i",
                           type=str,
                           nargs='+',
                           help='Specify randomise out corrp files. If this \
                           option is given, --directory input is ignored')

    argparser.add_argument("--threshold", "-t",
                           type=float,
                           help='Threshold for the significance',
                           default=0.95)

    argparser.add_argument("--contrast", "-c",
                           type=str,
                           help='Contrast file used for the randomise.')

    argparser.add_argument("--matrix", "-m",
                           type=str,
                           help='Matrix file used for the randomise')

    argparser.add_argument("--template", "-template",
                           type=str,
                           default='enigma',
                           help='FA template used (or created) in TBSS')

    argparser.add_argument("--subject_values", "-s",
                           action='store_true',
                           help='Print average in the significant cluster for \
                           all subjects')

    argparser.add_argument("--sig_only", "-so",
                           action='store_true',
                           help='Print only the significant statistics')

    argparser.add_argument("--f_only", "-fo",
                           action='store_true',
                           help='Print only the output from f-test')

    argparser.add_argument("--merged_img_dir", "-merged_image_d",
                           type=str,
                           help='Directory that contains merged files')

    argparser.add_argument("--merged_image", "-merged_image",
                           type=str,
                           help='Directory that contains merged files')

    argparser.add_argument("--atlasquery", "-a",
                           action='store_true',
                           help='Run atlas query on significant corrp files')

    argparser.add_argument("--figure", "-f",
                           action='store_true',
                           help='Create figures')

    argparser.add_argument("--tbss_fill", "-ff",
                           action='store_true',
                           help='Create figures with tbss_fill outputs')

    argparser.add_argument("--skeleton_summary", "-ss",
                           action='store_true',
                           help='Create summary from all skeleton and also '
                                'figures from merged_skeleton_images')

    args = argparser.parse_args()

    # if separate corrp image is given
    if args.input:
        corrpMaps = [Path(x) for x in args.input]
        corrp_map_classes = [CorrpMap(x, args.threshold) for x in corrpMaps]
        if args.matrix and args.contrast:
            map(lambda x: setattr(x, matrix_file, args.matrix),
                corrp_map_classes)
            map(lambda x: setattr(x, 'contrast_file', args.contrast),
                corrp_map_classes)
            map(lambda x: x.get_matrix_info(), corrp_map_classes)
            map(lambda x: x.get_contrast_info(), corrp_map_classes)
            print('a=>ah')
            map(lambda x: x.get_contrast_info_english(), corrp_map_classes)
            # corrp_map_classes[0].print_matrix_info()
            map(lambda x: x.update_with_contrast(), corrp_map_classes)
        if args.contrast:
            map(lambda x: setattr(x, contrast_file, args.contrast), 
                corrp_map_classes)
            map(lambda x: x.get_contrast_info(), corrp_map_classes)
            map(lambda x: x.get_contrast_info_english(), corrp_map_classes)
            map(lambda x: x.update_with_contrast(), corrp_map_classes)

    # or if randomise image is given
    else:
        if args.contrast and args.matrix:
            randomiseRun = RandomiseRun(args.directory,
                                        matrix_file=args.matrix,
                                        contrast_file=args.contrast)
        elif args.contrast:
            randomiseRun = RandomiseRun(args.directory, args.contrast)
        elif args.matrix:
            randomiseRun = RandomiseRun(args.directory,
                                        matrix_file=args.matrix)
        else:
            randomiseRun = RandomiseRun(args.directory)

        randomiseRun.get_matrix_info()
        randomiseRun.print_matrix_info()

        randomiseRun.get_contrast_info()
        randomiseRun.get_contrast_info_english()

        if args.f_only:
            randomiseRun.get_corrp_files_glob_string('*corrp_f*.nii.gz')
        else:
            randomiseRun.get_corrp_files()

        corrpMaps = randomiseRun.corrp_ps
        corrp_map_classes = [CorrpMap(x, args.threshold) for x in corrpMaps]

        # TODO : WHY DOES NOT MAP WORK in updating 'df' attribute within a class
        #map(lambda x: setattr(x, df, x.update_with_contrast()), corrp_map_classes)
        for corrpMap in corrp_map_classes:
            corrpMap.contrast_array = randomiseRun.contrast_array
            corrpMap.contrast_lines = randomiseRun.contrast_lines
            corrpMap.update_with_contrast()

    # get merged image files
    if not args.merged_img_dir:
        args.merged_img_dir = args.directory

    # if subject_values option is given
    if args.subject_values:
        print_head('Values extracted for each subject')
        values_df = pd.DataFrame()
        for corrpMap in corrp_map_classes:
            if corrpMap.significant:
                print('-'*80)
                print(corrpMap.name)
                print(corrpMap.modality)
                try:
                    # find merged_4d_file
                    merged_4d_file = list(Path(args.merged_img_dir).glob(
                        f'*all*_{corrpMap.modality}[_.]*nii.gz'))[0]
                except:
                    # print("missing all merged file")
                    sys.exit('missing all merged file')
                    # questions = [
                            # inquirer.List(
                                # 'merged 4d file',
                                # message="Merged 4d file",
                                # choices=[],
                                # )),
                            # ]
                    # merged_4d_file = inquirer.prompt(questions)
                corrpMap.merged_4d_file = merged_4d_file
                corrpMap.update_with_4d_data()
                values_df = pd.concat([values_df,
                                       corrpMap.cluster_averages_df],
                                      axis=1)

        # if any of corrp map had significant voxels
        try:
            values_df = pd.concat([values_df,
                                   randomiseRun.matrix_df],
                                  axis=1)
            values_df.to_csv(
                f'{randomiseRun.location}/values_extracted_for_all_subjects.csv'
            )
            print(f'{randomiseRun.location}/'
                  'values_extracted_for_all_subjects.csv is created.')
        # if none of corrp map had significant voxels
        except:
            values_df.to_csv(
                f'{randomiseRun.location}/values_extracted_for_all_subjects.csv'
            )
            print(f'{randomiseRun.location}/'
                  'values_extracted_for_all_subjects.csv is created.')

        values_df.index = [f'subject {x+1}' for x in values_df.index]
        print_df(values_df)

    # printing result summaryt
    df = pd.concat([x.df for x in corrp_map_classes], sort=False)
    df = df.sort_values('file name')
    print_head('Result summary')

    if args.sig_only:
        print_head('Only showing significant maps')
        df_sig = df.groupby('Significance').get_group(True)
        print_df(df_sig.set_index(df_sig.columns[0]))
    else:
        print_df(df.set_index(df.columns[0]))

    # If atlas query option is on
    if args.atlasquery:
        print_head('Atlas query of the significant cluster')
        for corrpMap in corrp_map_classes:
            if corrpMap.significant:
                corrpMap.get_atlas_query()
                print_df(corrpMap.df_query)

    # If figure option is on
    if args.figure or args.tbss_fill:
        print_head('Saving figures')
        for corrpMap in corrp_map_classes:
            if corrpMap.significant is True:
                # tbss_fill if tbss_fill=True
                if args.tbss_fill:
                    print_head(f'Estimating tbss_fill for {corrpMap.location}')

                    # run tbss_fill
                    tbss_fill_out = tempfile.NamedTemporaryFile(suffix='.nii.gz')
                    corrpMap.tbss_fill(tbss_fill_out.name)
                    corrpMap.corrp_data_filled = nb.load(tbss_fill_out.name).get_data()

                if args.template == 'enigma':
                    corrpMap.get_figure_enigma()
                else:
                    corrpMap.get_figure_enigma(mean_fa=args.template)

                # dark figure background
                plt.style.use('dark_background')

                # title
                try:
                    corrpMap.fig.suptitle(
                        f'{corrpMap.modality} {corrpMap.contrast_text}\n'
                        f'{corrpMap.location}',
                        y=0.95,
                        fontsize=20)
                except:
                    corrpMap.fig.suptitle(
                        f'{corrpMap.modality}\n'
                        f'{corrpMap.location}',
                        y=0.95,
                        fontsize=20)

                out_image_loc = re.sub('.nii.gz', '.png',
                                       str(corrpMap.location))
                print(out_image_loc)
                corrpMap.fig.savefig(out_image_loc, dpi=200)
                #corrpMap.fig.savefig('/PHShome/kc244/out_image_loc.png', dpi=100)

    # skeleton summary parts
    if args.skeleton_summary:
        for corrpMap in corrp_map_classes:
            if corrpMap.significant is True:
                if hasattr(corrpMap, 'merged_4d_file'):
                    print_head("Summarizing merged 4d file:"
                               f"{corrpMap.merged_4d_file}")
                    skeleton_summary(corrpMap)
