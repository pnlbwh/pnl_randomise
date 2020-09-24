#!/data/pnl/kcho/anaconda3/bin/python

print('Importing modules')
from pathlib import Path
import tempfile

# figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage

# Imaging
import argparse
import nibabel as nb
import pandas as pd
import re
import numpy as np
from os import environ
import os

# utils
from skeleton_summary import MergedSkeleton, SkeletonDir, SkeletonDirSig

from pnl_randomise_utils import get_nifti_data, get_nifti_img_data
from pnl_randomise_utils import print_head, print_warning, print_df
from pnl_randomise_utils import search_and_select_one

# figures
import sys
sys.path.append('/home/yoobin_kwak/software/nifti-snapshot')
import nifti_snapshot.nifti_snapshot as nifti_snapshot

print('Importing modules complete')

mpl.use('Agg')
pd.set_option('mode.chained_assignment', None)

from randomise_summary_web import create_html
from typing import Union, List, Tuple

'''
TODO:
    - Test
    - write to do again
    - work with get_figure
    - left and right hemispher mask to consistent = 1 rather than 0
    - link nifti_snapshots to it
    - why import take so long?
    - Save summary outputs in pdf, csv or excel?
    - TODO add "interaction" information
    - design mat and design con search function
'''

def get_corrp_map_locs(args: object) -> List[object]:
    '''Return corrp map paths from args'''
    if args.input: # Get information from individual corrp files
        corrp_map_locs = args.input

    else:
        # If args.input is not given, get a list of corrp files from the given
        # randomise directory
        # this part reads information from design matrix and contrast
        randomiseRun = RandomiseRun(args.directory,
                                    matrix_file=args.matrix,
                                    contrast_file=args.contrast)

        # load list of corrp files
        if args.f_only:
            randomiseRun.get_corrp_files_glob_string('*corrp_f*.nii.gz')
        else:
            randomiseRun.get_corrp_files()
        corrp_map_locs = randomiseRun.corrp_ps

    return corrp_map_locs


def get_corrp_map_classes(corrp_map_locs: List, 
                          args: object) -> List[object]:
    '''Return corrpMaps from the corrp_map_locs and additional args info'''
    
    # get corrpMap information
    print_head('Summarizing information for files below')

    corrp_map_classes = []
    # set caselist as empty string if the caselist is not given
    if args.caselist:
        caselist = args.caselist
    else:
        caselist = ''

    for corrp_map_loc in corrp_map_locs:
        print(f'\t{corrp_map_loc}')
        if args.merged_img_dir:
            corrpMap = CorrpMap(corrp_map_loc,
                                threshold=args.threshold,
                                contrast_file=args.contrast,
                                matrix_file=args.matrix,
                                merged_img_dir=args.merged_img_dir,
                                template=args.template,
                                caselist=caselist,
                                group_labels=args.grouplabels)
        else:
            corrpMap = CorrpMap(corrp_map_loc,
                                threshold=args.threshold,
                                contrast_file=args.contrast,
                                matrix_file=args.matrix,
                                template=args.template,
                                caselist=caselist,
                                group_labels=args.grouplabels)

        corrp_map_classes.append(corrpMap)

    # if no corrpMap is defined
    try:
        corrpMap
    except NameError:
        sys.exit('Please check there is corrp file')

    return corrp_map_classes


class RandomiseRun(object):
    """Randomise output class

    This class is used to contain information about FSL randomise run,
    from a randomise stats output directory. The input directory should
    have following files inside it.

    1. randomise output statistic files. (one more more)
        - `*${modality}*corrp*.nii.gz`
        - make sure the modality is included in the stat file name.
    2. merged skeleton file for all subjects.
        - `all_${modality}*_skeleton.nii.gz`
        - make sure the modality is included in the merged skeleton file name.
    3. design matrix and design contrast files used for the randomise.
        - It will be most handy to have them named `design.con` and
          `design.mat`

    Key arguments:
        location: str or Path object of a randomise output location.
                  Preferably with a 'design.con' and 'design.mat' in the same
                  directory.
        contrast_file: str, randomise contrast file.
                       default='design.con'
        matrix_file: str, randomise matrix file.
                     default='design.mat'
    """
    def __init__(self,
                 location: str = '.',
                 contrast_file: Union[bool, str] = False,
                 matrix_file: Union[bool, str] = False):
        # define inputs
        self.location = Path(location)

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
        self.contrast_df = pd.DataFrame(self.contrast_array)
        self.contrast_df.columns = [f'col {int(x)+1}' for x in
                                    self.contrast_df.columns]

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
                        if self.group_labels:
                            text = f'{self.group_labels[pos_col_num]} > {self.group_labels[neg_col_num]}'
                        else:
                            text = f'Group {pos_col_num+1} > Group {neg_col_num+1}'
                    else:
                        if self.group_labels:
                            text = f'{self.group_labels[neg_col_num]} < {self.group_labels[pos_col_num]}'
                        else:
                            text = f'Group {neg_col_num+1} < Group {pos_col_num+1}'
                self.contrast_lines.append(text)

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

        # TODO add interaction information
        # if group column is zero
        # 0   0   1    -1
        try:
            if np.equal(np.array(self.contrast_array), 
                        np.array([[0, 0, 1, -1], [0, 0, -1, 1]])).all():
                self.contrast_lines = ['Positive Interaction', 
                                       'Negative Interaction']
        except ValueError:
            pass


    def get_matrix_info(self):
        """Read design matrix file into a numpy array and summarize

        matrix_header: headers in the matrix file
        matrix_array: numpy array matrix part of the matrix file
        matrix_df: pandas dataframe of matrix array

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
        self.matrix_df.columns = [f'col {x+1}' for x in
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

        if 'col 1' not in min_0_max_1_col:
            print("Contrast file does not have 1s in the first column "
                  "- likely be correlation or interaction")
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
                # self.matrix_info.loc['column info', col] = f"Group {group_num}"
                if self.group_labels != False:
                    self.matrix_info.loc['column info', col] = \
                        self.group_labels[group_num-1]
                else:
                    self.matrix_info.loc['column info', col] = \
                        f"Group {group_num}"

                # count of each unique value as an extra row
                self.matrix_info.loc['count', col] = \
                    (self.matrix_df[col] == 1).sum()

            # wide to long
            df_tmp = self.matrix_df.copy()
            # print_df(df_tmp)
            for num, row in df_tmp.iterrows():
                for group_num, group_col in enumerate(self.group_cols, 1):
                    if row[group_col] == 1:
                        if self.group_labels != False:
                            df_tmp.loc[num, 'group'] = f'{self.group_labels[group_num-1]}'
                        else:
                            df_tmp.loc[num, 'group'] = f'Group {group_num}'
            # print_df(df_tmp)
                
            # # non-group column
            self.covar_info_dict = {}
            for col in self.matrix_info.columns:
                if col not in self.group_cols:
                    unique_values = self.matrix_df[col].unique()
                    if len(unique_values) < 5:
                        count_df_tmp = df_tmp.groupby(['group', col]).count()
                        count_df_tmp = count_df_tmp[['col 1']]
                        count_df_tmp.columns = ['Count']
                        self.covar_info_dict[col] = \
                            count_df_tmp.reset_index().set_index('group')
                    else:
                        self.covar_info_dict[col] = \
                            df_tmp.groupby('group').describe()[col]


    def get_corrp_files_glob_string(self, glob_string:str):
        """Find corrp files and return a list of Path objects"""
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
        self.corrp_ps = [str(x) for x in corrp_ps
                         if 'SEED' not in x.name and 'filled' not in x.name]

        if len(self.corrp_ps) == 0:
            print(f'There is no corrected p-maps in {self.location}')

    def print_matrix_info(self):
        print_head('Matrix summary')
        print(f'Contrast file : {self.contrast_file}')
        print(f'Matrix file : {self.matrix_file}')
        print()
        if hasattr(self, 'matrix_df'):
            print(f'total number of data point : {len(self.matrix_df)}')
        if hasattr(self, 'group_cols'):
            print(f'Group columns are : ' + ', '.join(self.group_cols))
        if hasattr(self, 'matrix_info'):
            print_df(self.matrix_info)


def print_and_return_corrpMaps_summary(
        corrp_map_classes: List[object],
        sig_only: bool = False) -> pd.DataFrame:
    """Print the information of each corrpMap in the corrp_map_classes list"""
    print_head('Result summary')

    # concatenate corrpMap.df
    df = pd.concat([x.df for x in corrp_map_classes], sort=False)
    df = df.sort_values('file name')

    if sig_only:
        print_head('Only showing significant maps')
        try:
            df_sig = df.groupby('Significance').get_group(True)
            print_df(df_sig.set_index(df_sig.columns[0]))
        except KeyError:
            print('There is no significant corrp map. Please return withtout '
                  'the -so option')
    else:
        print_df(df.set_index(df.columns[0]))

    return df


class CorrpMap(RandomiseRun):
    """Multiple comparison corrected randomise output class

    This class is used to extract information from the corrected-p maps of
    randomise output. It also reads in design contrast and matrix files, as
    well as the merged skeleton file to summarize information from the
    randomise comparison.

    Current pipeline is optimized for TBSS pipeline that uses ENIGMA target
    skeleton.

    Key arguments:
        loc: str or Path object, location for the corrp map.
        threshold: float, fsl-style (1-p) threhold for significance.
                   default=0.95
    """
    def __init__(self, location:Union[str,Path], threshold=0.95,
                 contrast_file=False, matrix_file=False, **kwargs):
        #TODO add group labels
        #TODO add merged image location
        #TODO add randomise script location and contents
        self.location = Path(location)
        self.name = self.location.name
        self.threshold = threshold

        if contrast_file == False: # contrast not specified
            self.contrast_file = ''
        else:
            self.contrast_file = contrast_file

        if matrix_file == False: # matrix file not specified
            self.matrix_file = ''
        else:
            self.matrix_file = matrix_file

        # group labels
        if 'group_labels' in kwargs:
            self.group_labels = kwargs.get('group_labels')
        else:
            self.group_labels = False
        # if caselist is given
        # in `get_corrp_map_classes`, '' is given as the caselist
        # when there is no caelist is given to the randomise_summary.py
        if 'caselist' in kwargs:
            caselist = kwargs.get('caselist')
            if Path(caselist).is_file():
                with open(caselist, 'r') as f:
                    self.caselist = [x.strip() for x in f.readlines()]

        if not Path(self.contrast_file).is_file():
            self.contrast_file = search_and_select_one(
                    'contrast_file',
                    self.location.parent,
                    ['*.con', 'contrast*'], depth=0)

        if not Path(self.matrix_file).is_file():
            self.matrix_file = search_and_select_one(
                    'matrix_file',
                    self.location.parent,
                    ['*.mat', 'matrix*'], depth=0)

        # Modality
        # modality must be included in its name
        self.modality_full_list = ['FW', 'FA', 'FAt', 'FAc', 'FAk',
                                   'iFW',
                                   'MK', 'MKc', 'MKk',
                                   'MD', 'MDt',
                                   'RD', 'RDt',
                                   'AD', 'ADt']
        try:
            self.modality = re.search(
                '.*_(' + '|'.join(self.modality_full_list) + ')_',
                self.location.name).group(1)
        except AttributeError:
            print_head(f'No modality is detected in the file: {self.name}\n'
                       'Please add modality in the file name')
            self.modality = 'unknown'

        # Merged skeleton file
        # find merged skeleton file
        merged_skel_pattern = [f'*all*_{self.modality}[_.]*skel*nii.gz',
                               f'*{self.modality}*merged*.nii.gz']

        if 'merged_img_dir' in kwargs:
            self.merged_4d_file = search_and_select_one(
                'merged_skeleton',
                kwargs.get('merged_img_dir'),
                merged_skel_pattern, depth=0)
        else:
            self.merged_4d_file = search_and_select_one(
                'merged_skeleton',
                self.location.parent,
                merged_skel_pattern, depth=0)

        # information from the file name
        self.test_kind = re.search(r'(\w)stat\d+.nii.gz', self.name).group(1)
        self.stat_num = re.search(r'(\d+).nii.gz', self.name).group(1)

        # Below variables are to estimate number of significant voxels in each
        # hemisphere

        # checking significance
        self.check_significance()

        # template settings: If not specified use ENIGMA
        if 'template' in kwargs:
            self.template = kwargs.get('template')
        else:
            self.template = 'enigma'
        self.template_settings()

        if self.significant:
            # if significant read in skeleton mask
            #TODO
            # enigma settings
            skel_img, self.skel_mask_data = get_nifti_img_data(self.skel_mask_loc)
            self.get_significant_info()
            self.get_significant_overlap()

            # uncache the skeleton data matrix
            skel_img.uncache()

        # summary in pandas DataFrame
        self.make_df()

        # if matrix or contrast file is given
        if self.matrix_file != 'missing':
            self.get_matrix_info()

        if self.contrast_file != 'missing':
            self.get_contrast_info()
            self.get_contrast_info_english()
            self.update_with_contrast()

    def template_settings(self):
        """Set TBSS template settings"""

        # if given TBSS template is enigma
        if self.template == 'enigma':
            self.fsl_dir = Path(environ['FSLDIR'])
            self.fsl_data_dir = self.fsl_dir / 'data'

            self.enigma_dir = Path(
                '/data/pnl/soft/pnlpipe3/tbss/data/enigmaDTI')
            self.enigma_table = self.enigma_dir / 'ENIGMA_look_up_table.txt'
            # self.fa_bg_loc = Path(self.template).absolute() / 'mean_FA.nii.gz'
            self.fa_bg_loc = self.enigma_dir / 'ENIGMA_DTI_FA.nii.gz'
            self.mean_fa = self.location.parent / 'mean_FA.nii.gz'
            self.skel_mask_loc = self.enigma_dir / \
                'ENIGMA_DTI_FA_skeleton_mask.nii.gz'

        # if given TBSS template is a string of the FA template
        # eg) mean_FA.nii.gz
        else:
            self.fa_bg_loc = Path(self.template).absolute()
            self.skel_mask_loc = re.sub('.nii.gz',
                                        '_skeleton_mask.nii.gz',
                                        str(self.fa_bg_loc))

        # # nibabel img
        # self.skel_mask_img = nb.load(str(self.skel_mask_loc))
        # self.fa_bg_img = nb.load(str(self.fa_bg_loc))

    def check_significance(self):
        """Any voxels with greater value than self.threshold

        The nifti file in the `self.location` is read and checked for
            self.data_shape:  shape
            self.voxel_max_p:  maximum intensity value
            self.significant:  any voxels greater than `self.threshold`
            self.corrp_data: and returns array data (only if there are any
                             voxels greater than `self.threshold`)
        """

        # read corrp images
        img, data = get_nifti_img_data(self.location)

        # add data resolution attribute
        self.data_shape = data.shape

        # max p-value
        self.voxel_max_p = data.max()

        # # Discrepancy between numpy and FSL
        # if len(data[(data < 0.95) & (data >= 0.9495)]) != 0:
            # self.threshold = self.threshold - 0.00001
            # print('There are voxels with p value between 0.9495 and 0.05. '
                  # 'These numbers are rounded up in FSL to 0.95. Threfore '
                  # 'to match to the FSL outputs, changing the threshold to '
                  # '(threshold - 0.00001)')

        # any voxels significant?
        if (data >= self.threshold).any():
            self.significant = True
            self.corrp_data = data
        else:
            self.significant = False

    def get_significant_info(self):
        """Get information of significant voxels"""

        # total number of voxels in the skeleton
        # there could be voxels with 0 in corrp map
        # p value of 1 --> represented as 0
        self.vox_num_total = np.count_nonzero(self.skel_mask_data)

        # number of significant voxels: greater or equal to 0.95 by default
        self.significant_voxel_num = \
            np.count_nonzero(self.corrp_data > self.threshold)

        # number of significant voxels / number of all voxels
        # (self.significant_voxel_num / np.count_nonzero(self.corrp_data)) \
        self.significant_voxel_percentage = \
            (self.significant_voxel_num / self.vox_num_total) \
            * 100

        # summary of significant voxels
        sig_vox_array = self.corrp_data[self.corrp_data > self.threshold]
        self.significant_voxel_mean = 1 - sig_vox_array.mean()
        self.significant_voxel_std = sig_vox_array.std()
        self.significant_voxel_max = 1 - sig_vox_array.max()

        # test print
        # for var in ['vox_num_total', 'significant_voxel_num']:
            # print(f'{var} {getattr(self, var)}')

    def get_significant_overlap(self):
        """Get overlap information in each hemisphere

        Works for ENIGMA template randomise outputs.
        - x=90 as the cut off value for the left and right hemisphere
        """
        right_mask = self.skel_mask_data.copy()
        right_mask[90:, :, :] = 0
        left_mask = self.skel_mask_data.copy()
        left_mask[:90, :, :] = 0

        try:
            for side, side_mask in zip(['left', 'right'],
                                       [left_mask, right_mask]):
                # get overlaps with each hemisphere
                side_skeleton_array = self.corrp_data * side_mask

                # get number of significant voxels
                significant_voxel_side_num = \
                    np.sum(side_skeleton_array > self.threshold)
                setattr(self, f'significant_voxel_{side}_num',
                        significant_voxel_side_num)

                # get percentage significant voxels in each hemisphere
                if np.count_nonzero(side_skeleton_array) == 0:
                    setattr(self, f'significant_voxel_{side}_percent', 0)
                else:
                    setattr(self, f'significant_voxel_{side}_percent',
                            (significant_voxel_side_num /
                                np.count_nonzero(self.skel_mask_data)) * 100)
        except:
            print('** This study has a specific template. The number of '
                  'significant voxels in the left and right hemisphere '
                  'will not be estimated')
            setattr(self, f'significant_voxel_{side}_percent', 'unknown')

    def make_df(self):
        """Make summary pandas df of each corrp maps"""
        if self.significant:
            self.df = pd.DataFrame({
                'file name': [self.name],
                'Test': self.test_kind,
                'Modality': self.modality,
                'Significance': self.significant,
                'Sig Max': self.voxel_max_p,
                'Sig Mean': self.significant_voxel_mean,
                'Sig Std': self.significant_voxel_std,
                '% significant voxels': self.significant_voxel_percentage,
                # '% left': self.significant_voxel_left_percent,
                # '% right': self.significant_voxel_right_percent
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
                'Significance': self.significant,
                'Sig Max': self.voxel_max_p,
            })

        self.file_df = pd.DataFrame({
            'corrp map': [self.location],
            'FA template': self.fa_bg_loc,
            'merged skeleton': self.merged_4d_file})

        # print_df(self.file_df)
        # other information

    def update_with_contrast(self):
        '''Update CorrpMap class when there the contrast file is available
        (when self.contrast_array is available)

        Requires:
            self.contrast_array : numpy array, of the design contrast file.
                                  Created by loading only the contrast lines
                                  text using np.loadtxt.
        '''
        # Contrast map
        line_num = int(self.stat_num)-1

        self.contrast_line = self.contrast_array[line_num, :]
        self.contrast_text = self.contrast_lines[line_num]

        # Change the numpy array to string
        self.df['contrast'] = np.array2string(self.contrast_line,
                                              precision=2)[1:-1]
        try:
            self.df['contrast_text'] = self.contrast_lines[line_num]
        except:
            self.df['contrast_text'] = '-'

        # if f-test
        if self.test_kind == 'f':
            self.df['contrast'] = 'f-test'
            # try:
                # with open(str(self.location.parent / 'design.fts'), 'r') as f:
                    # lines = f.readlines()
                    # design_fts_line = [x for x in lines
                                       # if re.search(r'^\d', x)][0]
                # self.df['contrast'] = '* ' + design_fts_line
            # except:

        # Reorder self.df to have file name and the contrast on the left
        self.df = self.df[['file name', 'contrast', 'contrast_text'] +
                          [x for x in self.df.columns if x not in
                              ['file name',
                               'contrast',
                               'contrast_text']]]
        return self.df

    def update_with_4d_data(self):
        """Get mean values for skeleton files in the significant voxels

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
        merged_4d_data = get_nifti_data(self.merged_4d_file)

        # get a map with significant voxels
        significant_cluster_data = np.where(
            self.corrp_data >= self.threshold, 1, 0)

        self.cluster_averages = {}
        # Get average of values in the `significant_cluster_data` map
        # for each skeleton volume
        for vol_num in np.arange(merged_4d_data.shape[3]):
            vol_data = merged_4d_data[:, :, :, vol_num]
            average = vol_data[significant_cluster_data == 1].mean()
            self.cluster_averages[vol_num] = average

        self.cluster_averages_df = pd.DataFrame.from_dict(
            self.cluster_averages,
            orient='index',
            columns=[f'{self.modality} values in the significant '
                     f'cluster {self.name}']
        )

    def get_atlas_query(self):
        """Return pandas dataframe summary of atlas_query outputs"""
        # threshold corrp file according to the threshold
        thresholded_map = tempfile.NamedTemporaryFile(suffix='tmp.nii.gz')
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
            lambda x: re.sub(r'\(.*\)', '', x))

        # Adding 'Side' column
        df_query['Side'] = df_query['Structure'].str.extract('(L|R)$')

        # Remove side information from Structure column
        df_query['Structure'] = df_query['Structure'].str.replace(
            '(L|R)$', '').str.strip()
        df_query.loc[df_query['Side'].isnull(), 'Side'] = 'M'

        # Side column to wide format
        self.df_query = pd.pivot_table(
            index=['file_name', 'Structure', 'atlas'],
            columns='Side',
            values='Percentage',
            data=df_query).reset_index()

        # TODO: change here later
        # self.df_query = self.df_query.groupby('atlas').get_group('Labels')
        self.df_query = self.df_query.sort_values('atlas')

    def get_figure(self, **kwargs):
        """Get corrpMap figure"""
        self.cbar_title = f'{self.modality} {self.contrast_text}'

        if hasattr(self, 'tbss_fill_out'): # for tbss fill option
            self.out_image_loc = re.sub(
                '.nii.gz', '.png', str(self.tbss_fill_out))
            self.title = f'{self.modality} {self.contrast_text}\n' \
                         f'{self.tbss_fill_out}'

            if 'figure_same_slice' in kwargs:
                same_slice = kwargs.get('figure_same_slice')

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

        # matrix_df[corrpMap.group_cols].astype(
            # 'int').to_string(header=False, index=False).split('\n')
            # fa_bg=self.fa_bg_loc,
            # skeleton_bg=self.template,
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


def skeleton_summary(corrpMap, warp_dir=False, caselist=False):
    """ Make summary from corrpMap, using its merged_skeleton"""
    mergedSkeleton = MergedSkeleton(
            str(corrpMap.merged_4d_file),
            corrpMap.skel_mask_loc)

    mergedSkeleton.fa_bg_loc = corrpMap.fa_bg_loc
    mergedSkeleton.template = corrpMap.template

    # mergedSkeleton.out_image_loc = 'tmp.png'
    # mergedSkeleton.cbar_title = 'tmp'
    # mergedSkeleton.title = 'tmp'

    mergedSkeleton.skeleton_level_summary()
    mergedSkeleton.subject_level_summary()


    # set modality of the merged skeleton file
    mergedSkeleton.modality = corrpMap.modality

    # get a list of groups
    # TODO: here might be changed into actual group names
    group_list = corrpMap.matrix_df[corrpMap.group_cols].astype(
        'int').to_string(header=False, index=False).split('\n')

    # create a dataframe that has
    # - mean of values in the non zero skeleton for each subject
    # - standard deviation of values in the nonzero skeleton for each subject
    print('Creating figures to summarize information from the skeletons')
    if warp_dir and caselist:
        print(f'Using information in {warp_dir} to '
              'extract values in warped maps')

        mergedSkeleton.subject_level_summary_with_warp(warp_dir, caselist)
        mergedSkeleton.df = pd.DataFrame({
            'subject': corrpMap.matrix_df.index,
            'mean': mergedSkeleton.subject_nonzero_means,
            'skeleton_volume': mergedSkeleton.subject_nonzero_voxel_count,
            'skeleton non_zero_std': mergedSkeleton.subject_nonzero_stds,
            'zero_skeleton':mergedSkeleton.subject_zero_skeleton_values,
            'group': group_list
            })
        SkeletonDir.get_subject_zero_skeleton_figure(mergedSkeleton)
        out_image_loc = re.sub('.nii.gz',
                               '_skeleton_zero_mean_in_warp.png',
                               str(corrpMap.merged_4d_file))

        mergedSkeleton.g_skel_zero.savefig(
                out_image_loc,
                facecolor='white', dpi=200)
        plt.close()
        print('\t- Values in the warpped maps for the zero voxels '
              'in the skeleton')
    else:
        mergedSkeleton.df = pd.DataFrame({
            'subject': corrpMap.matrix_df.index,
            'mean': mergedSkeleton.subject_nonzero_means,
            'skeleton_volume': mergedSkeleton.subject_nonzero_voxel_count,
            'skeleton non_zero_std': mergedSkeleton.subject_nonzero_stds,
            'group': group_list
            })


    # Figure that shows
    # - skeleton group average as ahline
    # - skeleton subject average as scatter dots
    # - tests between subject averages between groups
    plt.style.use('default')
    SkeletonDir.get_group_figure(mergedSkeleton)
    out_image_loc = re.sub('.nii.gz',
                           '_skeleton_average_for_all_subjects.png',
                           str(corrpMap.merged_4d_file))
    mergedSkeleton.g.savefig(out_image_loc, facecolor='white', dpi=200)
    plt.close()
    print('\t- Average for the skeleton in each subjects')

    SkeletonDir.get_subject_skeleton_volume_figure(mergedSkeleton)
    out_image_loc = re.sub('.nii.gz',
                           '_skeleton_volume_for_all_subjects.png',
                           str(corrpMap.merged_4d_file))
    mergedSkeleton.g_skel_vol.savefig(out_image_loc, facecolor='white', dpi=200)
    plt.close()
    print('\t- Volume for the skeleton in each subjects')



    mergedSkeleton.data_shape = corrpMap.data_shape
    mergedSkeleton.threshold = 0.01

    # Figure that shows
    # - skeleton alteration map
    #   - 1 where all subject has the skeleton
    #   - 0 where not all subject has the skeleton
    # enlarge the alteration map
    mergedSkeleton.skeleton_alteration_map = ndimage.binary_dilation(
            mergedSkeleton.skeleton_alteration_map,
            iterations=7).astype(mergedSkeleton.skeleton_alteration_map.dtype)

    # mean of the skeleton across the subject
    # out_image_loc = re.sub(
        # '.nii.gz', f'_average_across_subjects.png',
        # str(corrpMap.merged_4d_file))
    # meanSkeletonFigure = nifti_snapshot.TbssFigure(
        # image_files=[mergedSkeleton.merged_skeleton_mean_map],
        # output_file=out_image_loc,
        # cmap_list=['autumn'],
        # cbar_titles=[
            # f'Average {mergedSkeleton.modality} map across all subjects'],
        # alpha_list=[1],
        # title=f'Average {mergedSkeleton.modality} map across all subjects')
    # meanSkeletonFigure.images_mask_out_the_skeleton()

    # plot average map through `get_figure_enigma` function
    # TODO SPLIT below back again
    for map_data, name_out_png, title, vmin, vmax in zip(
            [mergedSkeleton.merged_skeleton_mean_map,
             mergedSkeleton.merged_skeleton_std_map,
             mergedSkeleton.merged_skeleton_data_bin_sum,
             mergedSkeleton.skeleton_alteration_map],
            ['average', 'std', 'bin_sum', 'bin_sum_diff'],
            ['All skeleton average map',
             'All skeleton standard deviation map',
             'Sum of binarized skeleton maps for all subjects',
             'Highlighting variability among binarized skeleton maps'],
            [0, 0, 0, 0],
            ['free', 'free', 'free', 1]):
        # set data input in order to use CorrpMap.get_figure_enigma function
        mergedSkeleton.corrp_data = map_data
        mergedSkeleton.type = name_out_png
        plt.style.use('dark_background')

        # try:
        # print(name_out_png)
        mergedSkeleton.vmin = map_data[mergedSkeleton.mask_data].min()
        # print(f'vmin for merged skeleton: {mergedSkeleton.vmin}')
        # except:
            # pass
        mergedSkeleton.vmax = vmax
        out_image_loc = re.sub('.nii.gz', f'_{name_out_png}.png',
                               str(corrpMap.merged_4d_file))
        mergedSkeleton.out_image_loc = out_image_loc
        mergedSkeleton.cbar_title = title
        mergedSkeleton.title = title
        CorrpMap.out_image_loc = out_image_loc
        CorrpMap.title = title + f'\n{corrpMap.merged_4d_file}'
        CorrpMap.get_figure_enigma(mergedSkeleton)

        # # title
        # print('\t- ' + title)
        # mergedSkeleton.fig.suptitle(
            # title
            # y=0.95, fontsize=20)
        # mergedSkeleton.fig.savefig(out_image_loc, facecolor='black', dpi=200)
        plt.close()


def check_corrp_map_locations(corrp_map_classes):
    """ Make sure all corrpMap are in a same directory """
    corrpMap_locations = list(
        set([x.location.parent for x in corrp_map_classes]))
    if len(corrpMap_locations) != 1:
        print_warning(
            'Input Corrp Maps are located in different directories. This '
            'may lead to randomise_summary.py catching a wrong merged 4d '
            'data for data summary. Please consider running separate '
            'randomise_summary.py runs for each corrp map or moving them '
            'into a single directory before running randomise_summary.py'
            )
    else:
        pass

def run_randomise_summary_basics(
        args: object) -> Tuple[List[CorrpMap], pd.DataFrame]:
    '''Run the randomise output summary (ran from commandline)'''
    # Get a list of corrp map paths based on the arg parse inputs
    corrp_map_locs = get_corrp_map_locs(args)

    # Get a list of corrpMap objects ** this is where corrpMaps are formed
    corrp_map_classes = get_corrp_map_classes(corrp_map_locs, args)

    # Print a matrix information of the first corrpMap in the corrp_map_classes
    # assuming all of corrpMaps in the list have the same matrix and contrast
    print_head('Matrix information')
    corrp_map_classes[0].print_matrix_info()

    # TODO: copy this to html summary
    if args.print_cov_info and \
            hasattr(corrp_map_classes[0], 'covar_info_dict'):
        print_head('Covariate summary')
        for col, table in corrp_map_classes[0].covar_info_dict.items():
            print(col)
            print_df(table)

    # Print information of each corrpMap
    df = print_and_return_corrpMaps_summary(corrp_map_classes,
                                            sig_only=args.sig_only)

    return corrp_map_classes, df


def run_atlas_query(args: object, corrp_map_classes: List[CorrpMap]) -> None:
    '''Atlas query'''
    # if atlas query option is on
    if args.atlasquery:
        print_head('Atlas query of the significant cluster')
        print_head('(atlasquery from FSL)')
        for corrpMap in corrp_map_classes:
            if corrpMap.significant:
                corrpMap.get_atlas_query()
                print_df(corrpMap.df_query)


def create_figure(args: object, corrp_map_classes: List[CorrpMap]) -> None:
    # If figure option is on
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
                    corrpMap.get_figure()
                    plt.close()


def get_individual_summary(args: object, corrp_map_classes: List[CorrpMap]) -> None:
    '''Get individual summary'''

    # TODO : check the structure below
    # if merged image location is not given
    if not args.merged_img_dir:
        if args.subject_values or args.skeleton_summary:
            # make sure all the input corrp maps are in the same directory
            check_corrp_map_locations(corrp_map_classes)
            args.merged_img_dir = str(corrp_map_classes[0].location.parent)

    # if subject_values option is given
    if args.subject_values:
        print_head('Values extracted for each subject')
        values_df = pd.DataFrame()
        modal_ms_dict = {}
        all_modalities = [x.modality for x in corrp_map_classes
                          if x.significant]
        for corrpMap in corrp_map_classes:
            if corrpMap.significant:
                if corrpMap.modality in modal_ms_dict.keys():
                    mergedSkeleton = modal_ms_dict[corrpMap.modality]
                    mergedSkeleton.update_with_corrpMap(corrpMap)
                else:
                    mergedSkeleton = MergedSkeleton(
                        corrpMap.merged_4d_file,
                        corrpMap.skel_mask_loc)
                    mergedSkeleton.update_with_corrpMap(corrpMap)

                    # if there is a need to save mergedSkeleton
                    if len([x for x in all_modalities
                            if x == corrpMap.modality]) > 2:
                        modal_ms_dict[corrpMap.modality] = mergedSkeleton

                values_df = pd.concat(
                    [values_df, mergedSkeleton.cluster_averages_df],
                    axis=1)

                # cluster average figure
                SkeletonDirSig.get_group_figure(mergedSkeleton)
                mergedSkeleton.g.ax.set_title(
                    f'Average {corrpMap.modality} in the significant cluster '
                    'for all subjects\n'
                    f'({corrpMap.location} > {corrpMap.threshold})',
                    fontweight='bold')
                mergedSkeleton.g.ax.set_ylabel(
                    f'{corrpMap.modality} in the significant cluster')
                out_image_loc = re.sub(
                    '.nii.gz', '_sig_average_for_all_subjects.png',
                    str(corrpMap.location))
                mergedSkeleton.g.savefig(out_image_loc,
                                         facecolor='white', dpi=200)
                print(out_image_loc)
                plt.close()
                mergedSkeleton = ''

        # if any of corrp map had significant voxels
        out_csv_name = 'values_extracted_for_all_subjects.csv'

        out_csv = f'{corrpMap.location.parent}/{out_csv_name}'
        print('Average value for the significant cluster for each subject '
              f'will be saved in {out_csv}')

        try:
            values_df = pd.concat([values_df,
                                   corrpMap.matrix_df],
                                  axis=1)
            values_df.to_csv(out_csv)
            print(f'{out_csv} is created.')


        # if none of corrp map had significant voxels
        except:
            values_df.to_csv(out_csv)
            print(f'{out_csv} is created.')

        values_df.index = [f'subject {x+1}' for x in values_df.index]
        print_df(values_df)


def get_skeleton_summary(args: object,
        corrp_map_classes: List[CorrpMap]) -> None:
    '''skeleton summary'''
    # skeleton summary parts
    if args.skeleton_summary:
        print_head('Running skeleton summary')
        summarized_merged_maps = []
        for corrpMap in corrp_map_classes:
            if not hasattr(corrpMap, 'matrix_df'):
                print('Please provide correct design matrix. The file is '
                      'required to read in the group infromation.')
                pass

            elif corrpMap.modality == 'unknown':
                print(f'The modality for {corrpMap.location} is unknown to '
                      'the current version of randomise_summary. Please check '
                      'the modality is in the list below.')
                print('  ' + ' '.join(corrpMap.modality_full_list))

            elif corrpMap.merged_4d_file == 'missing':
                print(f'Merged 4d file for {corrpMap.location} is missing. '
                      f'Please check there are all_{corrpMap.modality}'
                      '_skeleton.nii.gz in the same directory.')

            elif hasattr(corrpMap, 'merged_4d_file') and \
                    corrpMap.merged_4d_file not in summarized_merged_maps and \
                    corrpMap.merged_4d_file != 'missing' and args.tbss_all_loc:
                print_head("Summarizing merged 4d file:"
                           f"{corrpMap.merged_4d_file}")
                warp_dir = str(Path(args.tbss_all_loc) / corrpMap.modality / 'warped')
                print(warp_dir)
                caselist = str(Path(args.tbss_all_loc) / 'log/caselist.txt')

                skeleton_summary(corrpMap, warp_dir=warp_dir, caselist=caselist)
                summarized_merged_maps.append(corrpMap.merged_4d_file)
                print()

            elif hasattr(corrpMap, 'merged_4d_file') and \
                    corrpMap.merged_4d_file not in summarized_merged_maps and \
                    corrpMap.merged_4d_file != 'missing':
                print_head("Summarizing merged 4d file:"
                           f"{corrpMap.merged_4d_file}")
                skeleton_summary(corrpMap)
                summarized_merged_maps.append(corrpMap.merged_4d_file)
                print()


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
        ''', epilog="Kevin Cho 21st December 2019")

    argparser.add_argument(
        "--directory", "-d",
        type=str,
        help='Specify randomise out directory.',
        default=os.getcwd())

    argparser.add_argument(
        "--input", "-i", type=str, nargs='+',
        help='Specify randomise out corrp files. If this option is given, '
             '--directory input is ignored')

    argparser.add_argument(
        "--threshold", "-t", type=float, default=0.95,
        help='Threshold for the significance')

    argparser.add_argument(
        "--contrast", "-c", type=str, default='design.con',
        help='Contrast file used for the randomise.')

    argparser.add_argument(
        "--matrix", "-m", type=str, default='design.mat',
        help='Matrix file used for the randomise')

    argparser.add_argument(
        "--template", "-template", type=str, default='enigma',
        help='FA template used (or created) in TBSS - eg) mean_FA.nii.gz')

    argparser.add_argument(
        "--subject_values", "-s", action='store_true',
        help='Print average in the significant cluster for all subjects')

    argparser.add_argument(
        "--print_cov_info", "-ci", action='store_true',
        help='Print covariate information for each group')

    argparser.add_argument(
        "--sig_only", "-so", action='store_true',
        help='Print only the significant statistics')

    argparser.add_argument(
        "--f_only", "-fo", action='store_true',
        help='Print only the output from f-test')

    argparser.add_argument(
        "--merged_img_dir", "-merged_image_d", type=str,
        help='Directory that contains merged files')

    argparser.add_argument(
        "--merged_image", "-merged_image", type=str,
        help='Directory that contains merged files')

    argparser.add_argument(
        "--atlasquery", "-a", action='store_true',
        help='Run atlas query on significant corrp files')

    argparser.add_argument(
        "--figure", "-f", action='store_true', help='Create figures')

    argparser.add_argument(
        "--tbss_fill", "-tf", action='store_true',
        help='Create figures with tbss_fill outputs')

    argparser.add_argument(
        "--figure_same_slice", "-fss", action='store_true',
        help='use with -f or -tf, to get consistent slices for all figure')

    argparser.add_argument(
        "--skeleton_summary", "-ss", action='store_true',
        help='Create summary from all skeleton and also figures from '
             'merged_skeleton_images')

    argparser.add_argument(
        "--caselist", "-caselist", type=str,
        help='caselist file used to run the tbss')

    argparser.add_argument(
        "--tbss_all_loc", "-tal", type=str, help='tbss_all output path')

    argparser.add_argument(
        "--grouplabels", "-gl", type=str, nargs='+', default=False,
        help='List of group names in the same order as the matrix')

    argparser.add_argument(
        "--html_summary", "-hs", action='store_true',
        help='Create web summary from the randomise outputs')

    args = argparser.parse_args()

    # get list of corrpMaps 
    corrp_map_classes, df = run_randomise_summary_basics(args)

    # extra
    run_atlas_query(args, corrp_map_classes)
    create_figure(args, corrp_map_classes)

    # individual subject summary
    get_individual_summary(args, corrp_map_classes)

    # skeleton summary
    get_skeleton_summary(args, corrp_map_classes)

    # html summary
    if args.html_summary:
        create_html(corrp_map_classes, df, args)
        print(f'HTML summary is saved in '
              f"{corrp_map_classes[0].location.parent / 'randomise_summary.html'}")
