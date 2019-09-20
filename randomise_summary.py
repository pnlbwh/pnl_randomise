#!/data/pnl/kcho/anaconda3/bin/python

from pathlib import Path
import tempfile

# Imaging
import nibabel as nb
import argparse
import pandas as pd
import re
import numpy as np
from os import environ
import os

# Pandas dataframe print
from io import StringIO
from tabulate import tabulate
pd.set_option('mode.chained_assignment', None)


'''
TODO:
    - Test
    - Save summary outputs in pdf, csv or excel?
    - TODO add interaction information (group info to the get_contrast_info_english)
    - Check whether StringIO.io is used
    - Documentation
    - Estimate how much voxels overlap with different atlases with varying 
      thresholds.
    - Move useful functions to kcho_utils.
    - Develop it further to utilze orig_dir input option
    - Save significant voxel coordinates
        - to extract mean values at these coordinates for all subjects
        - to add extract information, 
            - ie.number of voxels, average P-values etc.
    - Parallelize
'''

def print_df(df):
    """Print pandas dataframe using tabulate.

    Used to print outputs when the script is called from the shell
    Key arguments:
        df: pandas dataframe
    """
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print()

def print_head(heading):
    print()
    print('-'*80)
    print(f'* {heading}')
    print('-'*80)

class RandomiseRun:
    """Randomise output class
    
    Used to contain information about FSL randomise run.

    Key arguments:
    location -- str or Path object of a randomise output location. 
              Preferably with a 'design.con' and 'design.mat' inside.
              (default:'.')
    threshold -- float, 1-p value used to threhold for significance.
                 (default:0.95)
    contrast_file -- design contrast file used as the input for randomise.
                     (default:'design.con')
    matrix_file -- design matrix file used as the input for randomise.
                   (default:'design.mat')
    orig_dir -- location of origdir.


    TODO:
        set TBSS and skeleton directory default
    """
    def __init__(self, 
                 location='.', 
                 contrast_file='design.con', 
                 matrix_file='design.mat'):
        # define inputs
        self.location = Path(location)
        if matrix_file == 'design.mat':
            self.matrix_file = self.location / matrix_file
        else:
            self.matrix_file = Path(matrix_file)

        if contrast_file == 'design.con':
            self.contrast_file = self.location / contrast_file
        else:
            self.contrast_file = Path(contrast_file)

    def get_contrast_info(self):
        """Read design contrast file into a numpy array

        'contrast_array' : numpy array of the contrast file excluding the 
                           headers
        'contrast_lines' : attributes that states what each row in the 
                           contrast array represents

        TODO:
            - check whether the randomise run is a f-test
            - add a attribute that states the comparison in plain English for
            other contrasts
        """

        with open(self.contrast_file, 'r') as f:
            lines = f.readlines()
            headers = [x for x in lines \
                       if x.startswith('/')]

        last_header_line_number = lines.index(headers[-1]) + 1
        self.contrast_array = np.loadtxt(self.contrast_file, 
                                         skiprows=last_header_line_number)

    def get_contrast_info_english(self):
        # if all lines are group comparisons --> simple group comparisons
        # TODO select group columns
        #group_cols_array = np.array([int(x.split('_')[1]) for x \
                                         #in self.group_cols])
        #print(group_cols_array)
        if (self.contrast_array.sum(axis=1)==0).all() :
            #TODO : do below at the array level?
            df_tmp = pd.DataFrame(self.contrast_array)
            contrast_lines = []
            for contrast_num, row in df_tmp.iterrows():
                # name of the column with value of 1
                pos_col_num = row[row==1].index.values[0]
                neg_col_num = row[row==-1].index.values[0]

                # Change order of columns according to their column numbers
                if pos_col_num < neg_col_num:
                    text = f'Group {pos_col_num+1} > Group {neg_col_num+1}'
                else:
                    text = f'Group {neg_col_num+1} < Group {pos_col_num+1}'
                contrast_lines.append(text)
            self.contrast_lines = contrast_lines
        elif (np.absolute(self.contrast_array.sum(axis=1))==1).all():
            #TODO : do below at the array level?
            df_tmp = pd.DataFrame(self.contrast_array)
            contrast_lines = []
            for contrast_num, row in df_tmp.iterrows():
                # name of the column with value of 1
                col_num = row[row!=0].index.values[0]

                # Change order of columns according to their column numbers
                if row.loc[col_num] == 1:
                    text = f'Positively correlated with col {col_num+1}'
                else:
                    text = f'Negatively correlated with col {col_num+1}'
                contrast_lines.append(text)
            self.contrast_lines = contrast_lines
        #elif 
        #TODO add interaction information
        # if group column is zero
        #0   0   1    -1


    def get_matrix_info(self):
        """Read design matrix file into a numpy array and summarize
        
        attributes added
        'matrix_header' : headers in the matrix file 
        'matrix_array' : numpy array matrix part of the matrix file
        'matrix_df' : pandas dataframe of matrix array

        TODO:
            add function defining which column is group
        """
        with open(self.matrix_file, 'r') as f:
            lines = f.readlines()
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

        self.matrix_array = np.loadtxt(self.matrix_file, 
                                       skiprows=len(matrix_lines))

        # matrix array into pandas dataframe
        self.matrix_df = pd.DataFrame(self.matrix_array)
        # rename columns to have 'col ' in each
        self.matrix_df.columns = [f'col {x}' for x in \
                                        self.matrix_df.columns]
        # summarize matrix
        self.matrix_info = self.matrix_df.describe()
        self.matrix_info = self.matrix_info.loc[['mean', 'std', 'min', 'max'],:]
        self.matrix_info = self.matrix_info.round(decimals=2)

        # For each column of the matrix, add counts of unique values to 
        # self.matrix_info 
        for col in self.matrix_df.columns:
            # create a dataframe that contains unique values as the index 
            unique_values = self.matrix_df[col].value_counts().sort_index()
            # If there are less than 5 unique values for the column
            if len(unique_values) < 5:
                # unique values as an extra row
                self.matrix_info.loc['unique', col] = np.array2string(
                    unique_values.index)[1:-1]
                # count of each unique value as an extra row
                self.matrix_info.loc['count', col] = np.array2string(
                    unique_values.values)[1:-1]
            # If there are 5 or more unique values in the column, leave the
            else:
                # 'unique' and 'count' as 'continuous values'
                self.matrix_info.loc['unique', col] = 'continuous values'
                self.matrix_info.loc['count', col] = 'continuous values'


        # define which column represent group column
        # Among columns where their min value is 0 and max value is 1,
        min_0_max_1_col = [x for x in self.matrix_df.columns \
                           if self.matrix_df[x].isin([0,1]).all()]

        # if sum of each row equal to 1, these columns would highly likely be 
        # group columns
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
            self.matrix_info.loc['count', col] = (self.matrix_df[col]==1).sum()

    def get_corrp_files(self):
        """Find corrp files and return a list of Path objects 
        """
        corrp_ps = list(self.location.glob('*corrp*.nii.gz'))
        # remove corrp files that are produced in the parallel randomise
        self.corrp_ps = [str(x) for x in corrp_ps if not 'SEED' in x.name]

        if len(self.corrp_ps) == 0:
            print(f'There is no corrected p-maps in {rand_loc}')


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
                '.*(FW|FA|MD|RD|AD|MD|FAt|MDt|RDt|ADt|MDt)_',
                self.location.name).group(1)
        except:
            self.modality = ''
        self.threshold = threshold
        self.test_kind = re.search('(\w)stat\d+.nii.gz', self.name).group(1)
        self.stat_num = re.search('(\d+).nii.gz', self.name).group(1)

        # Below variables are to estimate number of significant voxels in each
        # hemisphere
        self.fsl_dir = Path(environ['FSLDIR'])
        self.fsl_data_dir = self.fsl_dir / 'data'
        self.HO_dir = self.fsl_data_dir / 'atlases' / 'HarvardOxford'
        self.HO_sub_thr0_1mm = self.HO_dir / \
                'HarvardOxford-sub-maxprob-thr0-1mm.nii.gz'

        self.check_significance()
        if self.significant:
            self.get_significant_info()
            self.get_significant_overlap_with_HO()
        self.make_df()


    def check_significance(self):
        """Check whether there is any significant voxel"""
        img = nb.load(str(self.location))
        data = img.get_data()
        #data = np.around(data, decimals=6)

        # max p-value
        self.voxel_max_p = data.max()

        # Discrepancy between numpy and FSL
        if len(data[(data < 0.95) & (data >= 0.9495)]) != 0:
            self.threshold = self.threshold - 0.00001
            print('There are voxels with p value between 0.9495 and 0.05. '\
                  'These numbers are rounded up in FSL to 0.95. Threfore '\
                  'to match to the FSL outputs, changing the threshold to '\
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
        HO_data = nb.load(str(self.HO_sub_thr0_1mm)).get_data()

        # Few voxels from ENIGMA template skeleton spreads into the area
        # defined as a gray matter by Harvard Oxford atlas
        left_mask_array = np.where((HO_data==1) + (HO_data==2), 1, 0)
        left_skeleton_array = self.corrp_data * left_mask_array
        #print(left_skeleton_values.mean())
        right_mask_array = np.where((HO_data==12) + (HO_data==13), 1, 0)
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
                self.significant_voxel_left_num / \
                np.count_nonzero(left_skeleton_array)) * 100

        if np.count_nonzero(right_skeleton_array) == 0:
            self.significant_voxel_right_percent = 0
        else:
            self.significant_voxel_right_percent = (
                self.significant_voxel_right_num / \
                np.count_nonzero(right_skeleton_array)) * 100


    def make_df(self):
        """Make summary pandas df of each corrp maps"""
        if self.significant:
            self.df = pd.DataFrame({
                'file name':[self.name],
                'Test':self.test_kind,
                'Modality':self.modality,
                'Stat num':self.stat_num,
                'Significance':self.significant,
                'Sig Max':self.voxel_max_p,
                'Sig Mean':self.significant_voxel_mean,
                'Sig Std':self.significant_voxel_std,
                '% significant voxels':self.significant_voxel_percentage,
                '% left':self.significant_voxel_left_percent,
                '% right':self.significant_voxel_right_percent
            })

            # round up columns that stars with percent
            for percent_col in [x for x in self.df.columns \
                                if x.startswith('%')]:
                self.df[percent_col] = self.df[percent_col].round(
                    decimals=1)
        else:
            self.df = pd.DataFrame({
                'file name':[self.name],
                'Test':self.test_kind,
                'Modality':self.modality,
                'Stat num':self.stat_num,
                'Significance':self.significant,
                'Sig Max':self.voxel_max_p,
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
        self.contrast_line = self.contrast_array[line_num,:]

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
                                      if re.search('^\d', x)][0]
                self.df['contrast'] = '* ' + design_fts_line
            except:
                self.df['contrast'] = 'f-test'

        # Reorder self.df to have file name and the contrast on the left
        self.df = self.df[['file name', 'contrast', 'contrast_text'] + \
                [x for x in self.df.columns if not x in ['file name',
                                                         'contrast',
                                                         'contrast_text']]]
        return self.df

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


#class CorrpMapDetail(CorrpMap):
    #"""CorrpMap in detail

    #To be used when the matrix_file, skeleton_dir and merged_4d_file are given.

    #Args:
        #location
        #threshold
        #matrix_file
        #skeleton_dir
        #merged_4d_file

    #TODO:
        #- Documentation
        #- Remove overlapping functions from this class to the RandomiseRun

    #"""
    ##def __init__(self, location, threshold, 
                 ##matrix_file, skeleton_dir, merged_4d_file):
    #def __init__(self, corrpMap, randomiseRun, merged_4d_file):
        #self.corrpMap = corrpMap
        ##CorrpMap.__init__(self, location, threshold)
        ##self.skeleton_dir = Path(skeleton_dir)
        #self.merged_4d_file = Path(merged_4d_file)
        #self.matrix_file = randomiseRun.matrix_file
        #RandomiseRun.get_matrix_info(self)

    #def get_skel_files(self):
        #"""Find skeleton files and return a list of Path objects 
        #"""
        #skel_ps = list(Path(self.skeleton_dir).glob('*nii.gz'))

        ## check whether the number of images match number of rows in the marix
        #if len(skel_ps) == self.matrix_array.shape[0]:
            #self.skel_ps = skel_ps
        #else:
            #print(f'Number of nifti images in the {self.skeleton_dir} is \
                  #different from that of matrix - please check')
            #self.skel_ps = []


    #def get_mean_values_for_all_subjects(self):
        #"""get mean values for skeleton files in the significant voxels

        #Args:
            #skeleton_files: list of Path objects, skeleton file locations.

        #Retrun:
            #df: pandas dataframe of 
                #'corrp_file', 'skeleton_file', 'average'
        #TODO: 
            #- save significant voxels
            #- parallelize
            #- think about using all_modality_merged images?
        #"""
        #merged_4d_data = nb.load(str(self.merged_4d_file)).get_data()
        #significant_cluster_data = np.where(
            #self.corrpMap.corrp_data >= self.corrpMap.threshold, 1, 0)

        #self.cluster_averages = {}
        #for vol_num in np.arange(merged_4d_data.shape[3]):
            #average = merged_4d_data[:,:,:,vol_num]\
                    #[significant_cluster_data == 1].mean()

            #self.cluster_averages[vol_num] = average
            ##skeleton_img = nb.load(str(skeleton_file))
            ##skeleton_data = skeleton_img.get_data()

            ##average = skeleton_data[significant_cluster_data == 1].mean()

        #self.cluster_averages_df = pd.DataFrame.from_dict(
            #self.cluster_averages,
            #orient='index', 
            #columns=[f'{self.corrpMap.modality} values in the significant '\
                     #f'cluster {self.corrpMap.name}']
        #)

    #def get_mean_values_for_all_subjects_skeleton_dir(self):
        #"""get mean values for skeleton files in the significant voxels

        #Args:
            #skeleton_files: list of Path objects, skeleton file locations.

        #Retrun:
            #df: pandas dataframe of 
                #'corrp_file', 'skeleton_file', 'average'
        #TODO: 
            #- save significant voxels
            #- parallelize
            #- think about using all_modality_merged images?
        #"""
        #skeleton_files = list(Path(self.skeleton_dir).glob('*.nii.gz'))
        #significant_cluster_data = np.where(self.corrp_data >= self.threshold,
                                            #1, 0)

        #self.cluster_averages = {}
        #for skeleton_file in skeleton_files:
            #skeleton_img = nb.load(str(skeleton_file))
            #skeleton_data = skeleton_img.get_data()
            #average = skeleton_data[significant_cluster_data == 1].mean()
            #self.cluster_averages[skeleton_file] = average

        #self.cluster_averages_df_from_skeleton_dir = pd.DataFrame.from_dict(
            #self.cluster_averages,
            #orient='index', 
        #)

        ### column name change
        ##self.cluster_averages_df_from_skeleton_dir.columns = [
            ##f'{self.modality} values in the cluster'
        ##]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        randomise_summary.py --dir /example/randomise/output/dir/
        ''',epilog="Kevin Cho Thursday, August 22, 2019")

    argparser.add_argument("--directory","-d",
                           type=str,
                           help='Specify randomise out dir. This this option \
                                 is given, design.mat and design.con within \
                                 the directory are read by default.',
                           default=os.getcwd())

    argparser.add_argument("--input","-i",
                           type=str,
                           nargs='+',
                           help='Specify randomise out corrp files. If this \
                                 option is given, --directory input is ignored')

    argparser.add_argument("--threshold","-t",
                           type=float,
                           help='Threshold for the significance',
                           default=0.95)

    argparser.add_argument("--contrast","-c",
                           type=str,
                           help='Contrast file used for the randomise.')

    argparser.add_argument("--matrix","-m",
                           type=str,
                           help='Matrix file used for the randomise')

    argparser.add_argument("--subject_values", "-s",
                           action='store_true',
                           help='Print average in the significant cluster for \
                           all subjects')

    argparser.add_argument("--merged_img_dir", "-p",
                           type=str,
                           help='Directory that contains merged files')

    argparser.add_argument("--atlasquery","-a",
                           action='store_true',
                           help='Run atlas query on significant corrp files')

    argparser.add_argument("--figure","-f",
                           action='store_true',
                           help='Create figures')

    args = argparser.parse_args()

    if not args.merged_img_dir:
        args.merged_img_dir = args.directory
    # if separate corrp image is given
    if args.input:
        corrpMaps = [Path(x) for x in args.input]
        corrp_map_classes = [CorrpMap(x, args.threshold) for x in corrpMaps]
        if args.matrix:
            map(lambda x: setattr(x, matrix_file, args.matrix), 
                corrp_map_classes)
            map(lambda x: x.get_matrix_info(), corrp_map_classes)
            map(lambda x: x.get_matrix_info(), corrp_map_classes)
            corrp_map_classes[0].print_matrix_info()
        if args.contrast:
            map(lambda x: setattr(x, contrast_file, args.contrast), 
                corrp_map_classes)
            map(lambda x: x.get_contrast_info(), corrp_map_classes)
            map(lambda x: x.get_contrast_info_english(), corrp_map_classes)
            map(lambda x: x.update_with_contrast(), corrp_map_classes)

    # or if randomise image is given
    else:
        print(args.directory, args.contrast, args.matrix)
        if args.contrast and args.matrix:
            randomiseRun = RandomiseRun(args.directory)
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
        randomiseRun.get_corrp_files()

        corrpMaps = randomiseRun.corrp_ps
        corrp_map_classes = [CorrpMap(x, args.threshold) for x in corrpMaps]
        # TODO : WHY DOES NOT MAP WORK in updating 'df' attribute within a class
        #map(lambda x: setattr(x, df, x.update_with_contrast()), corrp_map_classes)
        for corrpMap in corrp_map_classes:
            corrpMap.contrast_array = randomiseRun.contrast_array
            corrpMap.contrast_lines = randomiseRun.contrast_lines
            corrpMap.update_with_contrast()

    # if subject_values option is given
    if args.subject_values:
        print_head('Values extracted for each subject')
        values_df = pd.DataFrame()
        for corrpMap in corrp_map_classes:
            if corrpMap.significant:
                print('-'*80)
                print(corrpMap.name)
                print(corrpMap.modality)
                # find merged_4d_file
                merged_4d_file = list(Path(args.merged_img_dir).glob(
                    f'*all*_{corrpMap.modality}[_.]*nii.gz'))[0]
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
            print(f'{randomiseRun.location}/'\
                  'values_extracted_for_all_subjects.csv is created.')
        # if none of corrp map had significant voxels
        except:
            values_df.to_csv(
                f'{randomiseRun.location}/values_extracted_for_all_subjects.csv'
            )
            print(f'{randomiseRun.location}/'\
                  'values_extracted_for_all_subjects.csv is created.')

        values_df.index = [f'subject {x+1}' for x in values_df.index]
        print_df(values_df)

    df = pd.concat([x.df for x in corrp_map_classes], sort=False)
    df = df.sort_values('file name')
    print_head('Result summary')
    print_df(df.set_index(df.columns[0]))

    # If atlas query option is on
    if args.atlasquery:
        print_head('Atlas query of the significant cluster')
        for corrpMap in corrp_map_classes:
            if corrpMap.significant:
                corrpMap.get_atlas_query()
                print_df(corrpMap.df_query)

    # TODO figure option
    #if args.merged_4d_file:
        #df_significant = df[df.Significance == True]
        #for file_name in
        #print_df(df_significant)

    # If figure option is on
