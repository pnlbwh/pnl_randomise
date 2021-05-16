import tempfile

from pnl_randomise_utils import get_nifti_data, get_nifti_img_data
from pnl_randomise_utils import np, pd, re, Path
from pnl_randomise_utils import search_and_select_one
from pnl_randomise_utils import print_head, print_df

from randomise_files import get_corrp_map_locs

from randomise_con_mat import RandomiseConMat
from randomise_figure import CorrpMapFigure

from os import environ
import os, re, sys
from typing import Union, List, Tuple

class CorrpMap(RandomiseConMat, CorrpMapFigure):
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
                '(' + '|'.join(self.modality_full_list) + ')_',
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
        print('Command used')
        print('\t'+re.sub('\s+', ' ', command))
        os.popen(command).read()

        # run atlas query from FSL
        # label
        command = f'atlasquery \
                -m {thresholded_map.name} \
                -a "JHU ICBM-DTI-81 White-Matter Labels"'
        text_label = os.popen(command).read()
        print('\t'+re.sub('\s+', ' ', command))
        print('\t\t' + re.sub('\n', '\n\t\t', text_label))

        # tract
        command = f'atlasquery \
                -m {thresholded_map.name} \
                -a "JHU White-Matter Tractography Atlas"'
        text_tract = os.popen(command).read()
        print('\t'+re.sub('\s+', ' ', command))
        print('\t\t' + re.sub('\n', '\n\t\t', text_tract))


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


def get_corrp_map_classes(corrp_map_locs: List, 
                          args: object) -> List[object]:
    '''Return corrpMaps from the corrp_map_locs and additional args info'''
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


def run_randomise_summary_basics(args: object) \
        -> Tuple[List[CorrpMap], pd.DataFrame]:
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


def run_atlas_query(args: object, corrp_map_classes: List[object]) -> None:
    '''Atlas query'''
    # if atlas query option is on
    if args.atlasquery:
        print_head('Atlas query of the significant cluster')
        print_head('(atlasquery from FSL)')
        for corrpMap in corrp_map_classes:
            if corrpMap.significant:
                corrpMap.get_atlas_query()
                print_df(corrpMap.df_query)
