import os
import sys
from pathlib import Path

script_dir = Path(os.path.realpath(__file__))
test_dir = script_dir.parent
pnl_randomise_dir = test_dir.parent / 'lib'
print(pnl_randomise_dir)

sys.path.append(str(pnl_randomise_dir))
from randomise_summary import *


def corrpmap():
    location = 'test_tbss/stats/tbss_FW_tfce_corrp_tstat1.nii.gz'
    threshold = 0.95
    threshold = 0.95 - 0.00001

    corrpMap = CorrpMap(location, threshold)
    return corrpMap

def test_corrpMap_info():
    corrpMap = corrpmap()

    assert corrpMap.modality == 'FW', 'modality does not match'
    assert corrpMap.significant == True, 'significance check'
    assert corrpMap.corrp_data.shape == (182,218,182), 'data array size check'
    assert corrpMap.vox_num_total == 117139, 'nonzero voxel number check in the skeleton'

def test_corrpMap_pstats():
    corrpMap = corrpmap()
    thresholded_map =  tempfile.NamedTemporaryFile(suffix='tmp.nii.gz')
    command = f'fslmaths {corrpMap.location} \
            -thr {corrpMap.threshold} -mas {corrpMap.skel_mask_loc} \
            -bin {thresholded_map.name}; fslstats {thresholded_map.name} -V'
    command = f'fslmaths {corrpMap.location} \
            -thr {corrpMap.threshold} -mas {corrpMap.skel_mask_loc} \
            -bin tmp; fslstats tmp -V'
    output = os.popen(command).read()
    assert corrpMap.significant_voxel_num == int(output.split(' ')[0]), \
            'nonzero voxel number check'

    command = f'fslmaths {corrpMap.location} \
            -thr {corrpMap.threshold} {thresholded_map.name};\
            fslstats {thresholded_map.name} -M'
    output = os.popen(command).read()
    np.testing.assert_almost_equal(corrpMap.significant_voxel_mean, 
                                   1-float(output.split(' ')[0]),
                                  decimal=4)
    
def test_corrpMap_sig_loc_each_hemisphere():
    corrpMap = corrpmap()
    fsl_dir = Path(environ['FSLDIR'])
    fsl_data_dir = fsl_dir / 'data'
    HO_dir = fsl_data_dir / 'atlases' / 'HarvardOxford'
    # 3d map
    HO_sub_thr0_1mm = HO_dir / 'HarvardOxford-sub-maxprob-thr0-1mm.nii.gz'

    HO_left_map = tempfile.NamedTemporaryFile(suffix='tmp.nii.gz')
    masked_map =  tempfile.NamedTemporaryFile(suffix='tmp.nii.gz', 
                                              prefix='./')
    command = f'''
    fslmaths {HO_sub_thr0_1mm} -thr 1 -uthr 2 -bin {HO_left_map.name};
    fslmaths {corrpMap.location} -mas {HO_left_map.name} {masked_map.name};
    fslmaths {masked_map.name} -thr 0.95 -bin {masked_map.name}; 
    fslstats {masked_map.name} -V'''
    output = os.popen(command).read()
    print(command)
    print(output)
    print(corrpMap.significant_voxel_left_num)
    assert corrpMap.significant_voxel_left_num==float(output.split(' ')[0]),\
            'checking the number of significant voxels on the left'


def test_corrpMap_update_with_contrast_test():
    corrpMap = corrpmap()

    #contrast_file = 'test_tbss/stats/design.con'
    corrpMap.contrast_array = np.array([[1, -1], [-1, 1]])
    corrpMap.get_contrast_info_english()
    assert corrpMap.contrast_lines == \
            ['Group 1 > Group 2', 'Group 1 < Group 2'], 'contrast line check'

    corrpMap.contrast_array = np.array([[-1, 1], [1, -1]])
    corrpMap.get_contrast_info_english()
    assert corrpMap.contrast_lines == \
            ['Group 1 < Group 2', 'Group 1 > Group 2'], 'contrast line check'

    # covariate
    corrpMap.contrast_array = np.array([[-1, 1, 0], [1, -1, 0]])
    corrpMap.get_contrast_info_english()
    assert corrpMap.contrast_lines == \
            ['Group 1 < Group 2', 'Group 1 > Group 2'], 'contrast line check'
    
    # three group
    corrpMap.contrast_array = np.array([[-1, 0, 1], [1, 0, -1]])
    corrpMap.get_contrast_info_english()
    assert corrpMap.contrast_lines == \
            ['Group 1 < Group 3', 'Group 1 > Group 3'], 'contrast line check'

    # three group
    corrpMap.contrast_array = np.array([[1, 0, -1], [-1, 0, 1]])
    corrpMap.get_contrast_info_english()
    assert corrpMap.contrast_lines == \
            ['Group 1 > Group 3', 'Group 1 < Group 3'], 'contrast line check'

    # correlation only
    corrpMap.contrast_array = np.array([[0, 0, 1], [0, 0, -1]])
    corrpMap.get_contrast_info_english()
    assert corrpMap.contrast_lines == \
            ['Positively correlated with col 3', 
             'Negatively correlated with col 3'], 'contrast line check'

    # correlation only
    corrpMap.contrast_array = np.array([[0, 0, 1], [0, 0, -1]])
    corrpMap.get_contrast_info_english()
    assert corrpMap.contrast_lines == \
            ['Positively correlated with col 3', 
             'Negatively correlated with col 3'], 'contrast line check'

    # interaction effect
    # correlation only
    corrpMap.contrast_array = np.array([[0, 0, 1, -1], [0, 0, -1, 1]])
    corrpMap.get_contrast_info_english()
    assert corrpMap.contrast_lines == \
            ['Positive Interaction', 
             'Negative Interaction'], \
            'interaction line check'

    matrix_file = 'test_tbss/stats/design.mat'
    corrpMap.matrix_file = matrix_file
    corrpMap.get_matrix_info()
    print(corrpMap.group_cols)
    # correlation only
    corrpMap.contrast_array = np.array([[0, 0, 1, -1], [0, 0, -1, 1]])
    #assert corr
    # if group column is zero


# todo write functions to test each functions
def get_detailed_corrp():
    # TEST randomise_summary
    location = 'test_tbss/stats/tbss_FW_tfce_corrp_tstat1.nii.gz'

    # Setting variables
    threshold = 0.95
    skeleton_dir = 'test_tbss/FW/skeleton'
    stats_dir = 'test_tbss/stats'
    matrix_file = 'test_tbss/stats/design.mat'
    contrast_file = 'test_tbss/stats/design.con'
    merged_4d_file = 'test_tbss/stats/all_FW_merged.nii.gz'

    # Create a RandomiseRun object
    # randomiseRun = RandomiseRun(stats_dir, contrast_file, matrix_file)
    corrpMap = CorrpMap(location, threshold, contrast_file=contrast_file,
            matrix_file=matrix_file)

    return corrpMap


def test_all():
    corrpMap = get_detailed_corrp()

    corrpMap.get_contrast_info()
    corrpMap.get_matrix_info()
    corrpMap.get_corrp_files()

def test_check_significance():
    corrpMap = get_detailed_corrp()
    corrpMap.check_significance()


def test_df_print():
    # Create a CorrpMap Detail object
    corrpMap = get_detailed_corrp()

    corrpMap.check_significance()
    corrpMap.df = pd.DataFrame({
        'file name':[corrpMap.name],
        'Test':corrpMap.test_kind,
        'Modality':corrpMap.modality,
        'Stat num':corrpMap.stat_num,
        'Significance':corrpMap.significant,
        'Max P':corrpMap.voxel_max_p,
        '% significant voxels':corrpMap.significant_voxel_percentage,
        '% left':corrpMap.significant_voxel_left_num,
        '% right':corrpMap.significant_voxel_right_num
    })

    print(corrpMap.df)

def test_skeleton_summary():
    # Create a CorrpMap Detail object
    corrpMap = get_detailed_corrp()

    corrpMap.check_significance()

    # corrpMap.get_skel_files()
    print(corrpMap.skel_mask_loc)
    print(corrpMap.skel_mask_loc)
    print(corrpMap.skel_mask_loc)
    print(corrpMap.skel_mask_loc)
    skeleton_summary(corrpMap)

    # compare values from the script vs that of FSL
    # corrpMap.get_mean_values_for_all_subjects()
    corrpMap.get_average_for_each_volume()

    # FSL values
    outputs = [0.078519, 0.107901, 0.123916, 0.124775]

    diff_array = np.array(corrpMap.cluster_averages_df.values.ravel()) - np.array(outputs)
    diff_threshold = 0.00005
    print(corrpMap.cluster_averages_df.reset_index())
    assert (diff_array < diff_threshold).all(), \
            f"Difference is greater than {diff_threshold}"

    print_df(corrpMap.cluster_averages_df.reset_index())

    # Create a function that checks the order of images in the merged_4d_file
    corrpMap.get_mean_values_for_all_subjects_skeleton_dir()
    assert (corrpMap.cluster_averages_df_from_skeleton_dir.values.ravel() == \
            corrpMap.cluster_averages_df.values.ravel()).all(), \
            "The order of skeleton in the merged_4d_file is different"

    # Check whether the number of images in the skeleton directory and the number
    # of volumes in the merged_4d_file are the same
    assert (corrpMap.matrix_array.shape[0] == \
            nb.load(str(corrpMap.merged_4d_file)).shape[-1]), \
            'matrix does not match 4d file'

    assert (nb.load(str(corrpMap.merged_4d_file)).shape[-1] == \
            len(corrpMap.skel_ps)), \
            'matrix does not match 4d file'

    # mark group columns correctly

def test_corrpMap_figure():
    location = 'test_tbss/stats/tbss_FW_tfce_corrp_tstat1.nii.gz'
    threshold = 0.95
    corrpMap = CorrpMap(location, threshold)

    corrpMap.out_image_loc = 'tmp.png'
    corrpMap.title = 'tmp.png'
    corrpMap.cbar_title = 'tmp.png'

    corrpMap.get_figure_enigma()


if __name__ == "__main__":
    print_head('Testing started')

    #corrpMap_test()
    print_head('Testing contrast detections')
    corrpMap_update_with_contrast_test()
    #corrpMap_figure()


