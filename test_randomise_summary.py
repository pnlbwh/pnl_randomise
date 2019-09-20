from randomise_summary import *
import os
from pathlib import Path

def corrpMap_test():
    location = 'test_tbss/stats/tbss_no_cov_FW_tfce_corrp_tstat1.nii.gz'
    threshold = 0.95
    corrpMap = CorrpMap(location, threshold)
    assert corrpMap.modality == 'FW', 'modality does not match'
    assert corrpMap.significant == True, 'significance check'
    assert corrpMap.corrp_data.shape == (182,218,182), 'data array size check'
    assert corrpMap.vox_num_total == 64155, 'nonzero voxel number check'


    thresholded_map =  tempfile.NamedTemporaryFile(suffix='tmp.nii.gz')
    command = f'fslmaths {corrpMap.location} \
            -thr {corrpMap.threshold} -bin \
            {thresholded_map.name}; fslstats {thresholded_map.name} -V'
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
    
    fsl_dir = Path(environ['FSLDIR'])
    fsl_data_dir = fsl_dir / 'data'
    HO_dir = fsl_data_dir / 'atlases' / 'HarvardOxford'
    # 3d map
    HO_sub_thr0_1mm = HO_dir / 'HarvardOxford-sub-maxprob-thr0-1mm.nii.gz'

    HO_left_map = tempfile.NamedTemporaryFile(suffix='tmp.nii.gz')
    masked_map =  tempfile.NamedTemporaryFile(suffix='tmp.nii.gz', 
                                              prefix='./')
    print(masked_map.name)
    command = f'''
    fslmaths {HO_sub_thr0_1mm} -thr 1 -uthr 2 -bin {HO_left_map.name};
    fslmaths {corrpMap.location} -mas {HO_left_map.name} {masked_map.name};
    fslmaths {masked_map.name} -thr 0.95 -bin {masked_map.name}; 
    fslstats {masked_map.name} -V'''
    output = os.popen(command).read()
    assert corrpMap.significant_voxel_left_num==float(output.split(' ')[0]),\
            'checking the number of significant voxels on the left'


def corrpMap_update_with_contrast_test():
    location = 'test_tbss/stats/tbss_no_cov_FW_tfce_corrp_tstat1.nii.gz'
    threshold = 0.95
    corrpMap = CorrpMap(location, threshold)

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
    matrix_file = 'test_tbss/stats/design.mat'
    corrpMap.matrix_file = matrix_file
    corrpMap.get_matrix_info()
    print(corrpMap.group_cols)
    # correlation only
    corrpMap.contrast_array = np.array([[0, 0, 1, -1], [0, 0, -1, 1]])
    #assert corr
    # if group column is zero



# todo write functions to test each functions
def test():
    # TEST randomise_summary

    # Setting variables
    threshold = 0.95
    skeleton_dir = 'test_tbss/FW/skeleton'
    stats_dir = 'test_tbss/stats'
    matrix_file = 'test_tbss/stats/design.mat'
    contrast_file = 'test_tbss/stats/design.con'
    merged_4d_file = 'test_tbss/stats/all_FW_merged.nii.gz'

    # Create a RandomiseRun object
    randomiseRun = RandomiseRun(stats_dir, contrast_file, matrix_file)
    randomiseRun.get_contrast_info()
    randomiseRun.get_matrix_info()
    randomiseRun.get_corrp_files()

    # Create a CorrpMap Detail object
    corrp_map = randomiseRun.corrp_ps[0]
    corrpMap = CorrpMapDetail(corrp_map, threshold, matrix_file, skeleton_dir, merged_4d_file)

    corrpMap.check_significance()
    corrpMap.df = pd.DataFrame({
        'file name':[corrpMap.name],
        'Test':corrpMap.test_kind,
        'Modality':corrpMap.modality,
        'Stat num':corrpMap.stat_num,
        'Significance':corrpMap.significant,
        'Max P':corrpMap.max_pval,
        '% significant voxels':corrpMap.vox_p,
        '% left':corrpMap.vox_left_p,
        '% right':corrpMap.vox_right_p
    })

    corrpMap.get_skel_files()

    # compare values from the script vs that of FSL
    corrpMap.get_mean_values_for_all_subjects()
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

if __name__ == "__main__":
    #corrpMap_test()
    corrpMap_update_with_contrast_test()


