import os
import randomise_summary
from pathlib import Path
script_root = Path(randomise_summary.__file__).parent.parent
script_dir = script_root / 'scripts'

import sys
sys.path.append(str(script_dir))

from skeleton_summary import parse_args, check_argparse_inputs, \
        run_skeleton_summary

# from randomise_summary import skeleton
# import numpy as np
# import tempfile
# import os

TEST_TBSS_DIR = script_root / 'test_tbss' / 'test_dir'

def test_skeleton_summary():
    stats_dir = TEST_TBSS_DIR / 'stats'
    merged_skeleton_loc = TEST_TBSS_DIR / 'stats' / 'all_FW_merged.nii.gz'
    tbss_all_out_dir = TEST_TBSS_DIR
    mask_loc = Path(os.environ['HOME']) / \
            'enigma_data/ENIGMA_DTI_FA_mask.nii.gz'

    args = parse_args([
        '-i', str(merged_skeleton_loc),
        '-d', str(stats_dir),
        '-m', str(mask_loc),
        '-tal', str(tbss_all_out_dir)
        ])

    check_argparse_inputs(args)
    run_skeleton_summary(args.merged_4d_file,
                         args.skeleton_mask,
                         args.tbss_all_loc,
                         directory=args.dir)



# def test_MergedSkeleton():
    # merged_skeleton_loc = TEST_TBSS_DIR / 'stats' / 'all_FW_merged.nii.gz'
    # mask_loc = Path(os.environ['HOME']) / \
            # 'enigma_data/ENIGMA_DTI_FA_mask.nii.gz'

    # mergedSkeleton = skeleton.MergedSkeleton(merged_skeleton_loc, mask_loc)

    # mergedSkeleton.skeleton_level_summary()
    # mergedSkeleton.subject_level_summary()

    # print(f'skeleton level info')
    # print(f'skeleton mean across subjects:'
          # f'{mergedSkeleton.merged_skeleton_mean}')
    # print(f'skeleton std across subjects:'
          # f'{mergedSkeleton.merged_skeleton_std}')

    # print(f'subject level info')
    # print(f'subject means : {mergedSkeleton.subject_nonzero_means}')
    # print(f'subject stds : {mergedSkeleton.subject_nonzero_stds}')

# def test():
    # skeleton_dir = '/data/pnl/kcho/PNLBWH/fsl_randomise/test_tbss/skeleton_dir'

    # skeletonDir = SkeletonDir(skeleton_dir)
    # skeletonDir.summary()
    # print_df(skeletonDir.df)
    # print_df(skeletonDir.merged_data_df)

    # # skeleton mean FSL
    # skeleton_files = [str(x) for x in list(Path(skeleton_dir).glob('*.nii.gz'))]
    # command = f'fslmaths {skeleton_files[0]} -add '
    # command += ' -add '.join(skeleton_files[1:])
    # output_file =  tempfile.NamedTemporaryFile(suffix='tmp.nii.gz')
    # command += ' '+output_file.name

    # command += f';fslmaths {output_file.name} -div {len(skeleton_files)} \
            # {output_file.name}'
    # command += f';fslstats {output_file.name} -M'
    # mean = os.popen(command).read()

    # # check mean
    # print ('Comparing result to that of FSL')
    # np.testing.assert_almost_equal(
        # skeletonDir.mean, 
        # float(mean),
        # decimal=4)
    
    # #skeletonDir.rms
    # #skeletonDir.figure()


# if __name__ == '__main__':
    # # test()
    # test_MergedSkeleton()
