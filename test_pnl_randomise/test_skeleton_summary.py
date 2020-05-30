from skeleton_summary import *
import numpy as np
from pathlib import Path
import tempfile
import os

def test_MergedSkeleton():
    skeleton_dir = 'test_tbss/stats/all_FW_merged.nii.gz'
    mergedSkeleton = MergedSkeleton(skeleton_dir)
    mergedSkeleton.skeleton_level_summary()
    mergedSkeleton.subject_level_summary()

    print(f'skeleton level info')
    print(f'skeleton mean across subjects:'
          f'{mergedSkeleton.merged_skeleton_mean}')
    print(f'skeleton std across subjects:'
          f'{mergedSkeleton.merged_skeleton_std}')

    print(f'subject level info')
    print(f'subject means : {mergedSkeleton.subject_nonzero_means}')
    print(f'subject stds : {mergedSkeleton.subject_nonzero_stds}')

def test():
    skeleton_dir = '/data/pnl/kcho/PNLBWH/fsl_randomise/test_tbss/skeleton_dir'

    skeletonDir = SkeletonDir(skeleton_dir)
    skeletonDir.summary()
    print_df(skeletonDir.df)
    print_df(skeletonDir.merged_data_df)

    # skeleton mean FSL
    skeleton_files = [str(x) for x in list(Path(skeleton_dir).glob('*.nii.gz'))]
    command = f'fslmaths {skeleton_files[0]} -add '
    command += ' -add '.join(skeleton_files[1:])
    output_file =  tempfile.NamedTemporaryFile(suffix='tmp.nii.gz')
    command += ' '+output_file.name

    command += f';fslmaths {output_file.name} -div {len(skeleton_files)} \
            {output_file.name}'
    command += f';fslstats {output_file.name} -M'
    mean = os.popen(command).read()

    # check mean
    print ('Comparing result to that of FSL')
    np.testing.assert_almost_equal(
        skeletonDir.mean, 
        float(mean),
        decimal=4)
    
    #skeletonDir.rms
    #skeletonDir.figure()


if __name__ == '__main__':
    # test()
    test_MergedSkeleton()
