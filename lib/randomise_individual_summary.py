from randomise_files import check_corrp_map_locations
from pnl_randomise_utils import pd, re
from pnl_randomise_utils import print_head, print_df
from skeleton_summary import MergedSkeleton, SkeletonDir, SkeletonDirSig

import matplotlib.pyplot as plt


from typing import List


def get_individual_summary(args: object, corrp_map_classes: List[object]) -> None:
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



