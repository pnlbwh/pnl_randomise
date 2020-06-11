#!/data/pnl/kcho/anaconda3/bin/python

print('Importing modules')

import argparse
import os
# corrp file
from randomise_corrp import run_randomise_summary_basics, run_atlas_query
# figure
from randomise_figure import create_figure
# individual summary
from randomise_individual_summary import get_individual_summary
# skeleton 
from skeleton_summary import get_skeleton_summary
# html
from randomise_summary_web import create_html

# options
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
pd.set_option('mode.chained_assignment', None)

print('Importing modules complete')

'''
TODO:
    - Test
    - write to do again
    - left and right hemispher mask to consistent = 1 rather than 0
    - link nifti_snapshots to it
    - why import take so long?
    - Save summary outputs in pdf, csv or excel?
    - TODO add "interaction" information
    - design mat and design con search function
'''


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
