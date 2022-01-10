#!/usr/bin/env python
import sys
import argparse
from pathlib import Path
from randomise_summary.skeleton import run_skeleton_summary


def parse_args(argv):
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Skeleton summary - assumes ENIGMA template',
        epilog="Kevin Cho Thursday, August 22, 2019")

    argparser.add_argument("--dir", "-d", type=str, help='design directory')
    argparser.add_argument("--contrast", "-con", type=str,
                           help='design contrast')
    argparser.add_argument("--matrix", "-mat", type=str, help='design matrix')
    argparser.add_argument("--merged_4d_file", "-i", type=str, required=True,
                           help='Merged 4d file')
    argparser.add_argument("--skeleton_mask", "-m", type=str, required=True,
                           help='Merged 4d file')
    argparser.add_argument("--warp_dir", "-w", type=str, help='warp dir')
    argparser.add_argument("--caselist", "-c", type=str, help='caselist')
    argparser.add_argument("--tbss_all_loc", "-tal", type=str,
                           help='tbss_all location')

    args = argparser.parse_args(argv)
    return args


def check_argparse_inputs(args):
    '''Check argparse inputs'''

    # print modality will be used
    modality = Path(args.merged_4d_file).name.split('_')[1]
    print(f'Modality of "{modality}" has been detected from the merged 4d file'
          ' name')

    cc = '- please check your inputs'

    assert Path(args.merged_4d_file).is_file(), \
        f"{args.merged_4d_file} is missing {cc}"

    assert Path(args.skeleton_mask).is_file(), \
        f"{args.skeleton_mask} is missing {cc}"


    if args.tbss_all_loc:
        assert Path(args.tbss_all_loc).is_dir(), \
            f"{args.tbss_all_loc} is missing {cc}"

        assert (Path(args.tbss_all_loc) / 'log/caselist.txt').is_file(), \
            f"{args.tbss_all_loc}/log/caselist.txt is missing {cc}"

        assert (Path(args.tbss_all_loc) / modality / 'warped').is_dir(), \
            f"{modality}/warped is missing under {args.tbss_all_loc} {cc}"
    else:
        for varstr in 'matrix', 'contrast', 'warp_dir', 'caselist':
            assert getattr(args, varstr), f"You need to provide --{varstr}"

    if args.dir:
        assert (Path(args.dir) / 'design.mat').is_file(), \
            f"{args.dir}/design.mat is missing {cc}"

        assert (Path(args.dir) / 'design.con').is_file(), \
            f"{args.dir}/design.mat is missing {cc}"
    else:
        assert Path(args.matrix).is_file(), f"{args.matrix} is missing {cc}"
        assert Path(args.contrast).is_file(), \
            f"{args.contrast} is missing {cc}"


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    check_argparse_inputs(args)

    if args.dir:
        run_skeleton_summary(args.merged_4d_file,
                             args.skeleton_mask,
                             args.tbss_all_loc,
                             directory=args.dir)
    else:
        run_skeleton_summary(args.merged_4d_file,
                             args.skeleton_mask,
                             args.tbss_all_loc,
                             matrix=args.matrix,
                             contrast=args.contrast,
                             caselist=args.caselist)
