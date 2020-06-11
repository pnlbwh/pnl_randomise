from pathlib import Path
from typing import Union, List, Tuple


def get_corrp_files_glob_string(location:Union[Path, str], glob_string:str) \
        ->List:
    """Find corrp files and return a list of Path objects"""
    corrp_ps = list(Path(location).glob(glob_string))
    # remove corrp files that are produced in the parallel randomise
    corrp_ps = [str(x) for x in corrp_ps if 'SEED' not in x.name]

    if len(corrp_ps) == 0:
        print(f'There is no corrected p-maps in {self.location}')

    return corrp_ps


def get_corrp_files(location:Union[Path, str]) -> List:
    """Find corrp files and return a list of Path objects"""
    corrp_ps = list(Path(location).glob('*corrp*.nii.gz'))
    # remove corrp files that are produced in the parallel randomise
    corrp_ps = [str(x) for x in corrp_ps 
            if 'SEED' not in x.name and 'filled' not in x.name]

    if len(corrp_ps) == 0:
        print(f'There is no corrected p-maps in {self.location}')

    return corrp_ps


def get_corrp_map_locs(args: object) -> List[object]:
    '''Return corrp map paths from args'''
    if args.input: # separate inputs
        corrp_map_locs = args.input

    else: # directory as the input
        # load list of corrp files
        if args.f_only:
            corrp_map_locs = get_corrp_files_glob_string(args.directory,
                                                         '*corrp_f*.nii.gz')
        else:
            corrp_map_locs = get_corrp_files(args.directory)

    return corrp_map_locs


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
