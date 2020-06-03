from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from pwd import getpwuid
import getpass
from os import stat
import os
import re
import time, datetime
import pandas as pd


def order_corrpMaps(corrpMaps):
    """Order corrpMaps so FA, FAt and FW are summarized before others"""
    corrpMap_modalities = [x.modality for x in corrpMaps]
    modality_order = ['FA', 'FAt', 'FW']
    modality_order = [x for x in modality_order if x in corrpMap_modalities]
    rest = [x for x in corrpMap_modalities if x not in modality_order]
    modality_order = modality_order + rest

    new_corrpMaps = []
    for modality in modality_order:
        for corrpMap in corrpMaps:
            if corrpMap.modality == modality:
                new_corrpMaps.append(corrpMap)

    return new_corrpMaps


def create_html(corrpMaps, df, args):
    """Create html that summarizes randomise_summary.py outputs"""
    # git version
    command = 'git rev-parse HEAD'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    git_hash = os.popen(command).read()

    corrpMaps = order_corrpMaps(corrpMaps)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(root, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('template.html')

    file_owners = []
    dates = []
    merged_4d_data_list = []
    modality_list = []

    outfigures = []
    filled_outfigures = []
    outsigfigures = []
    for corrpMap in corrpMaps:
        owner = getpwuid(stat(corrpMap.location).st_uid).pw_name
        file_owners.append(owner)
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = \
            os.stat(corrpMap.location)
        dates.append(time.strftime('%Y-%m-%d', time.localtime(mtime)))
        # print(time.strftime("%Y-%b-%d", time.ctime(mtime)))
        # if corrpMap.significant:

        corrpMap.out_image_loc = re.sub(
            '.nii.gz', '.png', str(corrpMap.location))
        corrpMap.filled_out_image_loc = re.sub(
            '.nii.gz', '_filled.png', str(corrpMap.location))

        if Path(corrpMap.out_image_loc).is_file():
            outfigures.append(True)
        else:
            outfigures.append(False)

        if Path(corrpMap.filled_out_image_loc).is_file():
            filled_outfigures.append(True)
        else:
            filled_outfigures.append(False)

        corrpMap.sig_out_image_loc = re.sub(
            '.nii.gz', '_sig_average_for_all_subjects.png',
            str(corrpMap.location))
        if Path(corrpMap.sig_out_image_loc).is_file():
            outsigfigures.append(True)
        else:
            outsigfigures.append(False)

        merged_4d_data_list.append(corrpMap.merged_4d_file)
        modality_list.append(corrpMap.modality)

    outfigures = any(outfigures)
    outsigfigures = any(outsigfigures)
    filled_outfigures = any(filled_outfigures)

    file_owners = list(set(file_owners))
    datecs = list(set(dates))
    merged_4d_data_list = list(set(merged_4d_data_list))
    modality_list = list(set(modality_list))

    # filename = os.path.join(root, 'prac_index.html')
    values_df_loc = corrpMap.location.parent / \
        'values_extracted_for_all_subjects.csv'
    if values_df_loc.is_file():
        values_df = pd.read_csv(str(values_df_loc), index_col=0)
        values_df = re.sub('dataframe', 'table', values_df.to_html())
    else:
        values_df = ''
    filename = corrpMap.location.parent / 'randomise_summary.html'

    # merged_4d_data_figures
    skeleton_average_figures = {}
    average_figures = {}
    std_figures = {}
    bin_sum_figures = {}
    bin_sum_diff_figures = {}
    skel_vol_figures = {}
    warped_map_figures = {}

    for merged_4d_file in merged_4d_data_list:
        try:
            modality = str(merged_4d_file.name).split('_')[1]
        except:
            modality = 'unknown'
        skeleton_average_figures[modality] = re.sub(
            '.nii.gz', '_skeleton_average_for_all_subjects.png',
            str(merged_4d_file))
        average_figures[modality] = re.sub(
            '.nii.gz', '_average.png',
            str(merged_4d_file))
        std_figures[modality] = re.sub(
            '.nii.gz', '_std.png',
            str(merged_4d_file))
        bin_sum_figures[modality] = re.sub(
            '.nii.gz', '_bin_sum.png',
            str(merged_4d_file))
        bin_sum_diff_figures[modality] = re.sub(
            '.nii.gz', '_bin_sum_diff.png',
            str(merged_4d_file))
        skel_vol_figures[modality] = re.sub(
            '.nii.gz', '_skeleton_volume_for_all_subjects.png',
            str(merged_4d_file))
        warped_map_figures[modality] = re.sub(
            '.nii.gz', '_skeleton_zero_mean_in_warp.png',
            str(merged_4d_file))

    # check whether the images exist
    skeleton_figure_check = all(
        [Path(x).is_file() for x in list(skeleton_average_figures.values())])

    skel_vol_figure_check = all(
        [Path(x).is_file() for x in list(skel_vol_figures.values())])

    warped_figure_check = all(
        [Path(x).is_file() for x in list(warped_map_figures.values())])


    mat_location = Path(corrpMaps[0].matrix_file)
    con_location = Path(corrpMaps[0].contrast_file)

    if not mat_location.is_absolute():
        mat_location = mat_location.resolve()
    if not con_location.is_absolute():
        con_location = con_location.resolve()

    with open(filename, 'w') as fh:
        fh.write(template.render(
            user=getpass.getuser(),
            owners=file_owners,
            datecs=datecs,
            total_number_of_subjects=len(corrpMaps[0].matrix_df),
            location=corrpMaps[0].location.parent,
            mat_location=mat_location,
            con_location=con_location,
            con_table=re.sub('dataframe', 'table',
                             corrpMaps[0].contrast_df.to_html()),
            mat_table=re.sub('dataframe', 'table',
                             corrpMaps[0].matrix_info.to_html()),
            table=re.sub('dataframe', 'table', df.to_html(index=False)),
            corrpMaps=[x.__dict__ for x in corrpMaps],
            outfigures=outfigures,
            outsigfigures=outsigfigures,
            filled_outfigures=filled_outfigures,
            values_df_loc=values_df_loc,
            values_df=values_df,
            skeleton_figure_check=skel_vol_figure_check,
            skel_vol_figure_check=skel_vol_figure_check,
            warped_figure_check=warped_figure_check,
            skeleton_average_figures=skeleton_average_figures,
            average_figures=average_figures,
            std_figures=std_figures,
            bin_sum_figures=bin_sum_figures,
            bin_sum_diff_figures=bin_sum_diff_figures,
            skel_vol_figures=skel_vol_figures,
            warped_map_figures=warped_map_figures,
            githash=git_hash
        ))
