from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from pwd import getpwuid
import getpass
from os import stat
import os
import re
import time, datetime

def create_html(corrpMaps, df, args):
    root = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(root, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('template.html')

    file_owners = []
    dates = []
    merged_4d_data_list = []
    modality_list = []
    for corrpMap in corrpMaps:
        owner = getpwuid(stat(corrpMap.location).st_uid).pw_name
        file_owners.append(owner)
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = \
            os.stat(corrpMap.location)
        dates.append(time.strftime('%Y-%m-%d', time.localtime(mtime)))
        # print(time.strftime("%Y-%b-%d", time.ctime(mtime)))
        if corrpMap.significant:
            corrpMap.out_image_loc = re.sub(
                '.nii.gz', '.png', str(corrpMap.location))
            corrpMap.sig_out_image_loc = re.sub(
                '.nii.gz', '_sig_average_for_all_subjects.png',
                str(corrpMap.location))

        merged_4d_data_list.append(corrpMap.merged_4d_file)
        modality_list.append(corrpMap.modality)

    file_owners = list(set(file_owners))
    datecs = list(set(dates))
    merged_4d_data_list = list(set(merged_4d_data_list))
    modality_list = list(set(modality_list))

    # filename = os.path.join(root, 'prac_index.html')
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
            skeleton_average_figures=skeleton_average_figures,
            average_figures=average_figures,
            std_figures=std_figures,
            bin_sum_figures=bin_sum_figures,
            bin_sum_diff_figures=bin_sum_diff_figures,
            skel_vol_figures=skel_vol_figures,
            warped_map_figures=warped_map_figures,
        ))
