#!/data/pnl/kcho/anaconda3/bin/python

from pathlib import Path
import tempfile

# Imaging
import argparse
import pandas as pd
import re
import numpy as np
from os import environ
import os

pd.set_option('mode.chained_assignment', None)

# figures
import matplotlib.pyplot as plt
from scipy import ndimage

# utils
from pnl_randomise_utils import get_nifti_data, get_nifti_img_data


'''
TODO:
    - Test
    - Save summary outputs in pdf, csv or excel?
    - TODO add interaction information (group info to the get_contrast_info_english)
    - Check whether StringIO.io is used
    - Estimate how much voxels overlap with different atlases with varying 
      thresholds.
    - Move useful functions to kcho_utils.
    - Parallelize
'''

class FA:
    def __init__(self, **kwargs):
        self.fa = kwargs.pop('fa')
        self.mask = kwargs.pop('mask')

    def read_fa_maps(self):
        self.fa_img, self.fa_data = get_nifti_img_data(self.fa)

    def read_mask_maps(self):
        self.mask_img, self.mask_data = get_nifti_img_data(self.mask)

    def zero_FA_within_mask(self):
        self.read_fa_maps()
        self.read_mask_maps()
        
        self.zero_within_mask = \
                (self.fa_data == 0) * (self.mask_data == 1)

        if (self.zero_within_mask).any():
            return True

    def get_distance_from_nearest_zero(self):
        pass


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        randomise_summary.py --dir /example/randomise/output/dir/
        ''',epilog="Kevin Cho Thursday, August 22, 2019")

    argparser.add_argument("--fa","-f",
                           type=str,
                           help='FA map')

    argparser.add_argument("--mask","-m",
                           type=str,
                           help='Mask')

    args = argparser.parse_args()


    #check_holes_FA(fa=args.fa, mask=args.mask)
    f = FA(fa=args.fa, mask=args.mask)
    print(f.zero_FA_within_mask())

