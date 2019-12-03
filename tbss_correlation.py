# array
import pandas as pd
import numpy as np

from pathlib import Path
import re

# figure
import matplotlib.pyplot as plt
import seaborn as sns

# stats
from scipy.stats import pearsonr
import sys
sys.path.append('/data/pnl/kcho/PNLBWH/devel/kchopy/kchopy/kchostats')
import partial_correlation_rpy


class ValuesExtracted:
    """Values extracted"""
    def __init__(self, values_loc, caselist):
        self.df = pd.read_csv(str(values_loc), index_col=0)
        
        if Path(caselist).name == 'allFAsequence.txt':
            order_df = pd.read_csv(caselist, index_col=0)
            self.caselist = order_df.caseid.astype('str').tolist()
        else:
            with open(caselist, 'r') as f:
                self.caselist = [x.strip() for x in f.readlines()]

        self.df['subject'] = self.caselist

        self.sig_cols = [x for x in self.df.columns \
                                 if x.endswith('nii.gz')]
        self.sig_cols_simple = [self.get_easier_name(x) for x in self.sig_cols]


    def merge_demo_df(self, demo_df):
        """
        demo_df: pd.DataFrame
        """
        # merge with the given demo dataframe
        self.all_df = pd.merge(demo_df, self.df, on='subject', how='right')

        # select int and float columns only
        self.all_df_int_float = self.all_df.select_dtypes(
            include=['float', 'int']
        )


    def get_corr_map(self):
        self.corr_map = self.all_df_int_float.corr()


    def get_stats(self):
        p_dict = {}
        r_dict = {}
        variables = []
        for sig_col in self.sig_cols:
            p_dict[sig_col] = []
            r_dict[sig_col] = []
            for other_col in self.all_df_int_float.columns:
                if other_col not in self.sig_cols:
                    df_clean = self.all_df_int_float[[sig_col, other_col]].dropna()
                    r, p = pearsonr(df_clean[sig_col], df_clean[other_col])
                    p_dict[sig_col].append(p)
                    r_dict[sig_col].append(r)
                    if other_col not in variables:
                        variables.append(other_col)

        self.p_df = pd.DataFrame(
            p_dict,
            index=variables
        )

        self.r_df = pd.DataFrame(
            r_dict,
            index=variables
        )


    def get_stats_group(self, group):
        if group == 'SLE':
            tmp_df = self.all_df[self.all_df.label.isin(['nonNPSLE', 'NPSLE'])]
        else:
            tmp_df = self.all_df[self.all_df.label == group]

        p_dict = {}
        r_dict = {}
        variables = []
        for sig_col in self.sig_cols:
            p_dict[sig_col] = []
            r_dict[sig_col] = []
            for other_col in self.all_df_int_float.columns:
                try:
                    if other_col not in self.sig_cols:
                        df_clean = tmp_df[[sig_col, other_col]].dropna()
                        # r, p = pearsonr(df_clean[sig_col], df_clean[other_col])
                        pcor_out_df = partial_correlation_rpy.pcor(
                            df_clean[sig_col], df_clean[other_col],
                            df_clean['Pat_age'])
                        print(pcor_out_df)
                        r = pcor_out_df['estimate']
                        p = pcor_out_df['p']

                        p_dict[sig_col].append(p)
                        r_dict[sig_col].append(r)
                        if other_col not in variables:
                            variables.append(other_col)
                except ValueError:
                    pass

        self.p_df = pd.DataFrame(
            p_dict,
            index=variables
        )

        self.r_df = pd.DataFrame(
            r_dict,
            index=variables
        )
        

    def get_heat_map_p(self, group='all subjects'):
        # plot correlation columns
        threshold = 0.05

        ncols = self.p_df.shape[-1]
        width = ncols * 5 #15

        fig, axes = plt.subplots(ncols=ncols, figsize=(width,10), dpi=150)

        self.p_df.columns = self.sig_cols_simple

        for num, array in enumerate(np.array_split(self.p_df, ncols)):
            try:
                sns.heatmap(
                    np.where(array < threshold, array, np.nan), 
                    ax=axes[num],
                    annot=True,
                    fmt='.3f',
                    center=0.05, 
                    cmap='autumn', 
                    cbar=False)
            except:
                pass
            sns.heatmap(
                array, ax=axes[num], #annot=True, fmt='.3f',
                cmap='autumn', #alpha=0.3, #vmax=0.05,
                cbar=False)

        plt.xticks(rotation=70)
        fig.subplots_adjust(wspace=1.3)
        fig.suptitle(f'Pearson correlation p-values\nbetween the values in '\
                     f'the significant clusters VS clinical variables in '\
                     f'{group} \n(only showing P < {threshold})', 
                     fontsize=13, 
                     fontweight='bold', y=.97)
        fig.show()

    def get_heat_map_r(self, group='all subjects'):
        # plot correlation columns
        threshold = 0.05

        ncols = self.p_df.shape[-1]
        width = ncols * 5 #15
        if ncols >= 5:
            width = ncols * 6

        if ncols == 2:
            width = ncols * 4

        # plot correlation columns
        fig, axes = plt.subplots(ncols=ncols, figsize=(width,10), dpi=150)

        self.p_df.columns = self.sig_cols_simple
        self.r_df.columns = self.sig_cols_simple

        for num, (p_array, r_array) in enumerate(
            zip(np.array_split(self.p_df, ncols),
                np.array_split(self.r_df, ncols))):
            try:
                sns.heatmap(
                    np.where(p_array < threshold, r_array, np.nan), 
                    annot=True,
                    fmt='.3f', 
                    ax=axes[num],
                    center=0, 
                    cmap='coolwarm', 
                    cbar=False)
            except:
                pass
            sns.heatmap(
                r_array,
                ax=axes[num], #annot=True, fmt='.3f',
                cmap='coolwarm', alpha=0.3, #vmax=0.05,
                cbar=False)

            axes[num].set_xticklabels(axes[num].get_xticklabels(),
                                      rotation=90)

        fig.subplots_adjust(wspace=1.3)
        fig.suptitle(
            'Pearson correlation r-values\nbetween the values in the '\
            f'significant clusters VS clinical variables in {group}\n'\
            f'(only showing P < {threshold})', 
            fontsize=13, 
            fontweight='bold', y=.97)
        fig.show()


    def get_easier_name(self, full_name):
        modality = full_name.split(' ')[0]
        comp_number = re.search('(\d).nii.gz', full_name).group(1)
        if comp_number == '1':
            text = 'reduced in SLE'
        else:
            text = 'increased in SLE'
        return f'{modality} {text}'

