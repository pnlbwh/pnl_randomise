# array
import pandas as pd
import numpy as np

from pathlib import Path
import re

# figure
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# stats
from scipy.stats import pearsonr, spearmanr
import sys
sys.path.append('/data/pnl/kcho/PNLBWH/devel/kchopy/kchopy/kchostats')
import partial_correlation_rpy
import os
import math


class ValuesExtracted:
    """Object for the extracted values from the tbss randomise"""

    def __init__(self, values_loc, caselist):
        """Read in values_loc and caselist

        values_loc: path of csv file created by randomise_summary.py
        caselist: path of csv file used to run tbss_all
        """

        self.df = pd.read_csv(str(values_loc), index_col=0)

        # add 'subject' column to the self.df
        # tbss_all creates allFAsequence.txt which could also be read
        if Path(caselist).name == 'allFAsequence.txt':
            order_df = pd.read_csv(caselist, index_col=0)
            self.caselist = order_df.caseid.astype('str').tolist()
        else:
            with open(caselist, 'r') as f:
                self.caselist = [x.strip() for x in f.readlines()]
        self.df['subject'] = self.caselist

        # list of column names that contain the extracted values
        self.sig_cols = [x for x in self.df.columns if x.endswith('nii.gz')]

    def get_sig_cols_simple(self, group_2):
        # use the number in the randomise output file name to return the
        # simplied comparison name
        self.sig_cols_simple = [
            self.get_easier_name(x, group_2) for x in self.sig_cols
        ]

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

    def get_stats_new(self, group, pcor=False, covar=[], method='pearson'):
        """Get correlation matrix between all the variables

        Key arguments:
            group: 'SLE', 'NPSLE', 'nonNPSLE' or 'HC' specific to lupus
        """

        if group == 'SLE':
            tmp_df = self.all_df[self.all_df.label.isin(['nonNPSLE', 'NPSLE'])]
        else:
            tmp_df = self.all_df[self.all_df.label == group]

        columns = tmp_df._get_numeric_data().columns
        col_combs = combinations(columns, 2)

        series_list = []
        for var1, var2 in col_combs:
            if (var1 in self.sig_cols or var2 in self.sig_cols) \
                    and var1 not in covar and var2 not in covar:
                df_clean = tmp_df[[var1, var2]].dropna()
                df_clean = pd.concat([
                    df_clean,
                    tmp_df.loc[df_clean.index][covar]],
                    axis=1)

                if pcor and len(df_clean) > 1:
                    pcor_out_df = partial_correlation_rpy.pcor(
                        df_clean[var1],
                        df_clean[var2],
                        np.array(df_clean[covar]),
                        method=method)
                    r = pcor_out_df['estimate']
                    p = pcor_out_df['p.value']
                elif not pcor and method == 'pearson':
                    r, p = pearsonr(
                        df_clean[var1],
                        df_clean[var2])
                elif not pcor and method == 'spearman':
                    r, p = spearmanr(
                        df_clean[var1],
                        df_clean[var2])
                else:
                    break

                s = pd.Series({
                    'var1': var1, 'var2': var2,
                    'n': len(df_clean),
                    'pcor': pcor,
                    'covar': ' '.join(covar),
                    'method': method,
                    'r': r,
                    'p': p})
                series_list.append(s)

        self.df_corr = pd.concat(series_list, axis=1).T

    def get_stats_group(self, group, pcor=False, covar=[], method='pearson'):
        if group == 'SLE':
            tmp_df = self.all_df[self.all_df.label.isin(['nonNPSLE', 'NPSLE'])]
        else:
            tmp_df = self.all_df[self.all_df.label == group]

        p_dict = {}
        r_dict = {}
        n_dict = {}

        variables = []
        for sig_col in self.sig_cols:
            p_dict[sig_col] = []
            r_dict[sig_col] = []
            n_dict[sig_col] = []
            for other_col in self.all_df_int_float.columns:
                try:
                    if other_col not in self.sig_cols:
                        df_clean = tmp_df[[sig_col, other_col]].dropna()

                        # attach covariate columns
                        df_clean = pd.concat([
                            df_clean,
                            tmp_df.loc[df_clean.index][covar]],
                            axis=1)


                        if pcor and len(df_clean) > 1:
                            pcor_out_df = partial_correlation_rpy.pcor(
                                df_clean[sig_col],
                                df_clean[other_col],
                                df_clean['Pat_age'],
                                method=method)
                            r = pcor_out_df['estimate']
                            p = pcor_out_df['p.value']
                        elif not pcor and method == 'pearson':
                            r, p = pearsonr(
                                df_clean[sig_col],
                                df_clean[other_col])
                        elif not pcor and method == 'spearman':
                            r, p = spearmanr(
                                df_clean[sig_col],
                                df_clean[other_col])
                        else:
                            break

                        p_dict[sig_col].append(p)
                        r_dict[sig_col].append(r)

                        # store number
                        n_dict[sig_col].append(len(df_clean))

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

        self.n_df = pd.DataFrame(
            n_dict,
            index=variables
        )

        if pcor:
            self.correlation_name = f'Partial correlation ' \
                f'({method.capitalize()} - {" ".join(covar)})'
        else:
            self.correlation_name = f'{method.capitalize()} correlation'

    def get_heat_map_p(self, group='all subjects', **kwargs):
        # plot correlation columns
        threshold = 0.05

        ncols = self.p_df.shape[-1]
        width = ncols * 5

        fig, axes = plt.subplots(
            ncols=ncols,
            figsize=(width, 10),
            dpi=150)

        # change name of the columns
        self.p_df.columns = self.sig_cols_simple

        # split the columns
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
                array, ax=axes[num],
                cmap='autumn',
                cbar=False)

            axes[num].set_xticklabels(
                   axes[num].get_xticklabels(), rotation=90)

        fig.subplots_adjust(wspace=1.3)
        fig.suptitle(
             f'{self.correlation_name} p-values\nbetween the values in '
             f'the significant clusters VS clinical variables in '
             f'{group} \n(only showing P < {threshold})',
             fontsize=13,
             fontweight='bold', y=.97)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        if 'out_img' in kwargs:
            out_img = kwargs.get('out_img')
            fig.savefig(out_img)
        else:
            fig.show()

    def get_heat_map_r(self, group='all subjects', **kwargs):
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

        for num, (p_array, r_array, n_array) in enumerate(
            zip(np.array_split(self.p_df, ncols),
                np.array_split(self.r_df, ncols),
                np.array_split(self.n_df, ncols))):
            try:
                matrix = np.where(p_array < threshold, r_array, np.nan)
                sns.heatmap(
                    matrix,
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

            axes[num].set_xticklabels(
                   axes[num].get_xticklabels(), rotation=90)

        fig.subplots_adjust(wspace=1.3)
        fig.suptitle(
            f'{self.correlation_name} r values\nbetween the values in the '
            f'significant clusters VS clinical variables in {group}\n'
            f'(only showing P < {threshold})',
            fontsize=13,
            fontweight='bold', y=.97)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        if 'out_img' in kwargs:
            out_img = kwargs.get('out_img')
            fig.savefig(out_img)
        else:
            fig.show()
        # fig.show()

    def get_easier_name(self, full_name, group_2):
        """Get easier name based on the file name

        Currently only works for simple t-test.
            '1' in the filename : Group 1 > Group 2
            '2' in the filename : Group 1 < Group 2

        Key arguments:
            full_name: str, file name
            group_2: str, name of the second group
        """
        modality = full_name.split(' ')[0]
        comp_number = re.search(r'(\d).nii.gz', full_name).group(1)
        if comp_number == '1':
            text = f'reduced in {group_2}'
        else:
            text = f'increased in {group_2}'
        return f'{modality} ({text})'


    def get_corr_graphs(self, group='all subjects', **kwargs):
        # plot correlation columns
        threshold = 0.05

        # self.p_df.columns = self.sig_cols_simple

        sig_num = 0
        for index, row in self.p_df.iterrows():
            for col in self.p_df.columns:
                if row[col] < threshold:
                    sig_num += 1

        # print(sig_num)

        height = math.ceil(sig_num/3) * 5
        fig, axes = plt.subplots(
                ncols=3,
                nrows=math.ceil(sig_num/3),
                figsize=(10, height), dpi=150)

        ax_num = 0
        for col_num, col in enumerate(self.p_df.columns):
            for index, row in self.p_df.iterrows():
                if row[col] < threshold:
                    ax = np.ravel(axes)[ax_num]

                    p = self.p_df.loc[index, col]
                    r = self.r_df.loc[index, col]

                    if group == 'SLE':
                        tmp_df = self.all_df[self.all_df.label.isin(['nonNPSLE', 'NPSLE'])][[index, col]]
                    else:
                        tmp_df = self.all_df[self.all_df.label == group][[index, col]]

                    sns.regplot(x=index, y=col, data=tmp_df, ax=ax)
                    ax.set_ylabel(self.sig_cols_simple[col_num])
                    ax.text(0.5, 0.9, f'r = {r:.3f}, P = {p:.3f}',
                            ha='center',
                            transform=ax.transAxes)
                    ax.text(0.5, 0.1, f'n = {len(tmp_df.dropna())}',
                            ha='center',
                            transform=ax.transAxes)
                    ax_num += 1

        fig.subplots_adjust(wspace=1.3)
        fig.tight_layout()
        print(axes.shape)
        try:
            fig.subplots_adjust(top=1-(0.15/axes.shape[1]))
        except:
            fig.subplots_adjust(top=0.83)
        fig.suptitle(
             f'{self.correlation_name} p-values\nbetween the values in '
             f'the significant clusters VS clinical variables in '
             f'{group} \n(only showing P < {threshold})',
             fontsize=13,
             fontweight='bold', y=.97)

        if 'out_img' in kwargs:
            out_img = kwargs.get('out_img')
            fig.savefig(out_img)
        else:
            fig.show()
