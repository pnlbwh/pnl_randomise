# Pandas dataframe print
# only library that is not included in the anaconda 3
from tabulate import tabulate

def print_df(df):
    """Print pandas dataframe using tabulate.

    Used to print outputs when the script is called from the shell
    Key arguments:
        df: pandas dataframe
    """
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print()


def print_head(heading):
    print()
    print('-'*80)
    print(f'* {heading}')
    print('-'*80)
