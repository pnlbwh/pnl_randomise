import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.weightstats import ttest_ind


def anova(df, formula):
    lm = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(lm, typ=2)
    return table


def ttest(x1, x2):
    return ttest_ind(x1, x2)
