import os
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.stats import ranksums
import seaborn as sns
from scipy.stats import ttest_ind
from itertools import combinations, product
from statannotations.Annotator import Annotator
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def plot_kaplan(
    dataset,
    duration,
    event,
    groups=None,
    save=None,
    show_ci=True,
    interval=None,
    make_png=True,
    surv_to_int=False,
    which_to_show=["At risk", "Censored", "Events"],
    ci_alpha=0.3,
    censor_styles=None,
):
    data = dataset.copy()
    data.dropna(axis=0, subset=[duration, event], inplace=True)
    if surv_to_int:
        data[duration] = data[duration].astype(np.int32)
    if groups is not None:
        grps = data.groupby(groups)
    else:
        grps = [(duration, data)]
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8 + int(0.5 * len(grps)), 8 + len(grps)))
    ax_kmf = plt.subplot(111)
    for y_adj, (name, grouped_df) in enumerate(grps):
        if groups is not None:
            lbl = (
                ", ".join([str(j) + ": " + str(i) for i, j in zip(name, groups)])
                if len(groups) > 1
                else groups[0] + ": " + str(name)
            )
        else:
            lbl = str(name)
        kmf.fit(
            durations=grouped_df[duration], event_observed=grouped_df[event], label=lbl
        )
        kmf.plot_survival_function(
            at_risk_counts=False,
            ax=ax_kmf,
            show_censors=True,
            ci_show=show_ci,
            ci_alpha=ci_alpha,
            censor_styles=censor_styles,
        )
        add_at_risk_counts(kmf, ypos=-1 - y_adj, rows_to_show=which_to_show)

    if len(groups) == 1:
        results_comparison = multivariate_logrank_test(
            data[duration], data[groups[0]], data[event]
        )
        print(
            "Statistics: {0:.2f}, p-value: {1:.3f}.".format(
                results_comparison.test_statistic, results_comparison.p_value
            )
        )
        plt.title(
            duration
            + " grouped by: "
            + str(groups)
            + " p: "
            + str(results_comparison.p_value)
        )
    else:
        results_comparison = None
        plt.title(duration + " grouped by: " + str(groups))
    plt.xlabel(duration + " months")
    plt.ylabel("cumulative survival")
    if interval is not None:
        plt.ylim(interval)
    plt.grid(True)
    plt.tight_layout()
    if save is not None:
        os.makedirs(os.path.split(save)[0], exist_ok=True)
        plt.savefig(save, transparent=False, facecolor="white")
        if make_png & save.endswith((".pdf", ".svg")):
            plt.savefig(os.path.splitext(save)[0] + ".png", transparent=False)
        plt.close()
        if results_comparison is not None:
            return results_comparison
    else:
        plt.show()
        plt.close()
        if results_comparison is not None:
            return results_comparison


def frac_tum_w_iel_r(nucs, radius):
    iel_crd = nucs[nucs.iel == 1][["x", "y"]].values
    tum_crd = nucs[nucs.tum == 1][["x", "y"]].values
    dist = cdist(tum_crd, iel_crd)
    iel_subset = dist < radius
    n_iels = iel_subset.sum(1)
    return np.sum(n_iels > 0) / tum.shape[0]


def vio_plot(df, score, split, rmo=False, path=None):
    df = df.dropna(subset=[split]).copy()
    if rmo:
        rm_o = np.abs(df[score] - df[score].mean()) <= (3 * df[score].std())
        df = df[rm_o]
    pv = ranksums(df.loc[df[split] == 0, score], df.loc[df[split] == 1, score])
    ttest = ttest_ind(
        df.loc[df[split] == 0, score],
        df.loc[df[split] == 1, score],
        equal_var=False,
        permutations=10000,
    )
    if path is None:
        print("ranksum:", pv)
        print("perm welch test:", ttest)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(
        111,
        xlabel="x",
        ylabel="y",
        title=f"ranksum: {pv[0]:e} p:{pv[1]:.2e} \n perm welch test: {ttest[0]:e} p:{ttest[1]:.2e}",
    )
    # ax = plt.subplot(111, xlabel='x', ylabel='y', title=f"p: {pv[1]:.4f}")
    ax.title.set_fontsize(20)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(20)

    g = sns.violinplot(
        x=split, y=score, inner="quartile", linewidth=4, data=df, order=[0, 1]
    )
    sns.stripplot(
        x=split, y=score, size=10, color="k", data=df, order=[0, 1], alpha=0.6
    )
    if path is None:
        plt.show()
    else:
        plt.savefig(path, transparent=False, facecolor="white")
        plt.close()


def vio_plot_multi(df, score, var, rmo=True, path=None, txt_form="full"):
    df = df.dropna(subset=[var]).copy()
    if rmo:
        rm_o = np.abs(df[score] - df[score].mean()) <= (3 * df[score].std())
        df = df[rm_o]

    df[var] = df[var].astype(str).values

    vals = sorted(df[var].unique())
    print(vals)
    pairs = list(combinations(vals, 2))
    print(pairs)

    plt.figure(figsize=(3 * len(vals), 8))
    ax = plt.subplot(111, xlabel="x", ylabel="y")
    ax.title.set_fontsize(20)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(20)

    g = sns.violinplot(
        x=var, y=score, inner="quartile", linewidth=3, data=df, order=vals
    )
    sns.stripplot(x=var, y=score, size=7, color="k", data=df, order=vals, alpha=0.6)

    annotator = Annotator(g, pairs, data=df, x=var, y=score, order=vals)
    annotator.configure(
        test="Mann-Whitney",
        text_format=txt_form,
        loc="inside",
        comparisons_correction="BH",
        correction_format="replace",
        show_test_name=False,
    )
    annotator.apply_and_annotate()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, transparent=False, facecolor="white")
        plt.close()


def bar_plot_multi_sub(
    df,
    score,
    var,
    subvar,
    rmo=True,
    path=None,
    pairs=None,
    txt_form="full",
    hide_ns=False,
    add_strip=False,
    ylim=None,
    order=None,
    custom_figsize=None,
    log_scale=False,
    comparison_correction="BH",
):
    df = df.dropna(subset=[var, subvar]).copy()
    if rmo:
        rm_o = np.abs(df[score] - df[score].mean()) <= (3 * df[score].std())
        df = df[rm_o]

    df[var] = df[var].astype(str).values
    df[subvar] = df[subvar].astype(str).values

    vals = sorted(df[var].unique()) if order is None else order
    if custom_figsize is None:
        plt.figure(figsize=(2.2 * len(vals), 8))
    else:
        plt.figure(figsize=custom_figsize)
    ax = plt.subplot(111, xlabel="x", ylabel="y")
    if ylim is not None:
        ax.set_ylim(ylim)
    if log_scale == True:
        ax.set_yscale("log")
    ax.title.set_fontsize(20)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(20)

    converted = []
    for i, r in df.iterrows():
        converted.append({var: r[var], subvar: r[subvar], score: r[score]})
    converted = pd.DataFrame(converted)

    hue_order = sorted(converted[subvar].unique())

    # pairs = list(combinations(product(vals, hue_order),2))
    if pairs is None:
        pairs = [
            i
            for sl in [
                list(
                    combinations(
                        product(
                            [j],
                            sorted(converted.loc[converted[var] == j, subvar].unique()),
                        ),
                        2,
                    )
                )
                for j in vals
            ]
            for i in sl
        ]

    # g = sns.violinplot(
    #    x=var, y=score, inner="quartile", linewidth=3, data=df, order=vals
    # )

    #
    df = df.sort_values(by=subvar)
    g = sns.boxplot(
        data=df,
        x=var,
        y=score,
        hue=subvar,
        order=vals,
        hue_order=hue_order,
        fliersize=0,
        fill=False,
    )
    if add_strip:
        g = sns.stripplot(
            x=var,
            y=score,
            size=4,
            palette="dark:k",
            data=df,
            order=vals,
            hue=subvar,
            hue_order=hue_order,
            alpha=1.0,
            ax=g,
            dodge=True,
            jitter=0.2,
            legend=False,
            marker="$\circ$",
            ec="face",
        )

    annotator = Annotator(
        g,
        pairs,
        data=df,
        x=var,
        y=score,
        hue=subvar,
        order=vals,
        hue_order=hue_order,
    )
    annotator.configure(
        test="Mann-Whitney",
        text_format=txt_form,
        loc="inside",
        comparisons_correction=comparison_correction,
        correction_format="replace",
        show_test_name=False,
        hide_non_significant=hide_ns,
        line_height=0.005,
    )
    annotator.apply_and_annotate()
    plt.tight_layout()
    plt.ylim(ylim)
    if path is None:
        plt.show()
    else:
        plt.savefig(path, transparent=False, facecolor="white")
        plt.close()


def stand(x):
    return (x - x.mean()) / x.std()


def get_dum(df, x):
    tmp = pd.get_dummies(df[x], dummy_na=True, prefix=x)
    tmp[tmp.iloc[:, -1] == 1] = np.NaN
    return tmp.iloc[:, :-1].copy()


def reg_plot(x, y, data, c=None, rmo=True, kin="scatter"):
    df = data.copy()
    df[x] = stand(df[x])
    df[y] = stand(df[y])
    if rmo:
        rm_o_1 = np.abs(df[x] - df[x].mean()) <= (3 * df[x].std())
        rm_o_2 = np.abs(df[y] - df[y].mean()) <= (3 * df[y].std())
        df = df[rm_o_1 & rm_o_2]

    if c is None:
        ax = sns.jointplot(data=df, x=x, y=y, kind="reg")
    else:
        if kin == "scatter":
            ax = sns.jointplot(
                data=df, x=x, y=y, hue=c, kind=kin, marker="o", s=50, alpha=0.7
            )
        else:
            ax = sns.jointplot(
                data=df, x=x, y=y, hue=c, kind=kin, levels=20, alpha=0.5, fill=True
            )
    pr, p = pearsonr(df[x], df[y])
    r2 = r2_score(df[x], df[y])
    ax.figure.suptitle(f"Pearson CC: {pr:.4f}, p: {p:e} \n R²:{r2:.4f}")
    ax.fig.set_figwidth(10)
    ax.fig.set_figheight(10)
    plt.tight_layout()
    plt.show()
    plt.close()


def rmo(df, x):
    rm_o = np.abs(df[x] - df[x].mean()) <= (3 * df[x].std())
    df.loc[~rm_o, x] = np.nan
    return df


def conv_bud(x):
    if np.isnan(x):
        return np.NaN
    elif x <= 4:
        return 1
    elif x <= 9:
        return 2
    else:
        return 3
