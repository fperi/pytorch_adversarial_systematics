import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import torch
import math


def colored_scatter(x, y, c=None):

    '''
    Auxiliary function to create a scatter plot
    of the samples

    :param x: x coordinate of the samples
    :param y: y coordinate of the samples
    :param c: color for the sample

    :return: matplot scatter plot
    '''

    def scatter(*args, **kwargs):
        args = (x, y)
        if c is not None:
            kwargs['c'] = c
        kwargs['alpha'] = 0.7
        kwargs['s'] = 2
        plt.scatter(*args, **kwargs)

    return scatter


def plot_scatter(df, features, classes, net=None):

    '''
    Function plots a 2d scatter plot of the available
    classes together with the corresponding projection
    on the two axes.

    :param df: input dataframe
    :param features: list of the features name
    :param classes: list of classes to be included in the plot
    :param net: neural network to be used to draw the decision boundaries

    :return: seaborn plot
    '''

    # plot a subsample of points
    sub_sample = 5000
    
    # set seeds
    RANDOM_STATE = 42

    # initialise legend
    legends = []

    # setup boundaries
    x_min, x_max = -6.5, 6.5
    y_min, y_max = -6.5, 6.5

    # initialise main plot
    g = sns.JointGrid(x=df.loc[df['test_flag'], 'x'][:sub_sample],
                      y=df.loc[df['test_flag'], 'y'][:sub_sample])

    # loop over classes
    for name, df_group in df.groupby('class_name'):

        if name not in classes:
            continue

        # append to legend
        legends.append(name)

        # create scatter plot
        g.plot_joint(
            colored_scatter(df_group.loc[df['test_flag'],
                                         'x'][:sub_sample],
                            df_group.loc[df['test_flag'],
                                         'y'][:sub_sample],
                            df_group.loc[df['test_flag'],
                                         'class_color'][:sub_sample]),
        )

        # create projection plots
        sns.distplot(
            df_group.loc[df['test_flag'], 'x'][:sub_sample].values,
            ax=g.ax_marg_x,
            color=list(df_group.loc[df['test_flag'],
                                    'class_color'][:sub_sample])[0],
            hist=False,
            kde_kws={"shade": True}
        )
        sns.distplot(
            df_group.loc[df['test_flag'], 'y'][:sub_sample].values,
            ax=g.ax_marg_y,
            color=list(df_group.loc[df['test_flag'],
                                    'class_color'][:sub_sample])[0],
            vertical=True,
            hist=False,
            kde_kws={"shade": True}
        )

    # when a neural network is passed, the scatter plot includes the
    # corresponding decision boundaries
    accuracy = None
    if net is not None:
        # Create grid
        spacing = min(x_max + 0.1 - x_min,
                      y_max + 0.1 - y_min) / 100
        XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                             np.arange(y_min, y_max + 0.1, spacing))

        # predict nnet output on grid
        data = torch.FloatTensor(np.hstack((XX.ravel().reshape(-1, 1),
                                            YY.ravel().reshape(-1, 1))))
        db_prob = net(data)
        # add decision boundaries to plot
        plt.contourf(XX, YY, db_prob.detach().numpy().reshape(XX.shape),
                     cmap=plt.cm.RdGy, alpha=0.2)

        # compute and print accuracy
        X_test = (df.loc[(df['test_flag']) &
                         (df['class_name'].isin(classes)), features]).values
        Y_test = (df.loc[(df['test_flag']) &
                         (df['class_name'].isin(classes)), 'class_idx']).values
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        X_test, Y_test = ros.fit_resample(X_test, Y_test)
        X_test_t = torch.FloatTensor(X_test)
        Y_test_t = torch.FloatTensor(Y_test).reshape(-1, 1)
        hat = net(X_test_t)
        hat_class = np.where(hat.detach().numpy() < 0.5, 0, 1)
        accuracy = (np.sum(Y_test_t.detach().numpy() == hat_class)
                    / len(Y_test_t))

    plt.legend(legends)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    g.fig.set_figheight(8)
    g.fig.set_figwidth(8)

    return g.fig, accuracy


def plot_hist(df, classes, pred):

    '''
    Function plots histogram of the output probability distributions
    of the neural network for the various classes

    :param df: input dataframe
    :param classes: for which classes to plot the distribution
    :param pred: column in which the network output is saved

    :return: matplot histogram
    '''

    # initialise legend
    legends = []

    # initialise number of bins
    bins = 10

    # setup boundaries
    x_min, x_max = 0, 1
    y_min, y_max = 0, 2

    # initialise plot
    fig = plt.figure(figsize=(7.5, 14))
    gs = fig.add_gridspec(5, 5)
    ax1 = fig.add_subplot(gs[0:3, :])

    # loop over classes
    for name, df_group in df.groupby('class_name'):

        if name not in classes:
            continue

        legends.append(name)

        sns.distplot(
            np.array(list(df_group.loc[df_group['test_flag'],
                                       pred].values))[:],
            color=list(df_group.loc[df_group['test_flag'], 'class_color'])[0],
            kde_kws={'clip': (0.0, 1.0)},
            hist_kws={'range': (0, 1)},
            kde=False,
            bins=bins)

    # plot main histograms
    ax1.set_xlim((x_min, x_max))
    ax1.legend(legends)

    # setup ratio plots
    b_ranges = np.arange(x_min, x_max + x_max / bins, x_max / bins)
    b_centers = np.arange(x_min + x_max / (2 * bins), x_max, x_max / bins)

    # add ratio plot for systematic 1
    if 'gaus_s1' in classes:

        ax2 = fig.add_subplot(gs[3, :])

        nominal_values = \
            np.array(list(df.loc[(df['test_flag']) &
                                 (df['class_name'] == 'gaus'),
                                 pred].values))[:]
        syst_values = \
            np.array(list(df.loc[(df['test_flag']) &
                                 (df['class_name'] == 'gaus_s1'),
                                 pred].values))[:]

        # get histograms
        hist_syst, edges_syst = np.histogram(syst_values, b_ranges)
        hist_nom, edges_nom = np.histogram(nominal_values, b_ranges)

        # compute ratio and errors
        hist_syst = [x / y if y != 0 else 0 for x, y
                     in zip(hist_syst, hist_nom)]
        erry = [3*math.sqrt(x) / y if y != 0 else 0 for x, y
                in zip(hist_syst, hist_nom)]
        errx = [0.5 / bins] * len(hist_nom)
        hist_nom = [1] * len(hist_nom)

        # plot
        ax2.set_xlim((x_min, x_max))
        ax2.set_ylim((y_min, y_max))
        ax2.errorbar(b_centers, hist_syst,
                     yerr=erry, xerr=errx,
                     fmt='o', color=list(df.loc[df['class_name'] == 'gaus_s1',
                                                'class_color'])[0])
        legends = ['ratio gaus_s1/gaus']
        ax2.errorbar(b_centers, hist_nom,
                     xerr=errx, fmt='o',
                     color=list(df.loc[df['class_name'] == 'gaus',
                                       'class_color'])[0])
        ax2.legend(legends)

    # add ratio plot for systematic 2
    if 'gaus_s2' in classes:

        ax3 = fig.add_subplot(gs[4, :])

        nominal_values = \
            np.array(list(df.loc[(df['test_flag']) &
                                 (df['class_name'] == 'gaus'),
                                 pred].values))[:]
        syst_values = \
            np.array(list(df.loc[(df['test_flag']) &
                                 (df['class_name'] == 'gaus_s2'),
                                 pred].values))[:]

        # get histograms
        hist_syst, edges_syst = np.histogram(syst_values, b_ranges)
        hist_nom, edges_nom = np.histogram(nominal_values, b_ranges)

        # compute ratio and errors
        hist_syst = [x / y if y != 0 else 0 for x, y
                     in zip(hist_syst, hist_nom)]
        erry = [3*math.sqrt(x) / y if y != 0 else 0 for x, y
                in zip(hist_syst, hist_nom)]
        errx = [0.5 / bins] * len(hist_nom)
        hist_nom = [1] * len(hist_nom)

        # plot
        ax3.set_xlim((x_min, x_max))
        ax3.set_ylim((y_min, y_max))
        ax3.errorbar(b_centers, hist_syst,
                     yerr=erry, xerr=errx,
                     fmt='o', color=list(df.loc[df['class_name'] == 'gaus_s2',
                                                'class_color'])[0])
        legends = ['ratio gaus_s2/gaus']
        ax3.errorbar(b_centers, hist_nom,
                     xerr=errx, fmt='o',
                     color=list(df.loc[df['class_name'] == 'gaus',
                                       'class_color'])[0])
        ax3.legend(legends)

    return fig
