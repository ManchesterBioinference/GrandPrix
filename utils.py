import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import  numpy as np
from collections import OrderedDict

matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.style.use('ggplot')

def calcroughness(x, pt):
    x=np.atleast_2d(x)
    i=np.argsort(pt)
    x = x[:,i]
    N = x.shape[1]
    assert(N > 0)
    S = x.std(axis=1)
    return np.sqrt(np.sum(np.square(x[:,0:(N-1)] - x[:, 1:N]),1) / (N-1)) / S

def plot(title, xLabel, yLabel, xData, yData, cpt, xErr=None):
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.figure(figsize=(8, 6))
    plt.title( '%s' % ( title ) )
    plt.xlabel('%s' % (xLabel), fontsize=16)
    plt.ylabel('%s' % (yLabel), fontsize=16)

    cellCapture = OrderedDict((('6', 'red'), ('18', 'green'), ('30', 'blue'), ('42', 'orange')))
    color_map = [0 for i in range(len(cpt))]

    for i in range(0, len(cpt)):
        if cpt[i] == 6:
            color_map[i] = 'red'
        elif cpt[i] == 18:
            color_map[i] = 'green'
        elif cpt[i] == 30:
            color_map[i] = 'blue'
        else:
            color_map[i] = 'orange'
    # print(cellCapture)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', ms=10, linestyle='') for color in cellCapture.values()]

    plt.scatter(xData, yData, 100, c=color_map, alpha=0.6)
    if xErr is not None:
        plt.errorbar(xData, yData, xerr=xErr, fmt='none', marker='none', ecolor=color_map)

    l = plt.legend(markers, cellCapture.keys(), numpoints=1, title='Capture', bbox_to_anchor=(1.1, 0.5), loc=10, fontsize=16)
    plt.setp(l.get_title(), fontsize=16)

def plot_comparison(plotDf, dataset='Windram'):
    title = 'Comparision to the DeLorean Model'
    xLabel = 'Number of inducing points'

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(16, 6))

    ax[0].plot(plotDf.inducingPoints, plotDf['sMean'], linestyle='--', color='r', linewidth=3.)
    ax[0].plot(plotDf.inducingPoints, plotDf['sMean'], 'rs', markersize=8)
    ax[0].plot(plotDf.inducingPoints, plotDf['sBest'], '--', color='r', linewidth=3.)
    ax[0].plot(plotDf.inducingPoints, plotDf['GPLVM_avg'], color='g', linewidth=3.)
    ax[0].plot(plotDf.inducingPoints, plotDf['GPLVM_best'], 'go', markersize=8)
    ax[0].plot(plotDf.inducingPoints, plotDf['GPLVM_best'], color='g', linewidth=3.)
    ax[0].set_ylabel('Spearman Correlation', fontsize=16)

    ax[1].plot(plotDf.inducingPoints, plotDf['timeDeLorean'], linestyle='--', color='r', linewidth=2.5)
    ax[1].plot(plotDf.inducingPoints, plotDf['GPLVM_fitting_time'], color='g', linewidth=2.5)
    ax[1].set_ylabel('Fitting time (s)', fontsize=16)

    plt.suptitle(title, fontsize=20)
    fig.text(0.5, 0.04, xLabel, ha='center', va='center', fontsize=16)
    plt.xticks(plotDf.inducingPoints)

    blue_line = mlines.Line2D([], [], color='green', linewidth=3., label='BGPLVM(Best)')
    red_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=3., label='DeLorean(Best)')
    dashed1 = mlines.Line2D([], [], color='red', marker='s', markersize=8, linestyle='--', linewidth=3.,
                            label='DeLorean(Avg)')
    dashed2 = mlines.Line2D([], [], color='green', marker='o', markersize=8, linewidth=3., label='BGPLVM(Avg)')

    l1 = plt.legend(handles=[red_line, blue_line, dashed1, dashed2], numpoints=1, bbox_to_anchor=(-0.4, 0.4), loc=10,
                    fontsize=12)

    red_line_dotted = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2.5, label='DeLorean')
    green_line_solid = mlines.Line2D([], [], color='green', linewidth=2.5, label='BGPLVM')
    l2 = plt.legend(handles=[red_line_dotted, green_line_solid], numpoints=1, bbox_to_anchor=(0.8, 0.4), loc=10,
                    fontsize=12)

    fig.gca().add_artist(l1)
    fig.gca().add_artist(l2)

def plot_fitting_time_comparison(plotDf):
    plt.figure(figsize=(8, 8))
    plt.plot(plotDf['inducingPoints'], plotDf['timeDeLorean'], linestyle='--', linewidth=2.5, color='r')
    plt.plot(plotDf['inducingPoints'], plotDf['GPLVM_fitting_time'], linewidth=2.5, color='g')
    _ = plt.ylabel('Fitting time (s)', fontsize=16)
    _ = plt.xlabel('Number of inducing points', fontsize=16)
    _ = plt.xticks(plotDf['inducingPoints'], fontsize=12)
    _ = plt.yticks(fontsize=12)

    green_line = mlines.Line2D([], [], color='green', linewidth=2.5, label='BGPLVM')
    red_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2.5, label='DeLorean')

    _ = plt.legend(handles=[red_line, green_line], bbox_to_anchor=(1.21, 0.5), loc=10, fontsize=20, frameon=False)
