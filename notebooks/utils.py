import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import  numpy as np
import  pandas as pd
from collections import OrderedDict
from cycler import cycler
import warnings
warnings.filterwarnings('ignore')

matplotlib.style.use('ggplot')
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['axes.edgecolor'] = 'black'
# plt.rc('axes', color_cycle=['royalblue', 'orange', 'green', 'red', 'blueviolet', 'sienna', 'hotpink', 'gray', 'y', 'c'])
# plt.rc('axes', color_cycle=['royalblue', 'green', 'sienna', 'c', 'orange', 'red', 'blueviolet', 'hotpink', 'gray', 'y'])
plt.rc('axes', prop_cycle=cycler(color=['royalblue', 'green', 'sienna', 'c', 'orange', 'red', 'blueviolet', 'hotpink', 'gray', 'y']))
# axes.prop_cycle : cycler('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k'])

def getari_for_latent_space(X, truelabels):
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import adjusted_rand_score
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    kmeans.labels_ = kmeans.labels_ + 1
    ARI = adjusted_rand_score(truelabels, kmeans.labels_)
    return ARI

def calcroughness(x, pt):
    x=np.atleast_2d(x)
    i=np.argsort(pt)
    x = x[:,i]
    N = x.shape[1]
    assert(N > 0)
    S = x.std(axis=1)
    return np.sqrt(np.sum(np.square(x[:,0:(N-1)] - x[:, 1:N]),1) / (N-1)) / S

def cbtime_to_tau(pTime, startTime, endTime, timeDiff):
    t = pTime * (endTime - startTime) / 100.
    t = t + (startTime + timeDiff / 2.)

    if t >= endTime:
        t = t - (endTime - startTime)
    return t


def tau_to_cbtime(tau, startTime, endTime, timeDiff):
    t = tau - (startTime + timeDiff / 2.)
    if t <= 0.:
        t = t + (endTime - startTime)
    t = t * 100. / (endTime - startTime)
    if t > 100:
        t = np.abs(100 - t)
    return t

def plot(title, xLabel, yLabel, xData, yData, cpt, xErr=None, **kwargs):
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    # plt.figure(figsize=(8, 6))
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

    if 'datset' in kwargs:
        cellCapture = OrderedDict((('0', 'red'), ('2', 'green'), ('4', 'blue'), ('7', 'orange')))
        for i in range(0, len(cpt)):
            if cpt[i] == 1:
                color_map[i] = 'red'
            elif cpt[i] == 2:
                color_map[i] = 'green'
            elif cpt[i] == 3:
                color_map[i] = 'blue'
            else:
                color_map[i] = 'orange'

    # print(cellCapture)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', ms=10, linestyle='') for color in cellCapture.values()]

    plt.scatter(xData, yData, 100, c=color_map, alpha=0.6)
    if xErr is not None:
        plt.errorbar(xData, yData, xerr=xErr, fmt='none', marker='none', ecolor=color_map)

    # l = plt.legend(markers, cellCapture.keys(), numpoints=1, title='Capture', bbox_to_anchor=(1.1, 0.5), loc=10, fontsize=16)
    l = plt.legend(markers, cellCapture.keys(), numpoints=1, title='Capture', loc=4, fontsize=16)
    plt.setp(l.get_title(), fontsize=16)
    # plt.show()

def plot_comparison(plotDf, dataset='Windram'):
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'

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
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
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

def plot_genes(pseudotimes, geneProfiles, geneData, cpt, prediction):
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'Gray'
    plt.rc('xtick', labelsize=15)

    startTime = 1.
    endTime = 3.55
    timeDiff = 0.85

    selectedGenes = geneProfiles.keys().values
    cbPeaktime = np.zeros(len(selectedGenes))
    for g in range(0, len(selectedGenes)):
        cbPeaktime[g] = cbtime_to_tau(geneData[selectedGenes[g]].cbPeaktime, startTime, endTime, timeDiff)
        # print(geneData[selectedGenes[g]].cbPeaktime)

    Xnew = prediction[0]
    meanDf = prediction[1]
    varDf = prediction[2]
    # Create a Dataframe to contain predictive mean and variance
    predictDf = {}
    for i in range(len(selectedGenes)):
        predictDf[selectedGenes[i]] = pd.DataFrame({'mean': meanDf[selectedGenes[i]], 'var': varDf[selectedGenes[i]]})

    # Plot the result
    title = 'McDavid'
    xLabel = 'Pseudotime'
    yLabel = 'Expression'

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 8))

    # plt.suptitle(title, fontsize=16)
    fig.text(0.5, -0.04, xLabel, ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, yLabel, ha='center', va='center', rotation='vertical', fontsize=20)

    xValues = np.array([1., 1.85, 2.7, 3.55])
    xString = np.array(['G2/M', 'G0/G1', 'S', 'G2/M'])
    plt.xticks(xValues, xString)
    plt.xlim(1., 3.55)

    # Following codes are used just to add legends
    cellCycleStages = {'g0/g1': u'red', 's': u'green', 'g2/m': u'blue'}
    stageColorCodes = ['red', 'green', 'blue']
    color_map = [stageColorCodes[cpt[i] - 1] for i in range(len(cpt))]
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', markersize=9, linestyle='') for color in
               cellCycleStages.values()]
    l = plt.legend(markers, cellCycleStages.keys(), numpoints=1, title='Capture Stages', bbox_to_anchor=(1.6, 1.1),
                   loc=10, fontsize=20, frameon=False)
    plt.setp(l.get_title(), fontsize=20)

    n = 0
    for row in ax:
        for col in row:
            col.plot(Xnew[:, 0], predictDf[selectedGenes[n]]['mean'].values, 'black', lw=1)
            col.fill_between(Xnew[:, 0], predictDf[selectedGenes[n]]['mean'].values - \
                             2 * np.sqrt(predictDf[selectedGenes[n]]['var'].values),
                             predictDf[selectedGenes[n]]['mean'].values + \
                             2 * np.sqrt(predictDf[selectedGenes[n]]['var'].values), color='grey', alpha=0.5)
            col.scatter(pseudotimes, geneProfiles[selectedGenes[n]], 130, marker='.', c=color_map, alpha=0.6)
            col.set_title(selectedGenes[n], fontsize=16)
            col.axvline(cbPeaktime[n], linestyle='--', color='black')
            plt.setp(col.xaxis.get_majorticklabels(), rotation=90)
            col.yaxis.set_tick_params(labelsize=14)
            n = n + 1

def plotcorrelation(X, Y, title, data_labels):
    # plt.rcParams['axes.facecolor'] = 'white'
    # plt.rcParams['axes.edgecolor'] = 'black'
    # plt.rc('axes', color_cycle=['royalblue', 'orange', 'green', 'red', 'blueviolet', 'sienna', 'hotpink', 'gray', 'y', 'c'])

    legend_order = ['1', '2', '4', '8', '16', '32 ICM', '32 TE', '64 PE', '64 TE', '64 EPI']
    # label_order = ['1', '16', '2', '32 ICM', '32 TE', '4', '64 PE', '64 TE', '64 EPI', '8']
    yVals = np.array([1, 2, 4, 8, 16, 24, 32])
    yStrings = np.array(['1', '2', '4', '8', '16', '32', '64'])

    for l in legend_order:
        x = Y[data_labels == l]
        if x[0]==64.:
            x = [x[i] - 32 for i in range(0,len(x))]
        elif x[0] == 1.:
            x = [x[i] - 0. for i in range(0, len(x))]
        elif x[0] == 4.:
            x = [x[i] + 0. for i in range(0, len(x))]
        elif x[0] == 32.:
            x = [x[i] - 8. for i in range(0, len(x))]

        plt.scatter(X[data_labels == l], x, 100, label=l)
        plt.tick_params(labelsize=14)
        plt.yticks(yVals, yStrings)
        plt.xlabel('Pseudotime', fontsize = 20)
        plt.ylabel('Capture time', fontsize=20)
        plt.title(title, fontsize=20)
        l = plt.legend(loc="lower right", fontsize=14, ncol=2, title="Capture stages", borderaxespad=0., columnspacing=0.2, handletextpad=0.1)
        plt.setp(l.get_title(), fontsize=16)

def plot_XY(X, Y, title, data_labels, label_order=None, **kwargs):
    # plt.rcParams['axes.facecolor'] = 'white'
    # plt.rcParams['axes.edgecolor'] = 'black'
    # plt.rc('axes', color_cycle=['royalblue', 'orange', 'green', 'red', 'blueviolet', 'sienna', 'hotpink', 'gray', 'y', 'c'])

    if label_order is None:
        label_order = ['1', '2', '4', '8', '16', '32 ICM', '32 TE', '64 PE', '64 TE', '64 EPI']

    mSize = 100
    if 'ms' in kwargs:    mSize = kwargs.pop('ms')

    fsize = 16
    if 'fontsize' in kwargs:    fsize = kwargs.pop('fontsize')

    for l in label_order:
        plt.scatter(X[data_labels == l], Y[data_labels == l], mSize, label=l)
        xPos = np.median(X[data_labels == l])
        yPos = np.median(Y[data_labels == l])
        # if title != 'With prior' and l == '32 TE':
        #     xPos = xPos - 0.4
        # if title == 'With prior' and l == '64 TE':
        #     xPos = xPos + 0.2
        plt.text(xPos, yPos, l, fontsize=fsize, weight='bold')

    xlabel = 'GPLVM-1 (Pseudotime)'
    ylabel = 'GPLVM-2'
    if 'xlabel' in kwargs:  xlabel = kwargs.pop('xlabel')
    if 'ylabel' in kwargs:  ylabel = kwargs.pop('ylabel')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)

def correlation_dpt(xData, yData, cpt, ax, title, diagLine=False):
    cellCapture = OrderedDict((('0','red'), ('2','green'), ('4','blue'), ('7','orange')))
    color_map = [0 for i in range(len(cpt))]

    for i in range(0,len(cpt)):
        if cpt[i] == 1:
            color_map[i] = 'red'
        elif cpt[i] == 2:
            color_map[i] = 'green'
        elif cpt[i] == 3:
            color_map[i] = 'blue'
        else:
            color_map[i] = 'orange'

    markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in cellCapture.values()]

    # plt.figure(figsize=(5, 5))
    ax.scatter(xData, yData, 10, c=color_map)
    if diagLine:
        ax.plot( [0,0.7],[0,0.7], linewidth=3)
    _=plt.xticks(fontsize=14)
    _=plt.yticks(fontsize=14)
    ax.set_xlabel('BGPLVM Pseudotime', fontsize=16)
    ax.set_ylabel('Diffusion Pseudotime', fontsize=16)
    ax.set_title(title, fontsize=18)
    # plt.title("Correlation (No Prior) = %f"%(spearmanr(pTimes['pt_np_32_trun'].values, pTimes['dpt'].values)[0]), fontsize=20)
    # plt.xlabel('BGPLVM', fontsize=20)
    # plt.ylabel('DPT', fontsize=20)
    # l = plt.legend(markers, cellCapture.keys(), numpoints=1, title='Capture', bbox_to_anchor=(1.15, 0.5), loc=10, fontsize=16)
    ax.legend(markers, cellCapture.keys(), numpoints=1, title='Capture', fontsize=14, frameon=False)
    # ax.setp(l.get_title(), fontsize=14)

def plot_robustness_across_prior_variance(array_of_values, single_value, title, xlabel, ylabel):
    xVals = np.array([0.01, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    xStrings = np.array(['0.01', '0.1', '0.2', '0.5', '0.8', '1.', '1.5', '2.', '3.', '5.', '10.', '25.', '50.', '100.', 'No Prior'])

    plt.scatter(xVals, array_of_values, linewidth=5)
    plt.plot(xVals, array_of_values, linewidth=4)

    plt.ylim([0., 1.0])

    plt.xticks(np.append(xVals, 70), xStrings, rotation=90, fontsize=16)
    plt.yticks(fontsize=14)

    plt.axvline(x=5, ymax=array_of_values[1], linestyle='--', linewidth=5, c='black')
    plt.scatter(70, single_value, linewidth=5, c='red')
    plt.axvline(x=70, ymax=single_value, linestyle='--', linewidth=5, c='red')

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)