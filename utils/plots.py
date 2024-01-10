import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import lines as mlines

def caltrendwithCI(probability_signal):

    # Authors: Saeed Montazeri / Jukka Ranta

    nClasses = 3

    # Compute BT as a weighted average of classes
    # Adjust the weights based on the number of classes
    A = np.multiply(np.array([0.5, 2, 3.5]), probability_signal)
    weightedavg = np.sum(A, axis=1)

    lowerlim = np.zeros((probability_signal.shape[0],1))
    upperlim = np.zeros((probability_signal.shape[0],1))
    classes = np.array([1, 2, 3])

    for inx in range(probability_signal.shape[0]):
        # Interpolate probability of the estimated class
        interpprob = np.interp(weightedavg[inx], classes, probability_signal[inx, :])
        modalC = np.around(weightedavg[inx])  # The modal class

        # Updating probability value for modal class with the distance between
        # interpolated probability of the weighted average class and
        # probability of the modal class

        #NOTE: Fixed indexing from prior python implementation
        probability_signal[inx, int(modalC) - 1] = probability_signal[inx, int(modalC) - 1] - interpprob

        # Upper and lower boundaries as the sum of weighted probabilities
        if modalC == nClasses:
            pLowers = np.sum(np.multiply(classes[0:2], probability_signal[inx, 0:2]))
            pUppers = classes[0] * probability_signal[inx, nClasses - 1]
        elif modalC == 1:
            pLowers = classes[0] * probability_signal[inx, 0]
            pUppers = np.sum(np.multiply(classes[0:2], probability_signal[inx, 1:3]))
        else:
            if modalC >= weightedavg[inx]:
                pLowers = classes[0] * probability_signal[inx, 0]
                pUppers = np.sum(np.multiply(classes[0:2], probability_signal[inx, 1:3]))
            else:
                pLowers = np.sum(np.multiply(classes[0:2], probability_signal[inx, 0:2]))
                pUppers = classes[0] * probability_signal[inx, 2]

        lowerlim[inx] = weightedavg[inx] - np.absolute(pLowers)
        upperlim[inx] = weightedavg[inx] + np.absolute(pUppers)

    # Thresholding
    l_idx=np.argwhere(lowerlim < 0.5)
    lowerlim[l_idx] = 0.5
    u_idx=np.argwhere(upperlim > nClasses+0.5)
    upperlim[u_idx]= nClasses+0.5

    return weightedavg, lowerlim, upperlim



def plot_time_series(labels, predicted_probabilities,
                      predicted_classes, class_names, titles=('Hypnogram', 'Sleep Depth Trend')):

    time = np.arange(0, labels.shape[0]) * 30 / 60 / 60

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    fig.tight_layout(pad=5.0)

    # Plot discrete hypnogram
    ax1.set_title(titles[0])
    ax1.step(time, labels, label='Ground Truth', color='C0', linewidth=3.0)
    ax1.step(time, predicted_classes, label='Classifier Prediction', color='red', linestyle='dashed')
    ax1.set_yticks(range(1, len(class_names) + 1), class_names)

    ax1.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower right")
    ax1.grid(True)
    ax1.autoscale(enable=True, axis='x', tight=True)

    # Plot SDT
    [weightedavg, lowerlim, upperlim] = caltrendwithCI(predicted_probabilities)
    ax2.set_title(titles[1])
    ax2.plot(time, weightedavg, color='C0', linewidth=2, label='Weighted Average')
    ax2.fill_between(time, upperlim[:,0], lowerlim[:,0], color='C0', alpha=0.3, label='Confidence Interval')
    ax2.axhline(y=1.5, color='black', linestyle='-', zorder=1)
    ax2.axhline(y=2.5, color='black', linestyle='-', zorder=1)

    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylim(0.5, 3.5)
    ax2.set_yticks(range(1, len(class_names) + 1), class_names)
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower right")

    ax2.set_xlabel('Time (hours)')

    return fig


def plot_confusion_matrix(cm, class_names, title='Confusion matrix'):

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    cm_percentages = (cm / cm.sum(axis=0))
    labels = np.asarray([f"{percent:.1%}\nn = {count}" for count,
                          percent in zip(cm.flatten(),
                          cm_percentages.flatten())]).reshape(cm.shape)

    palette = sns.cubehelix_palette(start=.2, rot=-.2, dark=0, light=.95, as_cmap=True)

    sns.heatmap(cm_percentages, annot=labels, fmt='', cmap=palette,
                cbar=True, vmax=1, vmin=0)

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .50, .75, 1])
    cbar.set_ticklabels(['0', '25%', '50%', '75%', '100%'])
    ax.set_xlabel("True State")
    ax.set_ylabel("Predicted State")

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_title(title)

    return fig

def box_plot(accuracies, MCCs, title):
  fig = plt.figure(figsize=(7,6))
  for i, dataset in enumerate([accuracies, MCCs], start=1):
      y = dataset
      x = [i] * len(y)
      scatter = plt.scatter(x, y, alpha=0.7, color='grey', s=10, label='Subject')

  bp = plt.boxplot([accuracies, MCCs], showmeans=False, meanline=False, widths=0.2)
  plt.xticks([1, 2], ['Accuracy', 'MCC'])

  median_line = mlines.Line2D([], [], color='C0', label='Median')
  outlier_point = mlines.Line2D([], [], color='none', marker='o',
                                markerfacecolor='none',
                                markeredgecolor='black', label='Outlier')

  for median in bp['medians']:
      median.set(color='C0', linewidth=1.5)

  plt.legend(handles=[median_line, outlier_point, scatter], loc='lower left')
  plt.yticks(np.arange(0, 1 + 0.1, 0.1))
  plt.ylim(0, 1)
  plt.ylabel('Performance Score')
  plt.title(title)

  return fig

def plot_learning_curves(all_train_metrics, all_test_metrics, num_epochs, fold, final_plot):

    epochs = range(1, num_epochs + 1)
    
    train_color, test_color = 'cornflowerblue', 'lightcoral'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if final_plot:
        train_loss_mean = np.mean(all_train_metrics[0,:,:], axis=0)       
        train_mcc_mean = np.mean(all_train_metrics[2,:,:], axis=0)

        test_loss_mean = np.mean(all_test_metrics[0,:,:], axis=0)
        test_mcc_mean = np.mean(all_test_metrics[2,:,:], axis=0)


        ax1.plot(epochs, train_loss_mean,
                  label='Mean Train Loss', color=train_color)      
        ax1.plot(epochs, test_loss_mean,
                  label='Mean Test Loss', color=test_color)
     
        ax2.plot(epochs, train_mcc_mean,
                  label='Mean Train MCC', color=train_color)            
        ax2.plot(epochs, test_mcc_mean,
                  label='Mean Test MCC', color=test_color)
        
    else:
        ax1.plot(epochs, all_train_metrics[0, fold, :],
                label='Train loss', color=train_color)
        ax1.plot(epochs, all_test_metrics[0, fold, :],
                label='Test loss', color=test_color)

        ax2.plot(epochs, all_train_metrics[2, fold, :],
                label='Train MCC', color=train_color)
        ax2.plot(epochs, all_test_metrics[2, fold, :],
                label='Test MCC', color=test_color)

    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')

    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Score')

    ax1.grid(True)
    ax2.grid(True)
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    ax2.set_ylim([0,1])
    ax2.set_yticks(np.arange(0,1.1, step=0.1))

    return fig