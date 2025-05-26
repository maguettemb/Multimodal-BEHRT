import pandas as pd
import numpy as np
from sklearn.metrics import *
import random
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#. 1. Follow training

def display_training_stats(data):  
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=data)
    
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
  
    # Display the table.
    return df_stats

def cal_mean(df_stats, feature, epochs):
    return [np.mean(df_stats.loc[x, feature]) for x in range(1,epochs+1)]

def cal_std(df_stats, feature, epochs):
    return [np.std(df_stats.loc[x, feature]) for x in range(1,epochs+1)]
     

def plot_history(training_stats, epochs):
    
    sn.set(style="whitegrid")
    fig, ((ax1, ax2)) = plt.subplots(1,2, figsize = (20,7))
    
    mean_f1_train = cal_mean(display_training_stats(training_stats), 'Average Train f1 score', epochs)
    std_f1_train = cal_std(display_training_stats(training_stats), 'Average Train f1 score', epochs)
    
    mean_f1_valid = cal_mean(display_training_stats(training_stats), 'Average Valid f1 score', epochs)
    std_f1_valid = cal_std(display_training_stats(training_stats), 'Average Valid f1 score', epochs)
    
    mean_loss_train = cal_mean(display_training_stats(training_stats), 'Training Loss', epochs)
    std_loss_train = cal_std(display_training_stats(training_stats), 'Training Loss', epochs)
    
    mean_loss_valid = cal_mean(display_training_stats(training_stats), 'Validation Loss', epochs)
    std_loss_valid = cal_std(display_training_stats(training_stats), 'Validation Loss', epochs)
    
    
    
    ax1.plot(range(epochs), mean_f1_train, 'b-', label='Train f1 score')
    ax1.fill_between(range(epochs), np.array(mean_f1_train) - np.array(std_f1_train), np.array(mean_f1_train) + np.array(std_f1_train), color='b', alpha=0.2)
    ax1.plot(range(epochs), mean_f1_valid, 'r--', label='Valid f1 score')
    ax1.fill_between(range(epochs), np.array(mean_f1_valid) - np.array(std_f1_valid), np.array(mean_f1_valid) + np.array(std_f1_valid), color='r', alpha=0.2)
   
    ax2.plot(range(epochs), mean_loss_train, 'b-', label='Train loss')
    ax2.fill_between(range(epochs), np.array(mean_loss_train) - np.array(std_loss_train), np.array(mean_loss_train) + np.array(std_loss_train), color='b', alpha=0.2)
    ax2.plot(range(epochs), mean_loss_valid, 'r--', label='Valid loss')
    ax2.fill_between(range(epochs), np.array(mean_loss_valid) - np.array(std_loss_valid), np.array(mean_loss_valid) + np.array(std_loss_valid), color='r', alpha=0.2)
   
    ax1.grid()
    ax2.grid()
    
    plt.legend()
    plt.show()
    return

## 2. Performances 

def plot_roc_auc(fpr, tpr, roc_auc, save_fig=False, fig_name=None):
    
    import matplotlib.pyplot as plt
    import sklearn.metrics as metrics
    
   
    sn.set_style("whitegrid")
    fig,ax = plt.subplots(figsize=(13, 9))
    
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot(fpr, tpr, '#067062', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if save_fig:
        plt.savefig(fig_name+'_roc_curve.png')
    plt.show()

def ConfusionMatrix(true, predicted, cmap = 'bwr', normalize = True, save_fig=False, fig_name=None):
    
    cm = confusion_matrix(true, predicted)

    df_cm = pd.DataFrame(cm)
    fig = plt.figure(figsize=(10,7))
    try:
        heatmap = sn.heatmap(df_cm, annot=True, cmap=cmap)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    if save_fig:
        fig.savefig(fig_name+'_cm.png')
    return 

def print_report(label, logits, save_fig=False, figname=None): 
    """
    Display information for the performances: the confusion matrix and the ROC-curve
    """
   
    # Compute micro-average ROC curve and ROC area
    label = np.argmax(label, axis=1)
    
    fpr, tpr, _ = sklearn.metrics.roc_curve(label, logits[:,1])
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    ## Plot ##  
    target_names = ['Non relapse', 'Relapse']

  #  print("f1 score is {:.4f}, Precision is {:.4f}, Recall is {:.4f}".format(f1, precision, recall))
    print(classification_report(label, np.argmax(output, axis=1), target_names=target_names))
    
    #. Display confusion matrix
    ConfusionMatrix(label, np.argmax(logits, axis=1), save_fig=save_fig, fig_name=fig_name)
   
    #. Display ROC curve
    plot_roc_auc(fpr, tpr, roc_auc)

    return fpr, tpr, roc_auc  

## Scores

def scores(logits, labels, training=False):
    """
    Compute scores for the training the valid and the test set:
    Take as input logits in probabilities and labels in a 2D array
    Return: f1score, recall, precision (average), average-precision-score and ROC-AUC score
    
    """
    
    if training:
        labels, logits = labels.cpu(), logits.detach().cpu()
    
    labels = np.argmax(labels, axis = 1)
     #. Let's compute first scores that consider probabilities: AUC-ROC and aps
    auc = roc_auc_score(labels, logits[:,1])
    aps = average_precision_score(labels, logits[:,1])
        
     #. We can now transform probabilities to label and compute the other scores
    logits = np.argmax(logits, axis=1)
    f1 = f1_score(labels, logits)
    recall = recall_score(labels, logits)
    precision = precision_score(labels, logits)
    
    return auc, aps, f1, recall , precision, logits, labels

## 3. MISC

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


""" Empty the cache to enable the use of the gpu """

def empty_cuda():
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    from sklearn.utils import gen_even_slices  


class StratifiedSampler(Sampler):
    """
    Over Sampling
    Provides equal representation of target classes in each batch
    Uses all the data
    Truly splits the data in batches like torch's BatchSampler
    """
    def __init__(self, X, y, batch_size, minority_pct=0.5):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.X = X
        self.y = torch.from_numpy(y.values)
        self.batch_size = batch_size
        self.minority_pct = minority_pct 
        # The number of splits/batches depends on how many samples
        # from the minority class we want to put in each batch
        self.n_splits = int(self.y.size(0) / (self.minority_pct * self.batch_size))

    def __iter__(self):        
        minority_idxs = np.where(self.y==1)[0]
        majority_idxs = np.where(self.y==0)[0]
        
        num_majority = len(majority_idxs)
        slices_generator = gen_even_slices(num_majority, self.n_splits)
        
        # Shuffle the majority class
        np.random.shuffle(majority_idxs)
                    
        # Random oversample of the whole minority class
        # to obtain as many samples as in the majority class
        minority_resampled = np.random.choice(minority_idxs, len(majority_idxs), replace=True)

        for sl in slices_generator:
            #print(sl)
            majority = majority_idxs[sl]
            minority = minority_resampled[sl]
            idxs = list(np.hstack([majority, minority]))
            yield idxs
     
    def __len__(self):
        # The iterator length is the number of batches
        return self.n_splits
