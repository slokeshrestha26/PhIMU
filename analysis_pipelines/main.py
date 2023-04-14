import os
import numpy as np 

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import pandas as pd
import feature_extraction as feat_ex
import dataset

#remove warnings
import warnings
warnings.filterwarnings("ignore")

ITERATION = 0 #numbering the experiment so that saving confusion matrices is easier
EXTRA_COMMENTS = "Building classifier after solving a bug that mismatched different channels in the dataset."
classes_of_interest = [dataset.labels["swipe_left_to_right"], \
                       dataset.labels["null_class"],\
                        dataset.labels["scrolling"],\
                        dataset.labels["portrait_tap"],\
                        dataset.labels["swipe_right_to_left"]
                        ]


def run_experiment(X_train, Y_train, X_test, Y_test, clf, clf_name):

    pipe = Pipeline([("scaler", StandardScaler()), 
                     (clf_name, clf)])

    pipe.fit(X_train,Y_train)
    disp = ConfusionMatrixDisplay(np.round(confusion_matrix(Y_test, pipe.predict(X_test),  normalize= "true"),2), display_labels=pipe.classes_)
    f_scr = f1_score(Y_test, pipe.predict(X_test), average = 'weighted')
    print("F1_score: ", f_scr)
    
    fig, ax = plt.subplots(figsize=(15, 13))

    # import pdb; pdb.set_trace()
    disp.plot(ax = ax)
    #make x and y font size bigger
    plt.xticks(rotation = 90)
    plt.title("{} | f1_score: {}".format(clf_name, f_scr))

def lopo_split(features):
    """ Leave one participant out splitting generator """
    # return index of left out participant
    parts = np.unique(features["participant"])
    for part in parts:
        left_out_idx = np.where(features["participant"] == part)[0]
        # return index of all other participants
        other_idx = np.where(features["participant"] != part)[0]

        # returns left_out_idx, other_idx, left out participant name, rest of participants
        yield left_out_idx, other_idx, part, [x for x in parts if x != part]

def savefig_util(dir, fname):
    """Handle FileNotFoundError by creating the directory"""
    try:
        #save figure by increasing figure size
        plt.savefig(dir+fname)
    except FileNotFoundError:
        os.makedirs(dir)
        plt.savefig(dir+fname)

def log_notes_util(f, parts, feat_ex):
    """Utility function to write notes to experiment directory"""
    f.write("This was confusion matrix results after leave one participant out scheme training with all the following subjects:\n")
    f.write("Participants: {} \n".format(str(parts)))

    f.write("Frame Window Size: {} \n".format(feat_ex.FEATURE_WINDOW_LEN))
    f.write("Slide Parameter: {} \n".format(feat_ex.SLID_PARAM))
    f.write("Moving Averae Window: {} \n".format(feat_ex.SMOOTHING_AV_WIND_SIZE))
    f.write("THROW_N: {} \n".format(feat_ex.THROW_N))
    f.write("EXTRA NOTES: " + EXTRA_COMMENTS)


def log_notes(dir, parts, feat_ex):
    """Log notes in a text file"""
    #handle FileNotFound exception
    try:
        with open(dir, "w") as f:
            log_notes_util(f, parts, feat_ex)
    except FileNotFoundError:
        os.makedirs(dir)
        with open(dir, "w") as f:
            log_notes_util(f, parts, feat_ex)

if __name__ == "__main__":
    """Levae one participant out scheme """

    #if features.csv exists, load them
    if os.path.exists("features.csv"):
        features = pd.read_csv("features.csv")
    else:
        f_e = feat_ex.Feature_Extractor()
        f_e.calculate_features()
        features = f_e.features
        #save features to csv
        features.to_csv("features.csv", index = False)

    # only get features with classes of interest
    features = features[features["label"].isin(classes_of_interest)]
    
    for left_out_idx, other_idx, left_out_part, rest_parts in lopo_split(features):
        X_train = features.iloc[other_idx, :-2].to_numpy()
        Y_train = features.iloc[other_idx, -1].to_numpy()

        X_test = features.iloc[left_out_idx, :-2].to_numpy()
        Y_test = features.iloc[left_out_idx, -1].to_numpy()

        run_experiment(X_train, Y_train, X_test, Y_test, RandomForestClassifier(), "random_forest")
        savefig_util("Confusion Matrices{}/random_forest/".format(ITERATION), "{}.png".format(left_out_part))

        run_experiment(X_train, Y_train, X_test, Y_test, LogisticRegression(max_iter=1000), "logistic_regression")
        savefig_util("Confusion Matrices{}/logistic_regression/".format(ITERATION), "{}.png".format(left_out_part))
        
        run_experiment(X_train, Y_train, X_test, Y_test, MLPClassifier(random_state = 1, max_iter = 1000), "mlp")
        savefig_util("Confusion Matrices{}/mlp/".format(ITERATION), "{}.png".format(left_out_part))
        
        log_notes("Confusion Matrices{}/notes.txt".format(ITERATION), rest_parts, feat_ex)