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

import util

#remove warnings
import warnings
warnings.filterwarnings("ignore")

ITERATION = 0 #numbering the experiment so that saving confusion matrices is easier
EXTRA_COMMENTS = "Building classifier after solving a bug that mismatched different channels in the dataset."
classes_of_interest = ["swipe_left_to_right", "null_class", "scroll", "portrait_tap", "swipe_right_to_left"]


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
    f.write("This was confusion matrix results after leave one participant out scheme training with the following subjects:\n")
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

    participants = os.listdir("dataset/")
    participants = participants[1:]

    # Leave one participant out
    for i in range(len(participants)):
        left_out_part = participants[i]
        train_part = participants[0:i] + participants[i+1:]
        print("Left out participants: {}. Training participants: {}".format(left_out_part, train_part))
        
        feat_ex = util.Feature_Extractor(FEATURES_WINDOW_LEN = 2, TRAIN_SET= train_part, TEST_SET=left_out_part)
        X_train, Y_train, X_test, Y_test = feat_ex.load_features()

        #restrict the classes to only scrolling, typing, swiping, and null for now
        filt_train = Y_train.isin(classes_of_interest)
        filt_test = Y_test.isin(classes_of_interest)
        
        Y_train, X_train = Y_train.loc[filt_train], X_train.loc[filt_train, :]
        Y_test, X_test = Y_test.loc[filt_test], X_test.loc[filt_test, :] 
    
        
        run_experiment(X_train, Y_train, X_test, Y_test, RandomForestClassifier(), "random_forest")
        savefig_util("Confusion Matrices{}/random_forest/".format(ITERATION), "{}.png".format(left_out_part))

        run_experiment(X_train, Y_train, X_test, Y_test, LogisticRegression(max_iter=1000), "logistic_regression")
        savefig_util("Confusion Matrices{}/logistic_regression/".format(ITERATION), "{}.png".format(left_out_part))
        
        run_experiment(X_train, Y_train, X_test, Y_test, MLPClassifier(random_state = 1, max_iter = 1000), "mlp")
        savefig_util("Confusion Matrices{}/mlp/".format(ITERATION), "{}.png".format(left_out_part))
    
    log_notes("Confusion Matrices{}/notes.txt".format(ITERATION), participants, feat_ex)