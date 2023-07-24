import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

participant = "Sloke0"
SMOOTHING_AV_WIND_SIZE = 0.10 #seconds #if you want to disable it, put it to 0.01 which is 1/SMPL_FREQ
SMPL_FREQ = 100 #Hz

def smoothing_average(df):
        window_size = SMOOTHING_AV_WIND_SIZE*SMPL_FREQ
        weights = np.repeat(1.0, window_size) / window_size

        smoothed_data = np.convolve(weights, np.array(df), 'valid')
        return (list(range(len(smoothed_data))), smoothed_data)
     

def vis_util_subplot(data, data_name):
    acc_title = ["acc x", "acc y", "acc z"]
    gyro_title = ["gyro x", "gyro y", "gyro z"]
    # get 3* 2 subplots
    fig, axs = plt.subplots(3 ,2 , figsize=(15, 13))
    #get 1,1
    ax = axs[0,0]
    #plot
    for i in range(3):
        ax = axs[i,0]
        x, y = smoothing_average(data.iloc[:,3 + i])
        ax.plot(x,y,\
             linewidth=0.5)
        ax.set_title(acc_title[i] + " " + data_name)

    #plotting magnitude
    # ax = axs[3,0]
    # ax.plot(list(range(len(data))), data.iloc[:,3]**2 + data.iloc[:,4]**2 + data.iloc[:,5]**2, linewidth=0.5)
    # ax.set_title("magnitude acc" + " " + data_name)

    for i in range(3):
        ax = axs[i,1]
        x, y = smoothing_average(data.iloc[:,7 + i])
        ax.plot(x,y,\
             linewidth=0.5)
        ax.set_title(gyro_title[i] + " " + data_name)

    #plotting magnitude
    # ax = axs[3,1]
    # ax.plot(list(range(len(data))), data.iloc[:,7]**2 + data.iloc[:,8]**2 + data.iloc[:,9]**2, linewidth=0.5)
    # ax.set_title("magnitude gyro" + " " + data_name)


def run_experiment_0():
    portrait_hod = pd.read_csv("dataset/{}/2023-02-17_13_37_49_portrait_hold.csv".format(participant), index_col= False)
    landscape_hold = pd.read_csv("dataset/{}/2023-02-17_13_39_49_landscape_hold.csv".format(participant), index_col= False)
    # scrolling = pd.read_csv("dataset/{}/2023-02-17_13_54_54_scroll.csv".format(participant), index_col= False)
    # swiping_right_to_left = pd.read_csv("dataset/{}/2023-02-17_13_52_39_swipe_right_to_left.csv".format(participant), index_col= False)
    # swiping_left_to_right = pd.read_csv("dataset/{}/2023-02-17_13_51_00_swipe_left_to_right.csv".format(participant), index_col= False)
    # taking_call = pd.read_csv("dataset/{}/2023-02-17_13_47_51_taking_call.csv".format(participant), index_col= False)
    # portrait_tap = pd.read_csv("dataset/{}/2023-02-17_14_02_20_portrait_tap.csv".format(participant), index_col= False)
    # tap_while_on_table = pd.read_csv("dataset/{}/2023-02-17_14_00_47_tap_while_on_table.csv".format(participant), index_col= False)

    data_list = [(portrait_hod, "portrait_hold"),\
                    (landscape_hold, "landscape_hold")]
                # (scrolling, "scrolling"),\
                #   (swiping_right_to_left, "swiping_right_to_left"),\
                #       (swiping_left_to_right, "swiping_left_to_right"), \
                #         (taking_call, "taking_call"),\
                #               (portrait_tap, "portrait_tap"), \
                #                 (tap_while_on_table, "tap_while_on_table")]
    for data, data_name in data_list:
        vis_util_subplot(data, data_name)
    
def run_experiment_1(filename, modality):
    modality_map = {"acc_x": "accelerometerAccelerationX(G)",
                    "acc_y": "accelerometerAccelerationY(G)",
                    "acc_z": "accelerometerAccelerationZ(G)",
                    "gyro_x": "gyroRotationX(rad/s)",
                    "gyro_y": "gyroRotationY(rad/s)",
                    "gyro_z": "gyroRotationZ(rad/s)"}
    
    # data = pd.read_csv("dataset/{}/{}".format(participant,filename), index_col= False)
    data = pd.read_csv("dataset/{}".format(filename), index_col= False)
    # import pdb; pdb.set_trace()
    modality_list = [modality_map[m] for m in modality]
    plt.plot(data.loc[:, modality_list], linewidth=0.5)
    # plt.title(modality + " " + filename)


if __name__ == "__main__":
    run_experiment_0()
    # run_experiment_1("Sloke0/2023-02-17_13_37_49_portrait_hold.csv", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
    # run_experiment_1("Sloke0/2022-11-14_13_38_44_swipe_left_to_right.csv", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
    # run_experiment_1("Sloke0/2022-11-14_13_52_33_scroll.csv", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])

    #tight layout
    plt.tight_layout()
    plt.show()