"""Testing segmentation algorithm for scrolling activity. Contains visualizations to make sense of the segmentation"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SF = 100 #SAMPLE FREQUENCY
FRAME_SIZE = 0.2 # 100ms
STRIDE = 0.1 # 50% overlap
THRESHOLD_RMS = 0.11


data = pd.read_csv("dataset/six_scrolls.csv", index_col= False)
# only get accelerometer data and gyro data
data = pd.concat([data.loc[:,"accelerometerAccelerationX(G)": "accelerometerAccelerationZ(G)"], data.loc[:,"gyroRotationX(rad/s)": "gyroRotationZ(rad/s)"]], axis=1)


def mean_normalize(data):
    return data - np.mean(data)
def rms_check(data, threshold):
    # calculate rms
    rms_acc_x = np.sqrt(np.mean(\
        mean_normalize(data.loc[:, "gyroRotationX(rad/s)"])**2)
        )
    rms_acc_y = np.sqrt(np.mean(\
        mean_normalize(data.loc[:, "gyroRotationY(rad/s)"])**2)
        )
    rms_acc_z = np.sqrt(np.mean(\
        mean_normalize(data.loc[:, "gyroRotationZ(rad/s)"])**2)
        )

    print("RMS Value: {}".format(np.max([rms_acc_x, rms_acc_y, rms_acc_z])))
    if(np.max([rms_acc_x, rms_acc_y, rms_acc_z]) > threshold):
        return True
    return False

def vis_segmentation(data, frame_size, stride = STRIDE):
    # visualize the segmentation of the data
    start = 0
    end = frame_size*SF
    while end <= len(data):
        #plot the frame of data
        data.iloc[int(start):int(end)].plot()
        print(rms_check(data.iloc[int(start):int(end)], THRESHOLD_RMS))
        plt.show()
        start += stride*SF
        end += stride*SF
       

if __name__ == "__main__":
    vis_segmentation(data, FRAME_SIZE)