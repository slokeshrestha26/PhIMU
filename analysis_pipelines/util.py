from scipy.stats import skew, kurtosis
from numpy import sqrt, arctan, arccos, mean, array
import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

SF = 100 #SAMPLE FREQUENCY
FRAME_SIZE = 0.2 # 100ms
STRIDE = 0.1 # 50% overlap
THRESHOLD_RMS = 0.11

# Flags for debugging
VISUALIZE_RAW = False
VISUALIZE_FEATURES = False
#run the script in interactive mode to get the plots


if(VISUALIZE_FEATURES or VISUALIZE_RAW):
    import matplotlib.pyplot as plt

class Feature_Extractor():

    def __init__(self, FEATURES_WINDOW_LEN = FRAME_SIZE, SMOOTHING_AV_WIND_SIZE = 1/SF, SMPL_FREQ = SF, SLID_PARAM = STRIDE, THROW_N = 5, TRAIN_SET = [], TEST_SET = []):
        self.SMPL_FREQ = SMPL_FREQ #sampling frequency is 100 Hz

        self.SLID_PARAM = SLID_PARAM #stride param.
        self.FEATURE_WINDOW_LEN = FEATURES_WINDOW_LEN #window length in seconds


        self.THROW_N = THROW_N*SMPL_FREQ #thorw away THROW_N SAMPLES of data from begining and end of each signal file
        self.TRAIN_SET = TRAIN_SET #name of participant to use for training
        self.TEST_SET = TEST_SET #name of participant to use for testing
        self.SMOOTHING_AV_WIND_SIZE = SMOOTHING_AV_WIND_SIZE

    def calc_features_uitl(self, data, class_name, part_name):
        """
        Utility function to calculate the feature from a window of data
        data = time series data of a single window of data
        class_name = corresponding class name of the window of the data
        """
        
        acc_corr_matrix = (data.loc[:,["acc_x",
                                    "acc_y",
                                    "acc_z"]] - 
                        data.loc[:,["acc_x",
                                    "acc_y",
                                    "acc_z"]].mean()).corr()  #correlation after the mean was taken out
        
        acc_mag = sqrt(data.loc[:,"acc_x"]**2 + 
                    data.loc[:,"acc_y"]**2 + 
                    data.loc[:, "acc_z"]**2)  #magnitude of the acceleration vector
        


        gyro_mag = sqrt(data.loc[:,"gyro_x"]**2 + 
                    data.loc[:,"gyro_y"]**2 + 
                    data.loc[:, "gyro_z"]**2)
        
        
        
        orientation = arccos(data.loc[:,"acc_y"]/acc_mag)

        feature = {"mean_acc_x": data.loc[:, "acc_x"].mean(), 
                                "mean_acc_y": data.loc[:, "acc_y"].mean(), 
                                "mean_acc_z": data.loc[:, "acc_z"].mean(),
                                
                                "variance_acc_x": data.loc[:, "acc_x"].var(), 
                                "variance_acc_y": data.loc[:, "acc_y"].var(), 
                                "variance_acc_z": data.loc[:, "acc_z"].var(),
                                
                            "q1_accx": data.loc[:, "acc_x"].quantile(0.25), 
                                "q3_accx": data.loc[:, "acc_x"].quantile(0.75), 
                                
                                "q1_accy": data.loc[:, "acc_x"].quantile(0.25), 
                                "q3_accy": data.loc[:, "acc_y"].quantile(0.75),
                                
                                "q1_accz": data.loc[:, "acc_z"].quantile(0.25), 
                                "q3_accz": data.loc[:, "acc_z"].quantile(0.75),
                                
                            "corr_acc_xy": acc_corr_matrix.iloc[0,1], 
                                "corr_acc_xz": acc_corr_matrix.iloc[0,2], 
                                "corr_acc_yz": acc_corr_matrix.iloc[1,2],
                                
                            "skewness_acc_x": skew(data.loc[:, "acc_x"]),
                                "skewness_acc_y": skew(data.loc[:, "acc_y"]),
                                "skewness_acc_z": skew(data.loc[:, "acc_z"]),
                                
                            "kurtosis_acc_x": kurtosis(data.loc[:, "acc_x"]),
                                "kurtosis_acc_y": kurtosis(data.loc[:, "acc_y"]),
                                "kurtosis_acc_z": kurtosis(data.loc[:, "acc_z"]),
                                
                            "mean_magnitude_acc": acc_mag.mean(), 
                                "var_magnitude_acc" : acc_mag.var(), 
                                
                                "q1_mag_acc": acc_mag.quantile(0.25),  
                                "q3_mag_acc": acc_mag.quantile(0.75),  
                                
                                "skewness_magnitude_acc": skew(acc_mag), 
                                "kurtosis_magnitude_acc": kurtosis(acc_mag),
                                
                            "omega_accx": arctan((data.loc[:,"acc_x"].max() - 
                                                data.loc[:,"acc_x"].min())/
                                                (data.loc[:,"acc_x"].idxmax() - 
                                                data.loc[:,"acc_x"].idxmin())),
                                
                            "omega_accy": arctan((data.loc[:,"acc_y"].max() - 
                                                data.loc[:,"acc_y"].min())/
                                                (data.loc[:,"acc_y"].idxmax() - 
                                                data.loc[:,"acc_y"].idxmin())),
                                
                            "omega_accz": arctan((data.loc[:,"acc_z"].max() - 
                                                data.loc[:,"acc_z"].min())/
                                                (data.loc[:,"acc_z"].idxmax() - 
                                                data.loc[:,"acc_z"].idxmin())),
                                
                            "omega_mag": arctan((acc_mag.max() - acc_mag.min())/(acc_mag.idxmax() - acc_mag.idxmin())), 
                                
                            "mean_orientation": orientation.mean(),
                                
                                "variance_orientation": orientation.var(),
                                "q_1_orientation": orientation.quantile(0.25),
                                "q_3_orientation": orientation.quantile(0.75),
                                "mean_gyro_x": data.loc[:, "gyro_x"].mean(),
                             "mean_gyro_y": data.loc[:, "gyro_y"].mean(),
                             "mean_gyro_z": data.loc[:, "gyro_z"].mean(),
                          "var_gyro_x": data.loc[:, "gyro_x"].var(),
                             "var_gyro_y": data.loc[:, "gyro_y"].var(),
                             "var_gyro_z": data.loc[:, "gyro_z"].var(),
                          "skewness_gyro_x": skew(data.loc[:, "gyro_x"]),
                             "skewness_gyro_y": skew(data.loc[:, "gyro_y"]),
                             "skewness_gyro_z": skew(data.loc[:, "gyro_z"]),
                          "kurtosis_gyro_x": kurtosis(data.loc[:, "gyro_x"]),
                             "kurtosis_gyro_y": kurtosis(data.loc[:, "gyro_y"]),
                             "kurtosis_gyro_z": kurtosis(data.loc[:, "gyro_z"]), 
                            
                          "omega_gyro_x": arctan((data.loc[:,"gyro_x"].max() - 
                                              data.loc[:,"gyro_x"].min())/
                                             (data.loc[:,"gyro_x"].idxmax() - 
                                              data.loc[:,"gyro_x"].idxmin())),
                            
                        "omega_gyro_y": arctan((data.loc[:,"gyro_y"].max() - 
                                              data.loc[:,"gyro_y"].min())/
                                             (data.loc[:,"gyro_y"].idxmax() - 
                                              data.loc[:,"gyro_y"].idxmin())),
                            
                        "omega_gyro_z": arctan((data.loc[:,"gyro_z"].max() - 
                                              data.loc[:,"gyro_z"].min())/
                                             (data.loc[:,"gyro_z"].idxmax() - 
                                              data.loc[:,"gyro_z"].idxmin())),
                            
                          "mean_magnitude_gyro": gyro_mag.mean(),
                          "variance_magnitude_gyro": gyro_mag.var(),
                          "q1_magnitude_gyro": gyro_mag.quantile(0.25), 
                            "q3_magnitude_gyro": gyro_mag.quantile(0.75),
                          "skewness_magnitude_gyro": skew(gyro_mag),
                          "kurtosis_magnitude_gyro": kurtosis(gyro_mag),
                          "omega_magnitude_gyro": arctan((gyro_mag.max() - gyro_mag.min())/(gyro_mag.idxmax())),
    #                             "zero_crossing_rate": [],
    "rms_acc_x": self.get_rms(data.loc[:, "acc_x"]),
    "rms_acc_y": self.get_rms(data.loc[:, "acc_y"]),
    "rms_acc_z": self.get_rms(data.loc[:, "acc_z"]),
    "rms_gyro_x": self.get_rms(data.loc[:, "gyro_x"]),
    "rms_gyro_y": self.get_rms(data.loc[:, "gyro_y"]),
    "rms_gyro_z": self.get_rms(data.loc[:, "gyro_z"]),
                            "label": class_name                                  
                                        }
        
        return feature

    def get_rms(self, data):
        normalized_mean = self.mean_normalize(data)
        return np.sqrt(np.sum(normalized_mean**2))/len(normalized_mean)

    def mean_normalize(self, data):
        return data - np.mean(data)

    def rms_check(self, data, threshold):
        # calculate rms for each channel in gyroscope
        rms_acc_x = np.sqrt(np.mean(\
            self.mean_normalize(data.loc[:, "gyro_x"])**2)
            )
        rms_acc_y = np.sqrt(np.mean(\
            self.mean_normalize(data.loc[:, "gyro_y"])**2)
            )
        rms_acc_z = np.sqrt(np.mean(\
            self.mean_normalize(data.loc[:, "gyro_z"])**2)
            )

        # print("RMS Value: {}".format(np.max([rms_acc_x, rms_acc_y, rms_acc_z])))
        if(np.max([rms_acc_x, rms_acc_y, rms_acc_z]) > threshold):
            return True
        return False

    def separate_null_positive(self, data_frame, class_name):
        """ Given a frame of data, data_frame, assign the class label only if the
        data_frame passes the threshhold rule. Otherwise, assign the class label as null_class
        """
        if (self.rms_check(data_frame, THRESHOLD_RMS)): 
            return class_name
        return "null_class"
    
    def calc_features(self, data, class_name, part_name):
        """
        Calculates the features from the input time series.
        data = the 2 minutes time series data
        class_name = activity performed
        """

        features = pd.DataFrame({"mean_acc_x": [], "mean_acc_y": [], "mean_acc_z": [],
                                "variance_acc_x": [], "variance_acc_y": [], "variance_acc_z": [],
                                "q1_accx": [], "q3_accx": [], "q1_accy": [], "q3_accy": [], "q1_accz": [], "q3_accz": [],
                                "corr_acc_xy": [], "corr_acc_xz": [], "corr_acc_yz": [],
                                "skewness_acc_x": [],"skewness_acc_y": [],"skewness_acc_z": [],
                                "kurtosis_acc_x": [],"kurtosis_acc_y": [],"kurtosis_acc_z": [],
                                "mean_magnitude_acc": [], "var_magnitude_acc" : [], "q1_mag_acc": [], "q3_mag_acc": [], "skewness_magnitude_acc": [], "kurtosis_magnitude_acc": [],
                                "omega_accx": [],
                                "omega_accy": [],
                                "omega_accz": [],
                                "omega_mag": [], 
                                "mean_orientation": [], "variance_orientation": [], "q_1_orientation": [], "q_3_orientation": [],
                                                            "mean_gyro_x": [], "mean_gyro_y": [], "mean_gyro_z": [],
                              "var_gyro_x": [], "var_gyro_y":[], "var_gyro_z": [],
                              "skewness_gyro_x": [], "skewness_gyro_y": [], "skewness_gyro_z": [],
                              "kurtosis_gyro_x": [], "kurtosis_gyro_y": [], "kurtosis_gyro_z": [], 
                              "omega_gyro_x": [],
                            "omega_gyro_y": [],
                            "omega_gyro_z": [],
                              "mean_magnitude_gyro": [],
                              "variance_magnitude_gyro": [],
                              "q1_magnitude_gyro":[], "q3_magnitude_gyro":[],
                              "skewness_magnitude_gyro":[],
                              "kurtosis_magnitude_gyro":[],
                              "omega_magnitude_gyro": [],  
                  "rms_acc_x": [], "rms_acc_y": [], "rms_acc_z": [],
                    "rms_gyro_x": [], "rms_gyro_y": [], "rms_gyro_z": [],
                                "label": []                                         
                                            }) # NOTE omega = angle between maximum and minimum

        
        start_idx = 0
        end_idx = self.FEATURE_WINDOW_LEN*self.SMPL_FREQ

        if(VISUALIZE_RAW):
            frame_counter = 0
        while(end_idx <= len(data) - 1):
            # data segmentation and class assignment
            # TODO: implement the data segmentation and class assignment
            data_frame = data.iloc[int(start_idx) : int(end_idx), :].copy()
            segment_class = self.separate_null_positive(data_frame, class_name)
            feature = self.calc_features_uitl(data_frame, segment_class, part_name)

            if(VISUALIZE_RAW):
                # Code for visualizing the time series for a frame
                data_frame.plot(title = class_name + " " + str(frame_counter))
                print("drawing plot for a frame for class: " + class_name)
                frame_counter += 1 #vis code
                #save figure
                # if no directory found, make one
                if not os.path.exists("raw_signals_plots/" + "{}/".format(part_name) + class_name):
                    os.makedirs("raw_signals_plots/" + "{}/".format(part_name) + class_name)
                plt.savefig("raw_signals_plots/" + "{}/".format(part_name) + class_name + "/" + class_name + "_" + str(frame_counter) + ".png")
                
            ## append the feature to the feature window
            features = features.append(feature, ignore_index= True)
            
            start_idx += self.SLID_PARAM*self.SMPL_FREQ
            end_idx += self.SLID_PARAM*self.SMPL_FREQ

            # if (VISUALIZE_RAW):
            #     if frame_counter > 1:
            #         break
                
        return features

    def smoothing_average(self, df):
        window_size = self.SMOOTHING_AV_WIND_SIZE*self.SMPL_FREQ
        weights = np.repeat(1.0, window_size) / window_size

        smooth_data_acc_x = np.convolve(weights, np.array(df.iloc[:,0]), 'valid')
        smooth_data_acc_y = np.convolve(weights, np.array(df.iloc[:,1]), 'valid')
        smooth_data_acc_z = np.convolve(weights, np.array(df.iloc[:,2]), 'valid')

        smooth_data_gyro_x = np.convolve(weights, np.array(df.iloc[:,3]), 'valid')
        smooth_data_gyro_y = np.convolve(weights, np.array(df.iloc[:,4]), 'valid')
        smooth_data_gyro_z = np.convolve(weights, np.array(df.iloc[:,5]), 'valid')

        hmap = {"acc_x": smooth_data_acc_x,
                "acc_y": smooth_data_acc_y,
                "acc_z": smooth_data_acc_z,
                "gyro_x": smooth_data_gyro_x,
                "gyro_y": smooth_data_gyro_y,
                "gyro_z": smooth_data_gyro_z}

        df_smoothed = pd.DataFrame(hmap)

        return df_smoothed

    def drop_non_acc_gyro(self,df):
        df_acc = df.loc[:,"accelerometerAccelerationX(G)": "accelerometerAccelerationZ(G)"]
        df_gyro = df.loc[:, "gyroRotationX(rad/s)": "gyroRotationZ(rad/s)"]
        return pd.concat([df_acc, df_gyro], axis= 1)


    def load_features_util(self, participant):
        ## input training data
        files = os.listdir("dataset/{}".format(participant))
        files = list(filter(lambda f: f.endswith('.csv'), files))

        features = []
        for file in files:
            print("Reading data ... {}".format(file))
            data = pd.read_csv("dataset/{}/{}".format(participant,file), index_col= False)
            activity_name = file[20:len(file) - 4]

            data = self.drop_non_acc_gyro(data)
            data = data.iloc[self.THROW_N: - self.THROW_N] #throw away small portion of data from start and end

            data = self.smoothing_average(data)

            print("Extracting features for {} ...".format(activity_name))
            features.append(self.calc_features(data, activity_name, participant))
            
        XY = pd.concat(features)
        
        X, Y = XY.iloc[:,:-1], XY.iloc[:,-1]

        print("Done!!")
        
        return X, Y
    
    def load_features(self):
        #load features for each participants
        # NOTE: TEST AND TRAIN SPLIT CAN BE OPTIMIZED
        # ATM, FOR EACH FOLDS, DATA IS BEING READ AND FEATURES ARE BEING RECALCULATED. VERY SLOW.
        # CAN FILTER A PRECALCULATED PANDAS DATAFRAME BY THE PARTICIPANT NAME IF OPTIMIZED

        # Returns X_train, Y_train, X_test, Y_test pandas dataframes
        X_train, Y_train, X_test, Y_test = [], [], [], []
        for part in self.TRAIN_SET:
            X_train_part, Y_train_part = self.load_features_util(part)
            X_train.append(X_train_part)
            Y_train.append(Y_train_part)

        X_train, Y_train = pd.concat(X_train), pd.concat(Y_train)
        
        X_test, Y_test = self.load_features_util(self.TEST_SET)

        return X_train, Y_train, X_test, Y_test
    

# test
if __name__ == "__main__":
    import os
    participants = os.listdir("dataset/")
    participants = participants[1:]
    
    
    feat_ex = Feature_Extractor(FEATURES_WINDOW_LEN = 2, TRAIN_SET= participants[1:], TEST_SET=participants[0])
    X_train, Y_train, X_test, Y_test = feat_ex.load_features()  

    if(VISUALIZE_FEATURES):
        #plot histogram of features for each target class
        for targ in Y_train.unique():
            X_train[Y_train == targ].hist(figsize= (20,10))
            #get a figure handle
            fig = plt.gcf()
            plt.suptitle(targ)
            fig.tight_layout()
            plt.savefig("histograms/{} left out/{}.png".format(participants[0],targ))
