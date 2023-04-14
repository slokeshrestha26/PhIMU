"""Extract IMU and audio features from raw data"""
from scipy.stats import skew, kurtosis
from numpy import sqrt, arctan, arccos, mean, array
import os
import numpy as np
import pandas as pd

import librosa
from harmonic_percussive_filter import harmonic_percussive_filter
import dataset
from cnn_embeddings import YamNetEmbeddings

from sklearn.validation import Cro

import warnings
warnings.filterwarnings("ignore")

SF_IMU = 100 #SAMPLE FREQUENCY
FRAME_SIZE = 0.2 # 200ms
STRIDE = 0.1 # 50% overlap. 100ms
THRESHOLD_RMS_GYRO = 0.11 # threhoold used for gyro for gesture segmentation
SF_AUDIO_YMNET = 16000 # sample frequency of audio recorded
THROW_N = 5 # number of frames to throw at the beginning and end of the recording

class Feature_Extractor():
    """Class that ecapsulates feature extraction
    Call self.calculate_features() to load features from available dataset.
    
    Dataset: annotated data collected from sensor logger app"""
    def __init__(self, 
                 FRAME_SIZE = FRAME_SIZE, 
                 SMOOTHING_AV_WIND_SIZE = 1/SF_IMU, 
                 SMPL_FREQ_IMU = SF_IMU, 
                 SMPL_FREQ_AUDIO = SF_AUDIO_YMNET,
                 SLID_PARAM = STRIDE, 
                 THROW_N = THROW_N):
        
        self.FRAME_SIZE = FRAME_SIZE
        self.SMOOTHING_AV_WIND_SIZE = SMOOTHING_AV_WIND_SIZE
        self.SMPL_FREQ_IMU = SMPL_FREQ_IMU
        self.SMPL_FREQ_AUDIO = SMPL_FREQ_AUDIO
        self.SLID_PARAM = SLID_PARAM
        self.THROW_N = THROW_N
        self.features = self.get_feature_df()

    def get_feature_df(self):
        """Return a dataframe with features columns"""
        #IMU features
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
    
    # add 150 columns on the feature dataframe for audio features
        for i in range(YamNetEmbeddings.n_embeddings):
            features['audio_embedding ' + str(i)] = []

        return features


    def calculate_features(self):
        """Calculate features from raw data in the dataset and 
        accumulates them in a dataframe, self.features"""

        
        # iterate over all files in the dataset using os.walk
        for part in os.listdir("dataset"):
            # ignore .DS_Store
            if part == ".DS_Store": continue

            # iterate over all files in the current directory
            data_folders = os.listdir("dataset/" + part)
            for d_fldr in data_folders:
                # get the full path of the file
                if ".DS_Store" in d_fldr: continue
                file_path = "dataset/{}/{}".format(
                                            part, d_fldr)
                # read the files
                for file in os.listdir(file_path):
                    if (file == "Accelerometer.csv"):
                        print("Reading file: ", f'{file_path}/{file}')
                        acc = pd.read_csv(f'{file_path}/{file}')
                        # throw away first and last THROW_N seconds
                        acc = acc.iloc[
                            int(self.THROW_N * self.SMPL_FREQ_IMU): 
                            - int(self.THROW_N * self.SMPL_FREQ_IMU)
                        ]
                        acc.pop("time")
                        acc.pop("seconds_elapsed")
                    elif (file == "Gyroscope.csv"):
                        print("Reading file: ", f'{file_path}/{file}')
                        gyro = pd.read_csv(f'{file_path}/{file}')
                        gyro = gyro.iloc[
                            int(self.THROW_N * self.SMPL_FREQ_IMU): 
                            - int(self.THROW_N * self.SMPL_FREQ_IMU)
                        ]
                        gyro.pop("time")
                        gyro.pop("seconds_elapsed")
                    elif (file == "Microphone.caf"):
                        print("Reading file: ", f'{file_path}/{file}')
                        audio, _ = librosa.load(f'{file_path}/{file}', sr = self.SMPL_FREQ_AUDIO)
                        # throw away first and last THROW_N seconds
                        audio = audio[
                            int(self.THROW_N * self.SMPL_FREQ_AUDIO):
                            - int(self.THROW_N * self.SMPL_FREQ_AUDIO)
                        ]
                

                # get the label of the file
                label = dataset.labels[d_fldr[:-20]]
                acc, gyro, audio = self.preprocess(acc, gyro, audio)
                # get the features of the file
                self.sl_wind_feat_extract(acc, gyro, audio, 
                                                     label,
                                                     part)


    
    def preprocess(self, acc, gyro, audio):
        """Average smooth IMU and get percussive audio"""
        self.smoothing_average(acc)
        self.smoothing_average(gyro)
        _, audio = harmonic_percussive_filter(audio)

        return acc, gyro, audio
    
    def smoothing_average(self, imu_df):
        """Smooth the IMU data using a moving average filter.
        Changes data in place"""
        window_size = self.SMOOTHING_AV_WIND_SIZE*self.SMPL_FREQ_IMU
        weights = np.repeat(1.0, window_size) / window_size

        smooth_data_x = np.convolve(weights, np.array(imu_df.iloc[:,0]), 'valid')
        smooth_data_y = np.convolve(weights, np.array(imu_df.iloc[:,1]), 'valid')
        smooth_data_z = np.convolve(weights, np.array(imu_df.iloc[:,2]), 'valid')

        imu_df.iloc[:,0] = smooth_data_x
        imu_df.iloc[:,1] = smooth_data_y
        imu_df.iloc[:,2] = smooth_data_z

    def sl_wind_feat_extract(self, 
                             acc, gyro, audio, 
                             label,
                             part):
        """Extract IMU and audio features on sliding windows"""
        
        # frame start and end times
        strt_time = 0
        end_time = self.FRAME_SIZE
        length = len(acc)/self.SMPL_FREQ_IMU #length of file in time
        while end_time <= length:
            strt_idx_imu , end_idx_imu = int(strt_time*self.SMPL_FREQ_IMU), \
                                            int(end_time*self.SMPL_FREQ_IMU)
            strt_idx_audio, end_idx_audio = int(strt_time*self.SMPL_FREQ_AUDIO), \
                                            int(end_time*self.SMPL_FREQ_AUDIO) 
            # get the features of the current frame

            #rms_check
            if(label == 2 | label == 4 | label == 5 | label == 6 | label == 7):
                segment_class = self.separate_null_positive(gyro, label)
            else:
                segment_class = label
            features = self.get_features_per_frame(acc.iloc[strt_idx_imu:end_idx_imu, :],
                                            gyro.iloc[strt_idx_imu:end_idx_imu, :],
                                            audio[strt_idx_audio:end_idx_audio],
                                            segment_class,
                                            part)

            # update the frame start and end times
            strt_time += self.SLID_PARAM
            end_time += self.SLID_PARAM

            # import pdb; pdb.set_trace()
            self.features = self.features.append(features, ignore_index = True)
    
    def get_features_per_frame(self,
                                 acc, gyro, audio,
                                 label,
                                 part):
        """Extracts features from a single frame"""
        # get the features of the current frame
        imu_features = self.get_imu_features(acc, gyro)
        audio_embeddings = YamNetEmbeddings.get_audio_embeddings(audio)
        
        self.preprocess_embeddings(imu_features)
        self.preprocess_embeddings(audio_embeddings)

        audio_imu_feat = pd.concat([imu_features, audio_embeddings], axis=1)

        audio_imu_feat["label"] = label
        audio_imu_feat["participant"] = part

        return audio_imu_feat

    def get_imu_features(self, acc, gyro):
        """
        Utility function to calculate the feature from a window of data
        data = time series data of a single window of data
        class_name = corresponding class name of the window of the data
        """
        print("Calculating IMU features ...")
        acc_corr_matrix = (acc.loc[:,["x",
                                    "y",
                                    "z"]] - 
                        acc.loc[:,["x",
                                    "y",
                                    "z"]].mean()).corr()  #correlation after the mean was taken out
        
        acc_mag = sqrt(acc.loc[:,"x"]**2 + 
                    acc.loc[:,"y"]**2 + 
                    acc.loc[:, "z"]**2)  #magnitude of the acceleration vector
        


        gyro_mag = sqrt(gyro.loc[:,"x"]**2 + 
                    gyro.loc[:,"y"]**2 + 
                    gyro.loc[:, "z"]**2)
        
        
        
        orientation = arccos(acc.loc[:,"y"]/acc_mag)

        feature = pd.DataFrame({"mean_acc_x": acc.loc[:, "x"].mean(), 
                                "mean_acc_y": acc.loc[:, "y"].mean(), 
                                "mean_acc_z": acc.loc[:, "z"].mean(),
                                
                                "variance_acc_x": acc.loc[:, "x"].var(), 
                                "variance_acc_y": acc.loc[:, "y"].var(), 
                                "variance_acc_z": acc.loc[:, "z"].var(),
                                
                            "q1_accx": acc.loc[:, "x"].quantile(0.25), 
                                "q3_accx": acc.loc[:, "x"].quantile(0.75), 
                                
                                "q1_accy": acc.loc[:, "x"].quantile(0.25), 
                                "q3_accy": acc.loc[:, "y"].quantile(0.75),
                                
                                "q1_accz": acc.loc[:, "z"].quantile(0.25), 
                                "q3_accz": acc.loc[:, "z"].quantile(0.75),
                                
                            "corr_acc_xy": acc_corr_matrix.iloc[0,1], 
                                "corr_acc_xz": acc_corr_matrix.iloc[0,2], 
                                "corr_acc_yz": acc_corr_matrix.iloc[1,2],
                                
                            "skewness_acc_x": skew(acc.loc[:, "x"]),
                                "skewness_acc_y": skew(acc.loc[:, "y"]),
                                "skewness_acc_z": skew(acc.loc[:, "z"]),
                                
                            "kurtosis_acc_x": kurtosis(acc.loc[:, "x"]),
                                "kurtosis_acc_y": kurtosis(acc.loc[:, "y"]),
                                "kurtosis_acc_z": kurtosis(acc.loc[:, "z"]),
                                
                            "mean_magnitude_acc": acc_mag.mean(), 
                                "var_magnitude_acc" : acc_mag.var(), 
                                
                                "q1_mag_acc": acc_mag.quantile(0.25),  
                                "q3_mag_acc": acc_mag.quantile(0.75),  
                                
                                "skewness_magnitude_acc": skew(acc_mag), 
                                "kurtosis_magnitude_acc": kurtosis(acc_mag),
                                
                            "omega_accx": arctan((acc.loc[:,"x"].max() - 
                                                acc.loc[:,"x"].min())/
                                                (acc.loc[:,"x"].idxmax() - 
                                                acc.loc[:,"x"].idxmin())),
                                
                            "omega_accy": arctan((acc.loc[:,"y"].max() - 
                                                acc.loc[:,"y"].min())/
                                                (acc.loc[:,"y"].idxmax() - 
                                                acc.loc[:,"y"].idxmin())),
                                
                            "omega_accz": arctan((acc.loc[:,"z"].max() - 
                                                acc.loc[:,"z"].min())/
                                                (acc.loc[:,"z"].idxmax() - 
                                                acc.loc[:,"z"].idxmin())),
                                
                            "omega_mag": arctan((acc_mag.max() - acc_mag.min())/(acc_mag.idxmax() - acc_mag.idxmin())), 
                                
                            "mean_orientation": orientation.mean(),
                                
                                "variance_orientation": orientation.var(),
                                "q_1_orientation": orientation.quantile(0.25),
                                "q_3_orientation": orientation.quantile(0.75),
                                "mean_gyro_x": gyro.loc[:, "x"].mean(),
                             "mean_gyro_y": gyro.loc[:, "y"].mean(),
                             "mean_gyro_z": gyro.loc[:, "z"].mean(),
                          "var_gyro_x": gyro.loc[:, "x"].var(),
                             "var_gyro_y": gyro.loc[:, "y"].var(),
                             "var_gyro_z": gyro.loc[:, "z"].var(),
                          "skewness_gyro_x": skew(gyro.loc[:, "x"]),
                             "skewness_gyro_y": skew(gyro.loc[:, "y"]),
                             "skewness_gyro_z": skew(gyro.loc[:, "z"]),
                          "kurtosis_gyro_x": kurtosis(gyro.loc[:, "x"]),
                             "kurtosis_gyro_y": kurtosis(gyro.loc[:, "y"]),
                             "kurtosis_gyro_z": kurtosis(gyro.loc[:, "z"]), 
                            
                          "omega_gyro_x": arctan((gyro.loc[:,"x"].max() - 
                                              gyro.loc[:,"x"].min())/
                                             (gyro.loc[:,"x"].idxmax() - 
                                              gyro.loc[:,"x"].idxmin())),
                            
                        "omega_gyro_y": arctan((gyro.loc[:,"y"].max() - 
                                              gyro.loc[:,"y"].min())/
                                             (gyro.loc[:,"y"].idxmax() - 
                                              gyro.loc[:,"y"].idxmin())),
                            
                        "omega_gyro_z": arctan((gyro.loc[:,"z"].max() - 
                                              gyro.loc[:,"z"].min())/
                                             (gyro.loc[:,"z"].idxmax() - 
                                              gyro.loc[:,"z"].idxmin())),
                            
                          "mean_magnitude_gyro": gyro_mag.mean(),
                          "variance_magnitude_gyro": gyro_mag.var(),
                          "q1_magnitude_gyro": gyro_mag.quantile(0.25), 
                            "q3_magnitude_gyro": gyro_mag.quantile(0.75),
                          "skewness_magnitude_gyro": skew(gyro_mag),
                          "kurtosis_magnitude_gyro": kurtosis(gyro_mag),
                          "omega_magnitude_gyro": arctan((gyro_mag.max() - gyro_mag.min())/(gyro_mag.idxmax())),
    #                             "zero_crossing_rate": [],
                            "rms_acc_x": self.get_rms(acc.loc[:, "x"]),
                            "rms_acc_y": self.get_rms(acc.loc[:, "y"]),
                            "rms_acc_z": self.get_rms(acc.loc[:, "z"]),
                            "rms_gyro_x": self.get_rms(gyro.loc[:, "x"]),
                            "rms_gyro_y": self.get_rms(gyro.loc[:, "y"]),
                            "rms_gyro_z": self.get_rms(gyro.loc[:, "z"])                             
                                                                },
                                                                index=[0])
        
        return feature
    

    def get_rms(self, data):
        """To Do: Test"""
        normalized_mean = self.mean_normalize(data)
        return np.sqrt(
            np.sum(normalized_mean**2)/len(normalized_mean)
            )

    def mean_normalize(self, data):
        return data - np.mean(data)
    
    def preprocess_embeddings(self, embeddings):
        """Normalize embeddings to unit length, zero mean and return them as a numpy array"""
        embeddings = embeddings / np.linalg.norm(embeddings, keepdims=True)
        embeddings = embeddings - np.mean(embeddings, axis=0)
        return embeddings
    

    def separate_null_positive(self, gyro, class_name):
        """ Given a frame of data, data_frame, assign the class label only if the
        data_frame passes the threshhold rule. Otherwise, assign the class label as null_class
        """
        if (self.rms_check(gyro, THRESHOLD_RMS_GYRO)): 
            return class_name
        return "null_class"
    
    def rms_check(self, gyro, threshold):
        # calculate rms for each channel in gyroscope
        rms_gyro_x = self.get_rms(gyro.loc[:, "x"])
        rms_gyro_y = self.get_rms(gyro.loc[:, "y"])
        rms_gyro_z = self.get_rms(gyro.loc[:, "z"])

        # print("RMS Value: {}".format(np.max([rms_acc_x, rms_acc_y, rms_acc_z])))
        if(np.max([rms_gyro_x, rms_gyro_y, rms_gyro_z]) > threshold):
            return True
        return False

if __name__ == "__main__":
    feature_extractor = Feature_Extractor()
    feature_extractor.calculate_features()