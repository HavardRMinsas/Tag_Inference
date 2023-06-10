from pycatch22 import catch22_all

import pandas as pd
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
import os
from tsfresh import extract_features
"""
filepath = './raw_data/test/Supply_Air_Temperature_Sensor/2022-06/2022-06_DBC=360.004-RT400_PV.feather'

f = pd.read_feather(filepath)

timeseries = f["value"].values.tolist()


features = catch22_all(timeseries, catch24=True)["values"]
print(features)

"""
class catch22_features():


    def __init__(self):
        self.settings = ComprehensiveFCParameters


    def construct_file_paths(self, dir="./raw_data/train/"):
        """
        Arguments: takes the directory name of which to create filepaths from
        Output: Returns a dictionary with sensor name as key and a list of filepaths as values
        {Sensor_name1: [filepath1, filepath2,...., filepath_n], SensorName2: [...]}
        """
        root_dir = dir
        # Gives us the names of each sensor (sub-directories) in a list
        sensor_names = next(os.walk(root_dir))[1]
        paths = {}
        for sensor in sensor_names:
            print(f"Extracting for: {sensor}")
            count_files = 0
            sensor_features = []
            """Read through each file in each subdirectory"""
            subdir_names = next(os.walk(root_dir+sensor))[1]
            for subdir in subdir_names:
                path = str(root_dir) + "/" + str(sensor) + "/" + str(subdir)
                # os.walk returns directory (current), subdirectories, files
                filenames = next(os.walk(path))[2]
                for filename in filenames:
                    count_files += 1
                    if sensor not in paths: #Create dictionary entries with sensor_names as keys and a list of file paths as values
                        paths[sensor] = [str(path) + "/" + str(filename)]
                    else:
                        paths[sensor].append(str(path) + "/" + str(filename))
            print(f"number of files for {sensor}: {count_files}")
        return paths


    def extract_features_from_file(self, filepaths):
        """
        Arguments: Filepaths on the format {Sensor_name1: [filepath1, filepath2,...., filepath_n], SensorName2: [...]} from self.construct_file_paths(dir)
        Output: Dictionary with senorname as key and dictionary with featurename and associated values as value.
                {Sensor_name1: [{Feat1: value, Feat2: value}, {Feat1: value, Feat2: value}]
        """
        feature_list = []
        for key in filepaths:
            print(f"extracting features from {key}")
            for filepath in filepaths[key]:
                #print(filepath)
                print(f"extracting from: {filepath}")
                df = pd.read_feather(filepath)
                df = self.df_even_interval(df)
                # We add this so that tsfresh do not group our values into subgroups based on id
                timeseries = df["value"].values.tolist()
                """features = extract_features(df, default_fc_parameters=MinimalFCParameters(
                ), column_id="id", column_sort="timestamp")"""
                try:
                    features = catch22_all(timeseries, catch24=True)
                    feature_names = features["names"]
                    features = features["values"]
                except:
                    f = open("failed_feature_extraction_catch22_5H.txt", "wa", encoding="UTF-8")
                    f.write(filepath + "\n")
                    print(f"Could not extract features from: {filepath}")
                    pass
                tmp_str= ""
                for i in range(len(features)):
                    tmp_str += feature_names[i] +" "+ str(features[i])+" " #Uses & to easily be able to seperate features in the txt
                feature_list.append(filepath + " " + tmp_str)
        return feature_list


    def df_even_interval(self, df):
        """Takes a dataframe and return a dataframe with even intervals (10 min) from starttime to endtime"""
        df["timestamp"] = df["timestamp"].values.astype('datetime64[s]')
        start = pd.to_datetime(str(df["timestamp"].min())) 
        end = pd.to_datetime(str(df["timestamp"].max()))
        timestamps = pd.date_range(start, end, freq="1H")
        df = df.drop_duplicates(subset="timestamp")
        df = df.set_index("timestamp").reindex(timestamps, method="ffill").fillna(0.0).reset_index()
        return df

    def write_to_file(self, features):
        f = open("catch22_5h_train.txt", "w", encoding="UTF-8")
        for el in features:
            f.write(el + "\n")
        f.close()
        print("write success")
        return

    def read_file(self, filename):
        feature_dict = {}
        f = open(filename, "r", encoding="UTF-8")
        for line in f:
            line = line.strip()
            line = line.replace(", ", "-") #Makes sure we do not split within feature names (some have spaces)
            line = line.replace("//", "/")
            line = line.replace("&", " ")
            line = line.split(" ")
            #print(f"reading for: {line[0]}")
            sensor = line[0].split("/")[3].strip()
            line[0] = sensor
            if line[0] not in feature_dict: #If sensor is not already a key in the dictionary
                feature_dict[line[0]] = []
            sensor_features = {}
            for i in range(1, len(line), 2):
                sensor_features[line[i]] = float(line[i+1])
            feature_dict[line[0]].append(sensor_features)
        return feature_dict

            

if __name__ == "__main__":
    extractor = catch22_features()
    filepaths = extractor.construct_file_paths()
    features = extractor.extract_features_from_file(filepaths)
    extractor.write_to_file(features)
    #print(extractor.read_file("catch22_10m_train.txt"))



















#Paper about catch22 https://link.springer.com/article/10.1007/s10618-019-00647-x