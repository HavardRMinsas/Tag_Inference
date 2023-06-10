import pandas as pd
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
import os
from tsfresh import extract_features

class comprehensive_features():


    def __init__(self):
        self.settings = ComprehensiveFCParameters


    def construct_file_paths(self, dir="./raw_data/test/"):
        """
        Arguments: takes the directory name of which to create filepaths from
        Output: Returns a dictionary with sensor name as key and a list of filepaths as values
        {Sensor_Type1: [filepath1, filepath2,...., filepath_n], SensorType2: [...]}
        """
        root_dir = dir
        # Gives us the names of each sensor (sub-directories) in a list
        sensor_names = next(os.walk(root_dir))[1]
        paths = {}
        already_extracted = self.read_already_extracted_files("comprehensive_features_5h_test.txt")
        for sensor in sensor_names:
            if sensor in already_extracted:
                continue
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
        for key in filepaths:
            feature_list = []
            print(f"extracting features from {key}")
            for filepath in filepaths[key]:
                #print(filepath)
                print(f"extracting from: {filepath}")
                df = pd.read_feather(filepath)
                df = self.df_even_interval(df)
                # We add this so that tsfresh do not group our values into subgroups based on id
                df["id"] = "id"
                try:
                    features = extract_features(df, default_fc_parameters=ComprehensiveFCParameters(), column_id="id", column_sort="index")
                except:
                    f = open("failed_feature_extraction_5h.txt", "a", encoding="UTF-8")
                    f.write(filepath + "\n")
                    print(f"Could not extract features from: {filepath}")
                    continue
                tmp_str= ""
                for feature in features:
                    tmp_str += feature +" "+ str(features[feature]["id"])+" " 
                feature_list.append(filepath + " " + tmp_str)
            self.write_to_file(feature_list, "comprehensive_features_5h_test.txt")
        print("completed write")


    def df_even_interval(self, df):
        """Takes a dataframe and return a dataframe with even intervals (10 min) from starttime to endtime"""
        df["timestamp"] = df["timestamp"].values.astype('datetime64[s]')
        start = pd.to_datetime(str(df["timestamp"].min())) 
        end = pd.to_datetime(str(df["timestamp"].max()))
        timestamps = pd.date_range(start, end, freq="5H")
        df = df.drop_duplicates(subset="timestamp")
        df = df.set_index("timestamp").reindex(timestamps, method="ffill").fillna(0.0).reset_index()
        return df

    def write_to_file(self, features, filename):
        f = open(filename, "a", encoding="UTF-8")
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
        f.close()
        return feature_dict

    def read_already_extracted_files(self, filename):
        f = open(filename, "r", encoding="UTF-8")
        already_extracted = []
        for line in f:
            line = line.strip()
            line = line.replace("&", " ")
            line = line.replace("//", "/")
            line = line.replace(", ", "-") #Makes sure we do not split within feature names (some have spaces)
            line = line.split(" ")
            sensor = line[0].split("/")[3].strip()
            already_extracted.append(sensor)
        f.close()
        return list(set(already_extracted))
            

if __name__ == "__main__":
    extractor = comprehensive_features()
    #extractor.construct_file_paths()
    filepaths = extractor.construct_file_paths()
    features = extractor.extract_features_from_file(filepaths)
    #extractor.write_to_file(features, "comprehensive_features_1m_train.txt")

    #filepaths = extractor.construct_file_paths(dir="./raw_data/test/")
    #features = extractor.extract_features_from_file(filepaths)
    #extractor.write_to_file(features, "comprehensive_features_1m_test.txt")
    #features_train = extractor.read_file("minimal_features_10m_train.txt")
    #features_test = extractor.read_file("comprehensive_features_10m_test.txt")