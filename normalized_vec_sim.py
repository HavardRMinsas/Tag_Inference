import pandas as pd
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
import os
from tsfresh import extract_features

class normalized_vec_sim():

    def __init__(self):
        parameters = MinimalFCParameters()

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
            sensor_features = []
            """Read through each file in each subdirectory"""
            subdir_names = next(os.walk(root_dir+sensor))[1]
            for subdir in subdir_names:
                path = str(root_dir) + "/" + str(sensor) + "/" + str(subdir)
                # os.walk returns directory (current), subdirectories, files
                filenames = next(os.walk(path))[2]
                for filename in filenames:
                    if sensor not in paths: #Create dictionary entries with sensor_names as keys and a list of file paths as values
                        paths[sensor] = [str(path) + "/" + str(filename)]
                    else:
                        paths[sensor].append(str(path) + "/" + str(filename))
        return paths


    def extract_features_from_file(self, filepaths):
        """
        Arguments: Filepaths on the format {Sensor_name1: [filepath1, filepath2,...., filepath_n], SensorName2: [...]} from self.construct_file_paths(dir)
        Output: Dictionary with senorname as key and dictionary with featurename and associated values as value.
                {Sensor_name1: [{Feat1: value, Feat2: value}, {Feat1: value, Feat2: value}]
        """
        feature_dict = {}
        for key in filepaths:
            print(f"extracting features from {key}")
            for filepath in filepaths[key]:
                #print(filepath)
                df = pd.read_feather(filepath)
                df = self.df_even_interval(df)
                # We add this so that tsfresh do not group our values into subgroups based on id
                df["id"] = "id"
                """features = extract_features(df, default_fc_parameters=MinimalFCParameters(
                ), column_id="id", column_sort="timestamp")"""
                features = extract_features(df, default_fc_parameters=MinimalFCParameters(), column_id="id", column_sort="timestamp")
                tmp_feature_dict = {}
                for feat_key in features: 
                    tmp_feature_dict[feat_key] = features[feat_key][0]
                if key in feature_dict:
                    feature_dict[key].append(tmp_feature_dict)
                else:
                    feature_dict[key] = [tmp_feature_dict] 
        return feature_dict

    def df_even_interval(self, df):
        """Takes a dataframe and return a dataframe with even intervals (1s) from starttime to endtime"""
        df["timestamp"] = df["timestamp"].values.astype('datetime64[s]')
        start = pd.to_datetime(str(df["timestamp"].min())) 
        end = pd.to_datetime(str(df["timestamp"].tail().min()))
        timestamps = pd.date_range(start, end, freq="1S")
        df = df.drop_duplicates(subset="timestamp")
        df = df.set_index("timestamp").reindex(timestamps).reset_index().reindex(columns=df.columns)
        df["value"] = df["value"].ffill()
        df["timestamp"] = timestamps
        return df

    def extract_maximum_feature_value(self, feature_dict):
        """
        Arguments: Dictionary on the form returned by extract_features_from_file()
        Output: Dictionary with feature_name as keys, and maximum value as value 
        """
        max_dict = {}
        for sensor in feature_dict: 
            for sensor_features in feature_dict[sensor]: 
                for feature_name in sensor_features:
                    if feature_name not in max_dict:
                        max_dict[feature_name] = sensor_features[feature_name]
                    elif sensor_features[feature_name] > max_dict[feature_name]:
                        max_dict[feature_name] = sensor_features[feature_name]
        return max_dict 

    def normalize(self, max_dict, feature_dict):
        """
        Input:
            max_dict: returned from extract_maximum_feature_values
            features: returned from extract_features from file
        Output:
            a feature_dict with values normalized.
        """
        for sensor in feature_dict: #Sensor --> list of dicts
            #print(f"sensor: {sensor}")
            for features in feature_dict[sensor]: #Dicts
                #print(f"feature in sensor: {features}")
                for feature_name in features:
                    features[feature_name] = features[feature_name] / max_dict[feature_name]
        return feature_dict

    def features_avg(self, feature_dict):
        """
        Takes a list of features and compute the average values for each sensor,
        return a dict with these values
        """
        avg_features = {}
        for sensor in feature_dict:
            features_sensor = {}
            for features_list in feature_dict[sensor]:
                for feature_name in features_list:
                    if feature_name not in features_sensor:
                        features_sensor[feature_name] = features_list[feature_name]
                    else:
                        features_sensor[feature_name] += features_list[feature_name]
            for feature in features_sensor:
                features_sensor[feature] = features_sensor[feature] / len(feature_dict[sensor])
            avg_features[sensor] = features_sensor
        return avg_features



    def infer_sensor(self, features, avg_sensor_features):
        """
        features are the normalized features belonging to the sensor we want to infer
        sensor_features is the complete list of features for every type of sensor
        """
        infered = {}
        for sensor in features:
            for el in features[sensor]: #Dict consisting of features and values
                ans = self.calc_distance(sensor, el, avg_sensor_features)
                if sensor in infered:
                    infered[sensor].append(ans)
                else:
                    infered[sensor] = [ans]
        return infered

    def calc_distance(self, actual_sensor, features, avg_features):
        """
        Returns the original sensor, the closest sensor in avg_features and their distance
        """
        infered_sensor = ""
        min_distance = float("inf")
        for sensor in avg_features:
            dist = 0
            for feature in avg_features[sensor]:
                dist += abs(features[feature] - avg_features[sensor][feature])
            if dist < min_distance:
                min_distance = dist
                infered_sensor = sensor
        return (actual_sensor, dist, infered_sensor)


    def write_avg_feat_to_file(self, avg_features, filename="avg_features_even_minimal.txt"):
        f = open(filename, "w", encoding="UTF-8")
        for key in avg_features:
            f.write(str(key) + str(avg_features[key]) + "\n")
        f.close()




if __name__ == "__main__":
    vec = normalized_vec_sim()
    filepaths = vec.construct_file_paths()
    features = vec.extract_features_from_file(filepaths)
    max_values = vec.extract_maximum_feature_value(features)
    features = vec.normalize(max_values, features)
    avg_features = vec.features_avg(features)
    #print(avg_features)
    test_filepaths = vec.construct_file_paths(dir="./raw_data/test/")
    test_features = vec.extract_features_from_file(test_filepaths)
    test_features = vec.normalize(max_values, test_features)
    #print(test_features)
    infered = vec.infer_sensor(test_features, avg_features)
    print(infered)
    vec.write_avg_feat_to_file(avg_features)
    vec.write_avg_feat_to_file(infered, filename="infered_vec_normalized.txt")

    
    correct = 0
    wrong = 0
    for key in infered:
        for el in infered[key]:
            if el[0] == el[2]:
                correct += 1
            else:
                 wrong += 1

    print(f"correct: {correct}")
    print(f"wrong: {wrong}")

    for key in infered:
        correct = 0
        wrong = 0
        print(key)
        for el in infered[key]:
            if el[0] == el[2]:
                correct += 1
            else:
                wrong += 1
        print(f"correct: {correct}")
        print(f"wrong: {wrong}")