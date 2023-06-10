import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math



class IQR():

    def __init__(self):
        #self.use_classes = [
        #'Outside_Air_Temperature_Sensor',
        #'Chilled_Water_Return_Temperature_Sensor', 'Chilled_Water_Supply_Temperature_Sensor', 'Hot_Water_Supply_Temperature_Sensor', 'Preheat_Supply_Air_Temperature_Sensor', 'Return_Air_Temperature_Sensor', 'Return_Water_Temperature_Sensor', 'Supply_Air_Temperature_Sensor',
        #Cooling_Valve', 'Reheat_Valve', 'Valve',
        #'Differential_Pressure_Sensor',
        #'Discharge_Air_Static_Pressure_Sensor', 'Supply_Air_Static_Pressure_Sensor',
        #'Heat_Exchanger', 'Variable_Frequency_Drive',
        #'Return_Fan', 'Supply_Fan',
        #'Power_Sensor',
        #'Pump',
        #'Energy_Sensor'
        #]
        self.use_classes = ['Outside_Air_Temperature_Sensor']


    def read_features_from_file(self, filename, inv_features= [], short_files=[]):
        print("reading from file")
        feature_dict = {}
        file_dict = {}
        f = open(filename, "r", encoding="UTF-8")
        invalid_features = inv_features
        #count = 0
        for line in f:
            #count += 1
            line = line.strip()
            line = line.replace("&", " ")
            line = line.replace("//", "/")
            line = line.replace(", ", "-") #Makes sure we do not split within feature names (some have spaces)
            line = line.split(" ")
            sensor = line[0].split("/")[3].strip()
            if sensor not in self.use_classes or line[0].strip() in short_files: #Skips sensor if it is not relevant
                continue
            for i in range(len(self.use_classes)):
                if sensor == self.use_classes[i]:
                    sensor = i
            
            if sensor not in feature_dict: #If sensor is not already a key in the dictionary
                feature_dict[sensor] = []
                file_dict[sensor] = []
            sensor_features = {}

            file_dict[sensor].append(line[0])
            
            for i in range(1, len(line)-1, 2):
            
                if line[i+1] == "nan":
                    sensor_features[line[i]] = 1
                    if line[i] not in invalid_features:
                        invalid_features.append(line[i])
                else:        
                    sensor_features[line[i]] = float(line[i+1])
            
            feature_dict[sensor].append(sensor_features)
        #print(f"num_files: {count}")
        #feature_dict = self.remove_invalid_values(feature_dict, list(set(invalid_features)))
        return feature_dict, invalid_features

    def read_features_from_file2(self, filename, invalid):
        feature_dict = {}
        f = open(filename, "r", encoding="UTF-8")
        for line in f:
            line = line.strip()
            line = line.replace("&", " ")
            line = line.replace("//", "/")
            line = line.replace(", ", "-") #Makes sure we do not split within feature names (some have spaces)
            line = line.split(" ")
            sensor = line[0].split("/")[3].strip()
            if sensor not in self.use_classes:
                continue

            feature_dict[line[0].strip()] = {}
            for i in range(1, len(line)-1, 2):
                if line[i].strip() not in invalid:
                    feature_dict[line[0].strip()][line[i]] = float(line[i+1])
        return feature_dict

    def remove_invalid_values(self, dict, invalid_list):
        print("removing invalid features")
        #print(f"invalid_list: {invalid_list}")
        for invalid_feature in invalid_list:
            for sensor in dict:
                for el in dict[sensor]:
                    el.pop(invalid_feature, None)
        return dict
    
    def dict_to_arrays(self, feature_dict, file_dict):
        """Convertin to arrays"""
        features = [] #2d array
        target = [] #1d array that holds the corresponding classes
        file_list = []
        num = 0

        for sensor in feature_dict:

            for i in range(len(feature_dict[sensor])):
                tmp_lst = []
                file_list.append(file_dict[sensor][i])

                for feature in feature_dict[sensor][i]:
                    tmp_lst.append(feature_dict[sensor][i][feature])
                features.append(tmp_lst)
                target.append(sensor)
        return np.array(features), np.array(target), np.array(file_list)
    
    def order_by_feat(self, features):
        dict = {}
        for sensor in features:
            for individual_sensor in features[sensor]: #This is a dict-element
                for feat in individual_sensor:
                    if feat not in dict:
                        dict[feat] = [individual_sensor[feat]]
                    else: 
                        dict[feat].append(individual_sensor[feat])
        return dict 
    

    def find_quartiles(self, grouped_dict):
        """Takes in a dict grouped by features, return a dict by features with lower and upped limit"""
        bound_dict = {}
        for feat in grouped_dict: 
            grouped_dict[feat].sort()
            tmp = grouped_dict[feat]
            lower_bound = math.floor(len(tmp) * 0.1)#Index for lower 25 % 
            upper_bound = math.ceil(len(tmp) * 0.9) - 1#Index for the upper 75%
            iqr = tmp[upper_bound] - tmp[lower_bound] #Interquartile range 
            min_outlier = tmp[lower_bound] - (1.5 * iqr)
            max_outlier = tmp[upper_bound] + (1.5 * iqr)
            bound_dict[feat] = [min_outlier, max_outlier]
        return bound_dict 

    def identify_outliers(self, features, bound_dict):
        outliers = []
        for el in features:
            count = 0
            for feat in features[el]:
                if features[el][feat] <= bound_dict[feat][0] or features[el][feat] >= bound_dict[feat][1]:
                    count += 1
                    #print("count upped")
            if count >= 275: #275 is used as deafult
                outliers.append(el.strip())
        return outliers
    
    def return_outliers(self, use_classes, total_invalid, short_files=[], filepath="comprehensive_features_10m_train.txt"): #Use classes is a list consisting of the sensors you want to find outliers for
        self.use_classes = use_classes
        feat, invalid = self.read_features_from_file(filepath, inv_features=total_invalid)

        grouped = self.order_by_feat(feat)

        bound_dict = self.find_quartiles(grouped)

        train_feat = self.read_features_from_file2(filepath, invalid)

        outliers = self.identify_outliers(train_feat, bound_dict)
        #print(self.use_classes)
        #print(len(outliers))
        return outliers
        

if __name__ == "__main__":
    iqr = IQR()

    #outliers = iqr.return_outliers(['Outside_Air_Temperature_Sensor'])

    #print(len(outliers))

    g1 = ['Outside_Air_Temperature_Sensor']
    g2 = ['Chilled_Water_Return_Temperature_Sensor', 'Chilled_Water_Supply_Temperature_Sensor', 'Hot_Water_Supply_Temperature_Sensor', 'Preheat_Supply_Air_Temperature_Sensor', 'Return_Air_Temperature_Sensor', 'Return_Water_Temperature_Sensor', 'Supply_Air_Temperature_Sensor']
    g3 = ['Cooling_Valve', 'Reheat_Valve', 'Valve']
    g4 = ['Differential_Pressure_Sensor']
    g5 = ['Discharge_Air_Static_Pressure_Sensor', 'Supply_Air_Static_Pressure_Sensor', ]
    g6 = ['Heat_Exchanger', 'Variable_Frequency_Drive']
    g7 = ['Return_Fan', 'Supply_Fan']
    g8 = ['Power_Sensor']
    g9 = ['Pump']
    g10 = ['Energy_Sensor']
    
    groupings = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
    outliers = []
    for i in range(len(groupings)):
        tmp = iqr.return_outliers(groupings[i], [])
        #print(f"len {i}: {len(tmp)}")
        outliers = outliers + tmp
        print(i)
        print(f"len_outliers: {len(tmp)}")
        print(tmp)

    print(len(outliers))