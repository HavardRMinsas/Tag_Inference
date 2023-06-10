import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn import tree
from node_one_to_n import gradient_node_one_to_n as grad_n
from IQR_outliers import IQR
from functools import reduce
import copy
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

class gradient_boost_compact():


    def __init__(self, g1=[], g2=[]):
        #self.clf = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, max_features=1000, random_state=0, verbose=10) 
        self.clf = HistGradientBoostingClassifier(max_iter=100, random_state=0, learning_rate=0.02, verbose=10, early_stopping=False) #learning_rate 0.01 gives 0.93046 score
        #self.clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
        #self.clf = DecisionTreeClassifier(max_depth=12, random_state=0)
        #self.clf = RandomForestClassifier(n_estimators=400, max_depth=16, random_state=0, verbose=10)
        self.iqr = IQR()
        #self.clf = tree.DecisionTreeClassifier(max_depth=10)
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
        
        self.tmp = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        self.groupings = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


    def read_features_from_file(self, filename, outliers, inv_features= [], short_files=[]):
        print("reading from file")
        feature_dict = {}
        file_dict = {}
        f = open(filename, "r", encoding="UTF-8")
        invalid_features = inv_features
        count = 0
        for line in f:
            line = line.strip()
            line = line.replace("&", " ")
            line = line.replace("//", "/")
            line = line.replace(", ", "-") #Makes sure we do not split within feature names (some have spaces)
            line = line.split(" ")
            sensor = line[0].split("/")[3].strip()

            if line[0].strip() in outliers or line[0].strip() in short_files: #Skips sensor if it is not relevant
                #print(f"skipped sensor: {line[0].strip()}")
                continue
                
            for i in range(len(self.groupings)):
                if sensor in self.groupings[i]:
                    sensor = i
            
            if sensor not in feature_dict: #If sensor is not already a key in the dictionary
                feature_dict[sensor] = []
                file_dict[sensor] = [] 
            sensor_features = {}

            file_dict[sensor].append(line[0])
            count += 1
            
            for i in range(1, len(line)-1, 2):
            
                #if line[i+1] == "nan" and line[i] not in invalid_features:
                if line[i+1] == "nan":
                    sensor_features[line[i]] = 1
                    invalid_features.append(line[i])
                else:        
                    sensor_features[line[i]] = float(line[i+1])
            
            feature_dict[sensor].append(sensor_features)
        
        #feature_dict = self.remove_invalid_values(feature_dict, list(set(invalid_features)))
        print(count)
        return feature_dict, invalid_features, file_dict
    

    
    def read_features_from_file_grouped_by_sensor(self, filename, outliers, inv_features= [], short_files=[]):
        print("reading from file")
        feature_dict = {}
        f = open(filename, "r", encoding="UTF-8")
        invalid_features = inv_features
        sensor_dict = {}
        for line in f:
            line = line.strip()
            line = line.replace("&", " ")
            line = line.replace("//", "/")
            line = line.replace(", ", "-") #Makes sure we do not split within feature names (some have spaces)
            line = line.split(" ")
            sensor = line[0].split("/")[3].strip()
            sensor_name = line[0].split("/")[5][8:-8] #Name of sensor
            if sensor_name not in sensor_dict:
                sensor_dict[sensor_name] = 1
            else:
                sensor_dict[sensor_name] += 1
            if line[0].strip() in outliers or line[0].strip() in short_files: #Skips sensor if it is not relevant
                #print(f"skipped sensor: {line[0].strip()}")
                continue
            for i in range(len(self.groupings)):
                if sensor in self.groupings[i]:
                    sensor = i
            
            if sensor not in feature_dict: #If sensor is not already a key in the dictionary
                feature_dict[sensor] = {}
            if sensor_name not in feature_dict[sensor]:
                feature_dict[sensor][sensor_name] = []
            sensor_features = {}
            
            for i in range(1, len(line)-1, 2):
            
                #if line[i+1] == "nan" and line[i] not in invalid_features:
                if line[i+1] == "nan":
                    sensor_features[line[i]] = 0
                    invalid_features.append(line[i])
                else:        
                    sensor_features[line[i]] = float(line[i+1])
            
            feature_dict[sensor][sensor_name].append(sensor_features)
        
        #feature_dict = self.remove_invalid_values_grouped(feature_dict, list(set(invalid_features)))
        #print(feature_dict)
        return feature_dict, invalid_features, sensor_dict
    
    def remove_invalid_values_grouped(self, dict, invalid_list):
        print("removing invalid features")
        #print(f"invalid_list: {invalid_list}")
        for invalid_feature in invalid_list:
            for sensor in dict:
                for sensor_name in dict[sensor]:
                    for el in dict[sensor][sensor_name]:
                        el.pop(invalid_feature, None)
        return dict
    


    def remove_invalid_values(self, dict, invalid_list):
        print("removing invalid features")
        #print(f"invalid_list: {invalid_list}")
        for invalid_feature in invalid_list:
            for sensor in dict:
                for el in dict[sensor]:
                    el.pop(invalid_feature, None)
        return dict
    
    def dict_to_arrays(self, feature_dict, file_dict):
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
    
    def read_short_files(self, filepath):
        short_files = []
        f = open(filepath, "r")
        for line in f:
            short_files.append(line.strip())
        return short_files

        

    def dict_to_array_grouped(self, test_feat, test_target):
        features = []
        target = [test_target]
        for el in test_feat:
            tmp_lst = []
            for feature in el:
                
                tmp_lst.append(el[feature])
            features.append(tmp_lst)
        if len(target) == 1:
            target = [target]
        return np.array(features), np.array(target)

    def construct_and_test(self):
        train_short_files = self.read_short_files("short_files_train.txt")
        test_short_files = self.read_short_files("short_files_test.txt")
        train_short_files = []
        test_short_files = []

        outliers = []
        #Maybe remove invalid features before finding outliers
        for i in range(len(self.groupings)): #Find outliers in each grouping
            tmp = self.iqr.return_outliers(self.groupings[i], [], short_files=train_short_files)
            outliers = outliers + tmp 
        print(f"len(outliers): {len(outliers)}")

        """outliers_test = [] #This does not make sense to do, as this method can not filter outliers unless it knows what sensor it is, something one would not know for new time series
        for i in range(len(self.groupings)): #Find outliers in each grouping
            tmp = self.iqr.return_outliers(self.groupings[i], [], short_files=train_short_files, filepath="comprehensive_features_10m_test.txt")
            outliers_test = outliers_test + tmp 
        print(f"len(outliers_test): {len(outliers_test)}")"""

        train_features, train_inv, train_file_dict = self.read_features_from_file("comprehensive_features_10m_train.txt", [], short_files=train_short_files)
        test_features, test_inv, test_file_dict = self.read_features_from_file("comprehensive_features_10m_test.txt", [], inv_features=train_inv, short_files=test_short_files)
        total_invalid = train_inv + test_inv
        total_invalid = []

        train_feat, train_invalid, train_file_dict = self.read_features_from_file("comprehensive_features_10m_train.txt", [], inv_features=total_invalid, short_files=train_short_files)
        train_feat, train_target, train_file_list = self.dict_to_arrays(train_feat, train_file_dict)

        gradient_boost = self.clf.fit(train_feat, train_target)



        test_feat, test_invalid = self.read_features_from_file_grouped_by_sensor("comprehensive_features_10m_test.txt", outliers, inv_features=total_invalid, short_files=test_short_files)
        #print(f"test_feat: {test_feat}")
        ans = []
        count = 0
        for sensor in test_feat: 

            for sensor_name in test_feat[sensor]: #Do for every unique sensor 
                
                feat, target = self.dict_to_array_grouped(test_feat[sensor][sensor_name], sensor)
                a = gradient_boost.predict(feat)
                values, counts = np.unique(a, return_counts=True)
                tmp = values[counts==counts.max()]
                if len(tmp)> 1:
                    tmp = -1
                else:
                    tmp=tmp[0]
                ans.append([tmp, sensor])
        self.organize_results(ans)
        return ans
    
    def organize_results(self, ans):
        correct = 0
        wrong = 0
        unresolved = 0
        for el in ans:
            if el[0] == -1:
                unresolved += 1
            elif el[0] == el[1]:
                correct += 1
            else:
                wrong += 1
        print(f"correct: {correct}")
        print(f"wrong: {wrong}")
        print(f"unresolved: {unresolved}")
        print(f"score: {correct / (wrong + unresolved + correct)}")


    def organize_results_proper(self, ans):
        ans_dict = {}
        ans_list = []
        for i in range(10):
            ans_dict[i] = {}
            correct = 0
            wrong = 0
            inconclusive = 0
            for j in range(len(ans)):
                if i == ans[j][1]:

                    if ans[j][0] == -1:
                        inconclusive += 1
                    elif ans[j][0] == ans[j][1]:
                        correct += 1
                    else:
                        wrong += 1
                        if ans[j][0] not in ans_dict[i]:
                            ans_dict[i][ans[j][0]] = 1
                        else:
                            ans_dict[i][ans[j][0]] += 1
            ans_list.append([correct, wrong, inconclusive])
        for i in range(10):
            print(f"sensor: {i}")
            print(f"distribution: {ans_list[i]}")
            print(f"wrongly infered: {ans_dict[i]}")
        return





"""values, counts = np.unique(infered, return_counts=True)

                tmp = values[counts == counts.max()]

                if len(tmp) == 1:
  
                    infered = tmp #Means undecided
                else:
                    infered = -1
                lst = [list(infered)[0], sensor, sensor_name]
        
                ans.append(lst)

        return ans
"""




if __name__ == "__main__":
    grad = gradient_boost_compact()
    feature_dict, invalid, sensor_dict = grad.read_features_from_file_grouped_by_sensor("comprehensive_features_10m_test.txt", [])
    feature_dict_train, invalid_train, sensor_dict_train = grad.read_features_from_file_grouped_by_sensor("comprehensive_features_10m_train.txt", [])
    test_lst = []
    train_lst = []

    for sensor in sensor_dict:
        print(f"{sensor}: {sensor_dict[sensor]}")
        test_lst.append(sensor)
    for sensor in sensor_dict_train:
        print(f"{sensor}: {sensor_dict_train[sensor]}")
        train_lst.append(sensor)
    count = 0
    print(test_lst)
    print(train_lst)
    for el in test_lst:
        if el in train_lst:
            count += 1
    print(count)
    print(len(train_lst))
    print(len(test_lst))
    #answers = grad.construct_and_test()
    #print(f"len(answers): {len(answers)}")
    #grad.organize_results_proper(answers)