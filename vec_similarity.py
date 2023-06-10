
from IQR_outliers import IQR


class vec_similarity():

    def __init__(self):
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
        
        self.groupings = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]

    def read_features_from_file(self, filename, outliers, inv_features= []):
        print("reading from file")
        feature_dict = {}
        file_dict = {}
        f = open(filename, "r", encoding="UTF-8")
        invalid_features = inv_features
        for line in f:
            line = line.strip()
            line = line.replace("&", " ")
            line = line.replace("//", "/")
            line = line.replace(", ", "-") #Makes sure we do not split within feature names (some have spaces)
            line = line.split(" ")
            sensor = line[0].split("/")[3].strip()
            if line[0].strip() in outliers: #Skips sensor if it is not relevant
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
            
            for i in range(1, len(line)-1, 2):
            
                #if line[i+1] == "nan" and line[i] not in invalid_features:
                if line[i+1] == "nan":
                    sensor_features[line[i]] = 0
                    invalid_features.append(line[i])
                else:        
                    sensor_features[line[i]] = float(line[i+1])
            
            feature_dict[sensor].append(sensor_features)
        
        #feature_dict = self.remove_invalid_values(feature_dict, invalid_features)
        return feature_dict, invalid_features, file_dict
    
    def remove_invalid_values(self, dict, invalid_list):
        print("removing invalid features")
        #print(f"invalid_list: {invalid_list}")
        for invalid_feature in invalid_list:
            for sensor in dict:
                for el in dict[sensor]:
                    el.pop(invalid_feature, None)
        return dict
    
    def find_max_value_features(self, feature_dict):
        print("find maximum values")
        max_dict = {}

        for sensor in feature_dict:
            for individual_sensor in feature_dict[sensor]:
                for feature in individual_sensor:
                    if feature not in max_dict:
                        max_dict[feature] = individual_sensor[feature]
                    elif max_dict[feature] < individual_sensor[feature]:
                        max_dict[feature] = individual_sensor[feature]
        return max_dict
    
    def normalize_features(self, max_dict, feature_dict):
        print("normalizing features")
        for sensor in feature_dict: #Sensor --> list of dicts
            #print(f"sensor: {sensor}")
            for features in feature_dict[sensor]: #Dicts
                #print(f"feature in sensor: {features}")
                for feature_name in features:
                    if max_dict[feature_name] != 0:
                        features[feature_name] = features[feature_name] / max_dict[feature_name]
                    else:
                        features[feature_name] = features[feature_name]
        return feature_dict
    
    def features_average_values(self, feature_dict):
        avg_features = {}
        print("averaging features")
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

    def calc_distance(self, average_features, normalized_features):
        infered_sensors = []
        for sensor in normalized_features:
            print(f"calculating distance for {sensor}")
            for individual_sensor in normalized_features[sensor]:
                min_dist = [float("inf"), "sens_name", sensor]
                
                for average_sensor in average_features:
                    tmp_dist = 0
                    for feature in individual_sensor:
                        tmp_dist += abs(average_features[average_sensor][feature] - individual_sensor[feature])
                    if tmp_dist < min_dist[0]:
                        min_dist[0] = tmp_dist
                        min_dist[1] = average_sensor
                infered_sensors.append(min_dist)
        return infered_sensors





if __name__ == "__main__":
    v = vec_similarity()
    iqr = IQR()

    train_features, train_inv, train_file_dict = v.read_features_from_file("comprehensive_features_10m_train.txt", [])
    test_features, test_inv, test_file_dict = v.read_features_from_file("comprehensive_features_10m_test.txt", [], inv_features=train_inv)
    total_invalid = train_inv + test_inv
    print(f"len total_invalid: {len(set(total_invalid))}")
    outliers = []
    for i in range(len(v.groupings)): #Find outliers in each grouping
        tmp = iqr.return_outliers(v.groupings[i], [])
        outliers = outliers + tmp 

    features, invalid_features, train_file_dict = v.read_features_from_file("comprehensive_features_10m_train.txt", outliers)
    
    max_values = v.find_max_value_features(features)
    normalized_features = v.normalize_features(max_values, features)
    average_features = v.features_average_values(normalized_features)


    test_features, test_invalid_features, test_file_dict = v.read_features_from_file("comprehensive_features_10m_test.txt", [], inv_features=total_invalid)
    test_features_norm = v.normalize_features(max_values, test_features)
    infered = v.calc_distance(average_features, test_features_norm)

    correct = 0
    wrong = 0
    for el in infered:
        if el[1] == el[2]:
            correct += 1
        else:
            wrong += 1
    print(f"correct: {correct}")
    print(f"wrong: {wrong}")

    answer_dict = {}
    other_ans = {}
    for el in infered:
        #print(el)
        if el[2] not in answer_dict:
            answer_dict[el[2]] = [0,0]
            other_ans[el[2]] = {}
        if el[1] == el[2]:
            answer_dict[el[2]][0] += 1 #Number of correct answers
        else: 
            answer_dict[el[2]][1] += 1 #Number of wrong answers
            if el[1] not in other_ans[el[2]]:
                other_ans[el[2]][el[1]] = 1
            else:
                other_ans[el[2]][el[1]] += 1
    
    for el in other_ans:
        print(f"{el}: {answer_dict[el]}")
        print(f"other_suggestions: {other_ans[el]}")



    """
    print(max_values)
    for sensor in features:
        for individual in features[sensor]:
            for feature in individual:
                if individual[feature] == "nan":
                    print(f"nan value: {feature}-{individual[feature]}")
"""