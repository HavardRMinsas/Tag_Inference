import numpy as np
import matplotlib.pyplot as plt
from IQR_outliers import IQR
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import tree
from pyts.classification import TimeSeriesForest



class multiple():

    def __init__(self):     
        self.iqr = IQR()   
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


    def read_features_from_file(self, filename, outliers, inv_features= [], short_files=[]):
        print("reading from file")
        feature_dict = {}
        file_dict = {}
        f = open(filename, "r", encoding="UTF-8")
        invalid_features = inv_features
        c = 0
        for line in f:
            c += 1
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
            
            for i in range(1, len(line)-1, 2):
            
                #if line[i+1] == "nan" and line[i] not in invalid_features:
                if line[i+1] == "nan":
                    sensor_features[line[i]] = 1.0
                    invalid_features.append(line[i])
                elif line[i+1] == "inf":
                    sensor_features[line[i]] = 100000.0
                else:        
                    sensor_features[line[i]] = float(line[i+1])
            
            feature_dict[sensor].append(sensor_features)
        #feature_dict = self.remove_invalid_values(feature_dict, list(set(invalid_features)))
        print(f"num_files: {c}")
        return feature_dict, invalid_features, file_dict


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
    
    def get_data(self):

        #train_features, train_inv, train_file_dict = self.read_features_from_file("comprehensive_features_10m_train.txt", [], short_files=[])
        #test_features, test_inv, test_file_dict = self.read_features_from_file("comprehensive_features_10m_test.txt", [], inv_features=train_inv, short_files=[])
        #total_invalid = train_inv + test_inv
        total_invalid = []
        
        outliers = []
        for i in range(len(self.groupings)): #Find outliers in each grouping
            tmp = self.iqr.return_outliers(self.groupings[i], [])
            outliers = outliers + tmp 
        print(f"len(outliers): {len(outliers)}")
        
        

        train_features, train_inv, train_file_dict = self.read_features_from_file("comprehensive_features_10m_train.txt", outliers, inv_features=total_invalid, short_files=[])
        train_features, train_target, train_file_list = self.dict_to_arrays(train_features, train_file_dict)

        test_features, test_inv, test_file_dict = self.read_features_from_file("comprehensive_features_10m_test.txt", [], inv_features=total_invalid, short_files=[])
        test_features, test_target, test_file_list = self.dict_to_arrays(test_features, test_file_dict)

        return train_features, train_target, test_features, test_target, train_file_list, test_file_list

    def organize_results(self, predicted_answers, target):
        results = {}
        other = []
        for i in range(len(self.groupings)):
            print(self.groupings[i])
            correct = 0
            wrong = 0
            other_suggestions = {}
            for j in range(len(predicted_answers)):
                if target[j] == i:
                    if predicted_answers[j] == target[j]:
                        correct += 1
                    else:
                        wrong += 1
                        if predicted_answers[j] not in other_suggestions:
                            other_suggestions[predicted_answers[j]] = 1
                        else:
                            other_suggestions[predicted_answers[j]] += 1
            results[i] = [correct, wrong]
            other.append(other_suggestions)
            print(f"correct: {correct}")
            print(f"wrong: {wrong}")
            print(f"other_suggestions: {other_suggestions}")
        for sensor in results:
            print(f"{sensor}: correct: {results[sensor][0]}, wrong: {results[sensor][1]}")
            print(f"other suggestions: {other[sensor]}")
        #print(f"results: {results}")
        #print(f"other: {other}")
        return results, other
    

    def histogram_classifier(self, train_feat, train_targ, test_feat, test_targ):
        clf = HistGradientBoostingClassifier(max_iter=1000, verbose=10, random_state=0, learning_rate=0.02, early_stopping=False) #0.8734602463605823 with learning rate = 0.1 and 10k estimatorrs, 0.8913773796192609 with learning rate 0.05 and estimators 10k, 0.9171332586786114 with 0.2 learning rate and 10k estimators,???? with learning rate 0.1 and estimators = 10k
        clf.fit(train_feat, train_targ)
        print("Histogram score:")
        print(clf.score(test_feat, test_targ))
        ans = clf.predict(test_feat)
        return ans
    
    def random_forest(self, train_feat, train_targ, test_feat, test_targ):
        clf = RandomForestClassifier(n_estimators=400, max_depth=16, random_state=0, verbose=10) #Managed 0.880552444 with 400 estimators and max_depth=16, 0.882045 for estimators = 10 000
        clf.fit(train_feat, train_targ)
        print("random forest score:")
        print(clf.score(test_feat, test_targ))
        ans = clf.predict(test_feat)
        return ans
    
    def neaural_network(self, train_feat, train_targ, test_feat, test_targ):
        clf = MLPClassifier(learning_rate_init=0.1, hidden_layer_sizes=(1,5), random_state=0, verbose=10)
        clf.fit(train_feat, train_targ)
        print("MLP classifier score:")
        print(clf.score(test_feat, test_targ))
        ans = clf.predict(test_feat)
        return ans
    
    def k_neighbours(self, train_feat, train_targ, test_feat, test_targ):
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance") #0.712952 for n_neighbours=3, works slightly better without outliers, 0.703994027622247 for n_neighbours = 3
        clf.fit(train_feat, train_targ)
        print("Kneighbors score:")
        print(clf.score(test_feat, test_targ))
        ans = clf.predict(test_feat)
        return ans
    
    def decision_tree(self, train_feat, train_targ, test_feat, test_targ):
        clf = tree.DecisionTreeClassifier(max_depth=12, random_state=0)
        clf.fit(train_feat, train_targ)

        #sensor_names = ['G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9']
        #tree.plot_tree(clf, fontsize=8, class_names=sensor_names, max_depth=2)
        #plt.show()
        print("Decision tree score:")
        print(clf.score(test_feat, test_targ))
        ans = clf.predict(test_feat)
        return ans
    
    def gradient_boost(self, train_feat, train_targ, test_feat, test_targ, random_state=0):
        clf = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, random_state=random_state, verbose=10)
        clf.fit(train_feat, train_targ)
        print("Gradient boost score:")
        s = clf.score(test_feat, test_targ)
        print(s)
        ans = clf.predict(test_feat)
        return ans

    
    def oneVsOne_classifier(self, train_feat, train_targ, test_feat, test_targ):
        clf = OneVsOneClassifier(GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, max_features=1000, random_state=0, verbose=10)) #0.8857782 for estimators = 200, 0.8872713 for estimators = 400
        #clf = OneVsOneClassifier(GradientBoostingClassifier(n_estimators=4, learning_rate=0.1, max_depth=6, max_features=1000, random_state=0, verbose=10))
        clf.fit(train_feat, train_targ)
        print("One vs One score:")
        print(clf.score(test_feat, test_targ))
        return clf
    
    def oneVsRest_classifier(self, train_feat, train_targ, test_feat, test_targ):
        clf = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, max_features=1000, random_state=0, verbose=10)) #Estimators = 200, score: 0.9033221; estimators=400, score= 0.9081746920492721
        #clf = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=4, learning_rate=0.1, max_depth=6, max_features=1000, random_state=0, verbose=10))
        clf.fit(train_feat, train_targ)
        print("One vs rest score:")
        print(clf.score(test_feat, test_targ))
        ans = clf.predict(test_feat)
        return ans

    def gradient_regressor(self, train_feat, train_targ, test_feat, test_targ):
        clf = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=6, random_state=0, verbose=10)
        clf.fit(train_feat, train_targ)
        print(f"Gradient boost score: {clf.score(test_feat, test_targ)}")
        return

    def normalize(self, features):
        #A function that normalize all features to values between -1 and 1, maybe it has an effect
        pass

    def time_series_forest(self, train_feat, train_targ, test_feat, test_targ):
        clf = TimeSeriesForest(random_state=0, n_windows=3, max_depth=16, max_features='sqrt', n_estimators=10000)
        clf.fit(train_feat, train_targ)
        print(f"score: {clf.score(test_feat, test_targ)}")


    def find_common_invalid_feat(self, invalid_list):
        common_invalid = {}
        for i in range(10): #Check for each sensor
            tmp = []
            common_invalid[i] = []
            count = 0
            for invalid_dict in invalid_list: #We want to check all invalid sensors for a given key, and return the intersection of these
                try:
                    if i not in invalid_dict:
                        continue
                    elif len(tmp) == 0:
                        tmp = invalid_dict[i]
                    else:
                        set(tmp).intersection(invalid_dict[i])
                        count += 1 
                except:
                    continue
            if count != 0: #Verify that we did an intersection between at least to lists
                common_invalid[i] = tmp
        for key in common_invalid:
            print(f"sensor: {key}")
            print(f"common_invalid: {common_invalid[key]}")

        f = open("common_invalid_10m.txt", "w", encoding="UTF-8")
        f.write(str(common_invalid))
        f.close()

     #tmp = set(tmp).intersection(tmp_dict[el][i])


    def find_invalid_files(self, ans, target, file_list):
        invalid = {}
        for i in range(10):
            invalid[i] = [] #Initiate the dict with each sensor group

        for i in range(len(ans)):
            if ans[i] != target[i]:
                invalid[target[i]].append((file_list[i], ans[i]))
        for el in invalid:
            print(f"sensor: {el}")
            print(f"invalid sensors: {invalid[el]}")
        return invalid

if __name__ == "__main__":
    classifier = multiple()
    train_feat, train_targ, test_feat, test_targ, train_file_list, test_file_list = classifier.get_data()
    
    #kneigh_ans = classifier.k_neighbours(train_feat, train_targ, test_feat, test_targ)
    #kneigh_invalid = classifier.find_invalid_files(kneigh_ans, test_targ, test_file_list)

    #dec_tree_ans = classifier.decision_tree(train_feat, train_targ, test_feat, test_targ) #Default value: -999 -->0.825307  default value: 0 --> 0.83426651, removing invalid --> 0.8290406, default value of -12345 --> 0.8253079, default value: 1 --> 0.83501306
    #dec_tree_invalid = classifier.find_invalid_files(dec_tree_ans, test_targ, test_file_list)

    #random_forest_ans = classifier.random_forest(train_feat, train_targ, test_feat, test_targ)
    #random_forest_invalid = classifier.find_invalid_files(random_forest_ans, test_targ, test_file_list)

    #hist_ans = classifier.histogram_classifier(train_feat, train_targ, test_feat, test_targ)
    #classifier.organize_results(hist_ans, test_targ)
    #hist_invalid = classifier.find_invalid_files(hist_ans, test_targ, test_file_list)

    grad_ans = classifier.gradient_boost(train_feat, train_targ, test_feat, test_targ)
    classifier.organize_results(grad_ans, test_targ)
    print("for depth = 6")
    #print("Results for 5h intervals")
    #grad_invalid = classifier.find_invalid_files(grad_ans, test_targ, test_file_list)

    #invalid_list = [dec_tree_invalid, hist_invalid, random_forest_invalid, kneigh_invalid, grad_invalid]


    #classifier.find_common_invalid_feat(invalid_list)

    #ans = classifier.neaural_network(train_feat, train_targ, test_feat, test_targ)
    #kneighbours = classifier.k_neighbours(train_feat, train_targ, test_feat, test_targ)
    #classifier.organize_results(kneighbours, test_targ)
    #onevone = classifier.oneVsOne_classifier(train_feat, train_targ, test_feat, test_targ)
    #onevall = classifier.oneVsRest_classifier(train_feat, train_targ, test_feat, test_targ)

    #classifier.time_series_forest(train_feat, train_targ, test_feat, test_targ)
"""
    grad1, s1 = classifier.gradient_boost(train_feat, train_targ, test_feat, test_targ, 0)
    grad2, s2 = classifier.gradient_boost(train_feat, train_targ, test_feat, test_targ, 1)
    grad3, s3 = classifier.gradient_boost(train_feat, train_targ, test_feat, test_targ, 2)
    grad4, s4 = classifier.gradient_boost(train_feat, train_targ, test_feat, test_targ, 3)
    grad5, s5 = classifier.gradient_boost(train_feat, train_targ, test_feat, test_targ, 4)

    eclf = VotingClassifier(estimators=[("grad1", grad1), ("grad2", grad2), ("grad3", grad3), ("grad4", grad4), ("grad5", grad5)], verbose=10, voting="soft")
    eclf = eclf.fit(train_feat, train_targ)
    print(f"eclf score: {eclf.score(test_feat, test_targ)}")

    print(f"Individual scores: {[s1, s2, s3, s4, s5]}")

    """



