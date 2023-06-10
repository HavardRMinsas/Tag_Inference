import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



class detect_outliers():

    def __init__(self):
        self.use_classes = [
        'Outside_Air_Temperature_Sensor',
        #'Chilled_Water_Return_Temperature_Sensor', 'Chilled_Water_Supply_Temperature_Sensor', 'Hot_Water_Supply_Temperature_Sensor', 'Preheat_Supply_Air_Temperature_Sensor', 'Return_Air_Temperature_Sensor', 'Return_Water_Temperature_Sensor', 'Supply_Air_Temperature_Sensor',
        #Cooling_Valve', 'Reheat_Valve', 'Valve',
        #'Differential_Pressure_Sensor',
        #'Discharge_Air_Static_Pressure_Sensor', 'Supply_Air_Static_Pressure_Sensor',
        #'Heat_Exchanger', 'Variable_Frequency_Drive',
        #'Return_Fan', 'Supply_Fan',
        #'Power_Sensor',
        #'Pump',
        #'Energy_Sensor'
        ]


    def read_features_from_file(self, filename, inv_features= []):
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
            if sensor not in self.use_classes: #Skips sensor if it is not relevant
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
            
                if line[i+1] == "nan" and line[i] not in invalid_features:
                    sensor_features[line[i]] = "nan"
                    invalid_features.append(line[i]) #Saves all of our features that have invalid values
                else:        
                    sensor_features[line[i]] = float(line[i+1])
            
            feature_dict[sensor].append(sensor_features)
        
        feature_dict = self.remove_invalid_values(feature_dict, invalid_features)
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
    


if __name__ == "__main__":
    detect_out = detect_outliers()
    
    train_features, invalid_features, file_dict = detect_out.read_features_from_file("catch22_10m_train.txt")


    feature_names = ["DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac",  "CO_FirstMin_ac",  "CO_HistogramAMI_even_2_5",  "CO_trev_1_num",  "MD_hrv_classic_pnn40",  "SB_BinaryStats_mean_longstretch1",  "SB_TransitionMatrix_3ac_sumdiagcov",  "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",  
                 "IN_AutoMutualInfoStats_40_gaussian_fmmi",  "FC_LocalSimple_mean1_tauresrat",  "DN_OutlierInclude_p_001_mdrmd",  "DN_OutlierInclude_n_001_mdrmd",
                  "SP_Summaries_welch_rect_area_5_1", "SB_BinaryStats_diff_longstretch0", "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1", "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",  "SP_Summaries_welch_rect_centroid", "FC_LocalSimple_mean3_stderr", "DN_Mean", "DN_Spread_Std"]
    
    train_features_df, target_df, file_df = detect_out.dict_to_arrays(train_features, file_dict)

    train_features_df = pd.DataFrame(train_features_df, columns=feature_names)

    t_plot = train_features_df.boxplot(column=feature_names[17:20], return_type="axes", boxprops=dict(linewidth=5.0, color="black"), whiskerprops=dict(linewidth=5.0, color="lightblue"), medianprops=dict(linewidth=5.0, color="lightgreen"), 
                                                                                                        flierprops=dict(linewidth=5.0, color="pink"), capprops=dict(linewidth=5.0, color="purple"), fontsize=20)
    print(t_plot)

    plt.show()

