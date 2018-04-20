'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'
'  This script shows an example how to compute the No-Reference (NR) 
'  features for EPFL-PoliMi 4CIF dataset, using compute_features.m
'
'  Written by Jari Korhonen, Shenzhen University.
'
'  To use the script, CSV files containing the features for the
'  EPFL-PoliMi 4CIF dataset video sequences should be downloaded and
'  the impaired sequences decoded in folder f:/epfl-polimi. If you use
'  some other path, change the path in the script accordingly. The
'  feature files can be generated using the related Matlab script
'  EPFL_PoliMi_4CIF_example.m
'
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Load libraries
import pandas
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import numpy as np

# ===========================================================================
# This function trains and validates regression model with one single content
# missing (i.e. leave-one-out validation)
#
def analysis(missing_content):
    
    # Initialization
    
    # Put the path to the feature files here
    path = "f:/epfl-polimi" 
    
    # The source contents here
    contents = ["CROWDRUN","DUCKSTAKEOFF","HARBOUR","ICE","PARKJOY","SOCCER"]
    
    # The MOS scores (averages for EPFL and PoliMi scores)
    MOS = [4.23619,3.32156,3.6414, 3.0716, 0.64216,0.74123,2.56719,2.76266, 
           2.29809,1.75318,1.19901,1.73928,3.75763,4.14346,3.51529,3.32815, 
           0.49251,0.66002,2.08555,2.7432, 1.89227,1.40177,0.71308,1.28187, 
           4.13761,3.64018,4.11086,4.03647,1.11672,1.56818,3.17968,3.1473, 
           2.28403,2.36008,1.89132,2.05709,3.97563,4.37436,3.58844,3.59108,
           0.70476,0.68054,2.8586, 2.37681,2.51964,1.89273,1.5172, 1.07961, 
           3.56554,3.88391,3.15553,3.13514,0.93952,0.45513,2.24453,2.40908, 
           1.48779,1.82523,1.33646,1.44142,3.96985,3.95962,3.33889,3.65777, 
           0.61223,0.73793,3.06467,2.53293,2.07506,1.85548,1.12259,1.4506] 
    
    # Load features for training
    temp = []
    for i in range(0,6):
        if i != missing_content:
            for j in range(1,13):
                filepath = "%s/%s_seq_%02d.csv" % (path,contents[i],j)
                df = pandas.read_csv(filepath, skiprows=[], header=None)  
                if np.size(temp, axis=0)>0:
                    temp = np.append(temp, df.values, axis=0)
                else:
                    temp = df.values
        
    train_array = np.asarray(temp)  
    X_train = train_array[:,range(1,25)]    # Features here
    Y_train = train_array[:,0]              # FR distortion index here
                
    # Load features for testing
    temp_ta = []
    num_frames = []
    i = missing_content
    for j in range(1,13):
        filepath = "%s/%s_seq_%02d.csv" % (path,contents[i],j)
        df = pandas.read_csv(filepath, skiprows=[], header=None)       
        if np.size(temp_ta, axis=0)>0:
            temp_ta = np.append(temp_ta, df.values, axis=0)
        else:
            temp_ta = df.values 
        num_frames.append(np.size(df.values, axis=0))
        
    test_array = np.asarray(temp_ta)
        
    X_validate = test_array[:,range(1,25)]  # Features here
    E_FR = test_array[:,0]                  # FR distortion index here
    
    # ======================================================================= 
    # Train the regression model here. We have used three different
    # models: SVR, Random Forest and Multi-Layer Perceptron
    
    model = SVR(kernel='rbf', gamma=0.9, C=pow(2,6), epsilon=0.1)
    '''
    model = RandomForestRegressor(n_estimators=100, criterion='mse', 
                                  max_depth=None, min_samples_split=2, 
                                  min_samples_leaf=1, max_features='auto', 
                                  max_leaf_nodes=None, bootstrap=True, 
                                  oob_score=False, random_state=0, verbose=0, 
                                  warm_start=False)
    '''
    '''
    model = MLPRegressor(hidden_layer_sizes=(16,),  activation='logistic', 
                         solver='lbfgs', alpha=0.1, batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.01, 
                         power_t=0.5, max_iter=1000, shuffle=True,
                         random_state=0, tol=0.0001, verbose=False, 
                         warm_start=False, momentum=0.9, 
                         nesterovs_momentum=True,
                         early_stopping=False, validation_fraction=0.1, 
                         beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    '''    
    
    model.fit(X_train, Y_train)    

    E_FR_pred = model.predict(X_validate)
    
    # Compute the predicted E_SQ_pred values
    E_SQ = [np.mean(E_FR[range(0, num_frames[0])])]
    E_SQ_pred = [np.mean(E_FR_pred[range(0, num_frames[0])])]
    idx = num_frames[0]
    for i in range(1,len(num_frames)):
        E_SQ = np.append(E_SQ, np.mean(E_FR[range(idx, idx+num_frames[i])]))
        E_SQ_pred = np.append(E_SQ_pred, np.mean(
                              E_FR_pred[range(idx, idx+num_frames[i])]))
        idx = idx + num_frames[i]
        
    # Find subjective scores
    filepath = "f:/epfl-polimi/subj_scores_4CIF.csv"
    df = pandas.read_csv(filepath, skiprows=[], header=None)
    array = df.values
    MOS = array[range(missing_content*12,missing_content*12+12),2]  
    
    # Print the results for this case
    print('Results for: ', contents[missing_content])
    print('E_FR^ vs E_FR (PLCC, SROCC):', 
          scipy.stats.pearsonr(E_FR_pred,E_FR)[0],
          scipy.stats.spearmanr(E_FR_pred,E_FR)[0])
    print('E_SQ^ vs E_SQ (PLCC, SROCC):',    
          scipy.stats.pearsonr(E_SQ_pred,E_SQ)[0],
          scipy.stats.spearmanr(E_SQ_pred,E_SQ)[0])
    print('E_SQ^ vs MOS (PLCC, SROCC):', 
          scipy.stats.pearsonr(E_SQ_pred,MOS)[0],
          scipy.stats.spearmanr(E_SQ_pred,MOS)[0])
    print('--------------------------------------------')

    out = [E_SQ, E_SQ_pred, MOS, E_FR, E_FR_pred]     

    return out
    
# ===========================================================================
# Here starts the main part of the script
#
all_scores = []
E_SQ_all = []
E_SQ_pred_all = []
MOS_all = []
E_FR_all = []
E_FR_pred_all = []

# The main loop
for i in range(0,6):
    new_scores = analysis(i)
    E_SQ_all.extend(new_scores[0])
    E_SQ_pred_all.extend(new_scores[1])
    MOS_all.extend(new_scores[2])
    E_FR_all.extend(new_scores[3])
    E_FR_pred_all.extend(new_scores[4])
    
# Present the aggregate results here
    
plt.xlabel('Mean Opinion Score')
plt.ylabel('Error Visibility Index (E_SQ / E^_SQ)')
plt.title('E_SQ and E_SQ^ versus MOS (4CIF)')
plt.plot(MOS_all,E_SQ_all, 'ro', MOS_all,E_SQ_pred_all, 'ko')

print('===================================================')
print('Aggregate results ')
print('E_seq vs. E^_seq (PLCC, SROCC): ',
      scipy.stats.pearsonr(E_SQ_all,E_SQ_pred_all)[0],
      scipy.stats.spearmanr(E_SQ_all,E_SQ_pred_all)[0])
print('E^_seq vs. MOS (PLCC, SROCC): ',
      scipy.stats.pearsonr(MOS_all,E_SQ_pred_all)[0],
      scipy.stats.spearmanr(MOS_all,E_SQ_pred_all)[0])
print('E_frame vs. E^_frame (PLCC, SROCC): ',
      scipy.stats.pearsonr(E_FR_all,E_FR_pred_all)[0],
      scipy.stats.spearmanr(E_FR_all,E_FR_pred_all)[0])
print('===================================================')

