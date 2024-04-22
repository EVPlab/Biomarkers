"""
# UK Biobank Data Setup and Logistic Regression Classification

This script sets up the UK Biobank dataset containing participants with data from brain imaging, bone imaging, and psychosocial assessments. 
It instantiates a logistic regression machine learning model with nested cross-validation and randomized hyperparameter search.

## Overview

The script aims to classify participants reporting a pain-associated diagnosis from diagnosis-free controls based on biological and psychosocial features.

## Features Included

- Brain imaging (DWI, T1, rsfMRI)
- Bone scans (DXA)
- Psychosocial assessments

## Machine Learning Approach

- Logistic regression model
- Nested cross-validation
- Randomized hyperparameter search

## Purpose

The purpose of this script is to provide a robust framework for classifying participants based on their pain-associated diagnosis using a combination of biological and psychosocial features.
"""

# Load data
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from confounds import Residualize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from snapml import LogisticRegression as SnapLR
from snapml import SupportVectorMachine as SnapSVM
from snapml import SnapBoostingMachineClassifier as SnapBoost
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, matthews_corrcoef, average_precision_score
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')
from pymatch.Matcher import Matcher
from pymatch.functions import find_nearest_n
import prince
import numpy as np
import pandas as pd
import pickle
import random
import time
import sys
import os

###################################################### Functions to run classification model ############################################################

def clean_data(data):
        
    motion_out = confounds[confounds['fMRIMotion_T2'] > np.percentile(confounds['fMRIMotion_T2'].dropna(), 99.75)].eid # SD version: UKB.Motion_T2.mean() + (5 * UKB.Motion_T2.std())
    CNR_out = confounds[confounds['InvertedSNR_T2'] > np.percentile(confounds['InvertedSNR_T2'].dropna(), 99.75)].eid
    align_out = confounds[confounds['DescrepancyNonLinearAlignment_T2'] > np.percentile(confounds['DescrepancyNonLinearAlignment_T2'].dropna(), 99.75)].eid
    ids_out = pd.concat([motion_out,CNR_out,align_out])
    ids_out = ids_out.drop_duplicates()
    data = data[~data.eid.isin(ids_out)]
            
    return data

##################################################################################################################


def pca_to_weights(clf):
        
    try:
        w = clf.best_estimator_[-1].coef_.reshape(1,-1)
    except (AttributeError, ValueError):
        w = clf.best_estimator_[-1].feature_importances_.reshape(1,-1)
    pc = clf.best_estimator_[-2].components_
    pcw = pc * w.T
    pcws = pcw.T.sum(axis=1)
    
    return pcws

def get_feat_imp(clf,X_train,feature_labels):

    feat_imp = []

    try:
        w = clf.best_estimator_[-1].feature_importances_.ravel()
        importances = pd.DataFrame(w,index=feature_labels,columns=['importance'])
    except (AttributeError, ValueError):
        try:
            w = clf.best_estimator_[-1].coef_.ravel()
            importances = pd.DataFrame(w,index=feature_labels,columns=['weight'])
        except (AttributeError, ValueError):
            try:
                pcw = pca_to_weights(clf)
                importances = pd.DataFrame(pcw,index=feature_labels,columns=['weight'])
            except (AttributeError, ValueError):
                importances = np.nan
                                        
    # Haufe transform
    try:
        prob_tr = clf.best_estimator_[-1].predict_proba(X_train)[:,1]
    except:
        prob_tr = clf.best_estimator_[-1].decision_function(X_train)

    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    w = [np.cov(i, prob_tr)[0,1] for i in X_train.T]
    hauf = pd.DataFrame(w,index=feature_labels,columns=['Haufe'])
                
    feat_imp.append(importances)
    feat_imp.append(hauf)
                
    return feat_imp


##################################################################################################################
    
    
def compute_metrics(y_true, y_pred, y_prob):
    """Compute various evaluation metrics."""
    acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return acc, auc, mcc, ap

def get_probabilities(model, X):
    """Get model probabilities. Use decision function if predict_proba is not available."""
    try:
        probabilities = model.predict_proba(X)[:, 1]
    except AttributeError:
        probabilities = model.decision_function(X)
    return probabilities

def convert_to_float32(item):
    if isinstance(item, np.ndarray):
        return item.astype(np.float32)
    elif isinstance(item, pd.DataFrame):
        return item.astype(np.float32)
    elif isinstance(item, dict):
        return {k: convert_to_float32(v) for k, v in item.items()}
    else:
        return item

def get_results(model, mod_name, X_train, y_train, X_test, y_test, eid_test, X_o_test, y_o, eid_o, X_l_test, eid_l, feature_imp, targ, rs, it, fold):
    results_dic = {}

    # Get predicted probabilities and values for train and test sets
    prob_train = get_probabilities(model, X_train)
    prob_test = get_probabilities(model, X_test)
    prob_o_test = get_probabilities(model, X_o_test)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Compute metrics
    acc_train, auc_train, mcc_train, ap_train = compute_metrics(y_train, pred_train, prob_train)
    acc_test, auc_test, mcc_test, ap_test = compute_metrics(y_test, pred_test, prob_test)

    # ROC-AUC results dataframe
    norm_df = pd.DataFrame({
        'Acc_train': [acc_train], 'Acc_test': [acc_test],
        'AUC_train': [auc_train], 'AUC_test': [auc_test],
        'AP_train': [ap_train], 'AP_test': [ap_test],
        'MCC_train': [mcc_train], 'MCC_CV': [model.best_score_], 'MCC_test': [mcc_test]
    }, index=[f'{targ}_{mod_name}_{it}-{fold}'])

    # Calculate signature from estimated coefficients
    def calculate_signature(X, weights):
        return np.dot(X, weights).tolist()

    weights = feature_imp[0]['weight'].values
    signature = np.stack([eid_test.values.tolist(), calculate_signature(X_test, weights)])
    signature_o = np.stack([eid_o.values.tolist(), calculate_signature(X_o_test, weights)])

    results_dic = {
        'weights': feature_imp,
        'results_df': norm_df,
        'roc_curve': np.stack([eid_test, y_test, prob_test]),
        'roc_curve_other': np.stack([eid_o, y_o, prob_o_test]),
        'random_state': rs,
        'best_params': model.best_params_,
        'signature': signature,
        'signature_other': signature_o
    }

    # Additional computations for specific models
    if mod_name in ['T1', 'DWI', 'fMRI', 'Stacked']:
        prob_l_test = get_probabilities(model, X_l_test)
        signature_l = np.stack([eid_l.values.tolist(), calculate_signature(X_l_test, weights)])
        
        results_dic['roc_curve_longitudinal'] = np.stack([eid_l, prob_l_test])
        results_dic['signature_longitudinal'] = signature_l

    # Convert all float values in the dictionary to float32
    results_dic = convert_to_float32(results_dic)

    return results_dic
    
##################################################################################################################


def get_clf(clf):
        
        # Flat params
        if clf == 'LR_lin':
            clf = SnapLR(max_iter=10000,kernel='linear',class_weight='balanced',n_jobs=1,use_gpu=True)
            params_full = {
                                'clf__regularizer': np.logspace(-1,7,100),
        }
                
        # Flat params
        if clf == 'LR_rbf':
            clf = SnapLR(max_iter=10000,kernel='rbf',normalize=True,class_weight='balanced',n_jobs=1,use_gpu=True)
            params_full = {
                                'clf__regularizer': np.logspace(0,75),
                                'clf__gamma': np.logspace(-1,4,20),
        }
                
        if clf == 'XGB':
            clf = SnapBoost(class_weight='balanced',n_jobs=1,use_gpu=True)
            params_full = {
            'clf__num_round': [10,25,50,75,100,1000],
            'clf__learning_rate': [0.0001,0.001,0.03,0.06,0.1,0.5],
            'clf__subsample': [0.2,0.5,0.8,1],
            'clf__max_depth':list(range(1,11)),
            'clf__lambda_l2':np.linspace(1,1000,25),
            'clf__colsample_bytree':np.linspace(0.1,1,25),
            'clf__regularizer': np.logspace(-6,3,25),
            'clf__gamma': np.logspace(-3,3,25),
    }

        if clf == 'SVM':
            clf = SnapSVM(max_iter=100000,kernel='linear',normalize=False,class_weight='balanced',n_jobs=1,use_gpu=True)
            params_full = {
                            'clf__loss' : ['hinge', 'squared_hinge'],
                            'clf__regularizer': np.logspace(-1,6,100),
    }

        # Full model
        scale = StandardScaler()
        pipe = Pipeline(steps=[
        ('scaler',scale),
        ('clf',clf)])
    
        return pipe, params_full
    
##################################################################################################################


def train_model(targ,clf,data,y_var,mod_name,modality,meta_dic,iterations,iter_l,lng_brain):
    
    # Get classifier and parameters for gridsearch
    pipe,params = get_clf(clf)

    if clf == 'XGB':
        jobs = None
        dispatch = 5
    else:
        jobs = 10
        dispatch = 2*jobs
        
    # Instantiate gridsearch estimator
    model = RandomizedSearchCV(estimator=pipe,
                                    n_iter=30,
                                    param_distributions=params,
                                    scoring=make_scorer(roc_auc_score,needs_proba=True),
                                    cv=5,
                                    return_train_score=False,
                                    n_jobs=jobs,
                                    pre_dispatch=dispatch,
                                    verbose=0,
                                    refit=True,
                                    random_state=173
                                    )
    
    # Remove modality specific null values (alternative to the above loop)
    model_vars = ['eid']+modality+[y_var]+confound_cols
    clean_eid = data[model_vars].dropna()['eid']
    data_prep = data[data.eid.isin(clean_eid)]
    # Keep only NCI-Free or subjects with illness under evaluation
    data_tmp = pd.concat([data_prep[data_prep['NCI_T2'] == 0],data_prep[data_prep[y_var] == targ]])
        
    if mod_name == 'fMRI':
        confounds = [
            'fMRIMotion_T2', 
            'Age_T2', 
            'Sex_T0',
                ]
        
    if mod_name == 'T1':
        confounds = [
            'T1Motion_T2',
            'HeadScaling_T2',
            'Age_T2', 
            'Sex_T0',
                ]
        
    if mod_name == 'DWI':
        confounds = [
            'dMRIMotion_T2',
            'HeadScaling_T2',
            'Age_T2', 
            'Sex_T0',
                ]
        
    if mod_name == 'Psychosocial':
        confounds = [
            'Age_T2', 
            'Sex_T0',
                ]
    
    if mod_name == 'Bone':
        confounds = [
            'Age_T2', 
            'Sex_T0',
                ]
    else:
        confounds = confound_cols
        
    # Predict subjects removed for CV
    data_other = data_prep[~data_prep.eid.isin(data_tmp.eid)]

    # Create feature, target, and confound matrices
    X = data_tmp[modality]
    y = data_tmp[y_var]
    C = data_tmp[confounds]
    eid = data_tmp['eid']
    
    # Other vars
    X_o = data_other[modality]
    y_o = data_other[y_var]
    C_o = data_other[confounds]
    eid_o = data_other['eid']
    
    # Longitudinal
    if mod_name == 'T1' or mod_name == 'DWI' or mod_name == 'fMRI' or mod_name == 'Stacked':
        X_l = lng_brain[modality]
        C_l = lng_brain[confounds]
        eid_l = lng_brain['eid']
    else:
        X_l_test = np.nan
        eid_l = np.nan

    for it,rs in enumerate(random.sample(range(1, 10000), iterations)):
    # Nested Cross-Validation        
        cval = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        for fold, (train, test) in enumerate(cval.split(X,y)):

            X_train = X.iloc[train,:]
            y_train = y.iloc[train]
            X_test = X.iloc[test,:]
            y_test = y.iloc[test]
            C_train = C.iloc[train,:]
            C_test = C.iloc[test,:]
            eid_test = eid.iloc[test]

            # Fit regression model to train set confounds
            resid = Residualize()
            resid.fit(X_train, C_train)
            
            # Residualize train and test sets using fitted model
            X_train = resid.transform(X_train, C_train)
            X_test = resid.transform(X_test, C_test) 
            X_o_test = resid.transform(X_o, C_o)
            
            # Residualize longitudinal brain features
            if mod_name == 'T1' or mod_name == 'DWI' or mod_name == 'fMRI' or mod_name == 'Stacked':
                X_l_test = resid.transform(X_l, C_l)        

            # For file name structure
            targ = y_var[:-3]
                    
            start = time.time()
            
            # Fit gridsearch CV model on training data
            model.fit(X_train,y_train)

            # Extract feature importance weights from train set
            feature_imp = get_feat_imp(clf=model,X_train=X_train,feature_labels=X.columns)

            # Instantiate dictionary with results for train and test set
            results_dic = get_results(model,mod_name,X_train,y_train,X_test,y_test,eid_test,X_o_test,y_o,eid_o,X_l_test,eid_l,feature_imp,targ,rs,it,fold)
            meta_dic[f'{targ}_{mod_name}_Iter-{it}_Fold-{fold}'] = results_dic
        
            iter_l.append(meta_dic[f'{targ}_{mod_name}_Iter-{it}_Fold-{fold}']['results_df'])
        
            end = time.time()
            print(f'Finished: {targ}_{mod_name}_{it}_{fold} \n Time to complete: {end-start} \n ######################################')
            
    return meta_dic, iter_l

########################################################### Run gridsearch classification and save outputs #########################################################################

def run_model(data,clf,mod_names,modalities,outcome,iterations,lng_brain):
    
    # Dictionary for appending results
    meta_dic = {}
    iter_l = []

    # Loop through set number of iterations and train classifier on pain group of interest
    for mod_ind,modality in enumerate(modalities):

            if outcome == 'diagnoses':

                    pain_NCI = [
                    #### Start 1 ####
                    'NCI_fibromyalgia_T2',
                    'NCI_polymyalgia rheumatica_T2',
                    'NCI_cervical spondylosis_T2',
                    'NCI_joint pain_T2',
                    'NCI_back pain_T2',
                    'NCI_spine arthritis/spondylitis_T2',
                    'NCI_trapped nerve/compressed nerve_T2',
                    'NCI_sciatica_T2',
                    'NCI_hernia_T2',
                    'NCI_irritable bowel syndrome_T2',
                    'NCI_gastro-oesophageal reflux (gord) / gastric reflux_T2',
                    'NCI_arthritis (nos)_T2',
                    #### Start 2 ####
                    'NCI_osteoarthritis_T2',
                    'NCI_osteoporosis_T2',
                    'NCI_rheumatoid arthritis_T2',
                    'NCI_migraine_T2',
                    'NCI_headaches (not migraine)_T2',
                    'NCI_carpal tunnel syndrome_T2',
                    'NCI_angina_T2',
                    'NCI_endometriosis_T2',
                    'NCI_gout_T2',
                    'NCI_chronic fatigue syndrome_T2',
                    'NCI_ankylosing spondylitis_T2',
                    'NCI_trigemminal neuralgia_T2',
                    #### Start 3 ####
                    'NCI_crohns disease_T2',
                    'NCI_spinal stenosis_T2',
                    'NCI_peripheral neuropathy_T2',
                    'NCI_ulcerative colitis_T2',
                    'NCI_pulmonary embolism +/- dvt_T2',
                    'NCI_chronic obstructive airways disease/copd_T2',
                    'NCI_stroke_T2',
                    'NCI_multiple sclerosis_T2',
                    'NCI_psoriatic arthropathy_T2',
                    'NCI_Post Surgical Pain_T2',
                    'NCI_parkinsons disease_T2',
                    'NCI_disc disorder_T2',
                    'NCI_peripheral vascular disease_T2'
                ]
                    targ = 1
                    for var in pain_NCI:
                    
                        if mod_names[mod_ind] == 'Biology':
                            var_str = var[:-3]
                            modality_new = data[modality].drop(columns=data.filter(like='PRS')).columns.append(PRS.filter(like=var_str).columns[:-1])
                        else:
                            modality_new = modality
                    ####### Train model and get results #######
                        meta_dic, iter_l = train_model(targ=targ,clf=clf,data=data,y_var=var,mod_name=mod_names[mod_ind],
                                        modality=modality_new,iterations=iterations,iter_l=iter_l,meta_dic=meta_dic,
                                        lng_brain=lng_brain)

            iter_df = pd.concat(iter_l)
    
        # Save results dictionary
    path_out = '/lustre06/project/6055423/UK_Biobank/Slurm/mfill/UKB_CP_Classification/Phenotypes/PickledModels/'
    with open(path_out+f'T2_{outcome}_clf-{clf}.pickle', 'wb') as file:
        pickle.dump(meta_dic, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return iter_df, meta_dic

################################################################ Run gridsearch classification and save outputs ###############################################################################

home_dir = '/lustre06/project/6055423/UK_Biobank/Files_CSV/'

confound_cols = [
            'T1Motion_T2',
            'fMRIMotion_T2', 
            'dMRIMotion_T2',
            'HeadScaling_T2',
            'Age_T2', 
            'Sex_T0',
                ]

NCI_cols = [
        'eid',
        'NCI_T2',
        #### Start 1 ####
        'NCI_fibromyalgia_T2',
        'NCI_polymyalgia rheumatica_T2',
        'NCI_cervical spondylosis_T2',
        'NCI_joint pain_T2',
        'NCI_back pain_T2',
        'NCI_spine arthritis/spondylitis_T2',
        'NCI_trapped nerve/compressed nerve_T2',
        'NCI_sciatica_T2',
        'NCI_hernia_T2',
        'NCI_irritable bowel syndrome_T2',
        'NCI_gastro-oesophageal reflux (gord) / gastric reflux_T2',
        'NCI_arthritis (nos)_T2',
        #### Start 2 ####
        'NCI_osteoarthritis_T2',
        'NCI_osteoporosis_T2',
        'NCI_rheumatoid arthritis_T2',
        'NCI_migraine_T2',
        'NCI_headaches (not migraine)_T2',
        'NCI_carpal tunnel syndrome_T2',
        'NCI_angina_T2',
        'NCI_endometriosis_T2',
        'NCI_gout_T2',
        'NCI_chronic fatigue syndrome_T2',
        'NCI_ankylosing spondylitis_T2',
        'NCI_trigemminal neuralgia_T2',
        #### Start 3 ####
        'NCI_crohns disease_T2',
        'NCI_spinal stenosis_T2',
        'NCI_peripheral neuropathy_T2',
        'NCI_ulcerative colitis_T2',
        'NCI_pulmonary embolism +/- dvt_T2',
        'NCI_chronic obstructive airways disease/copd_T2',
        'NCI_stroke_T2',
        'NCI_multiple sclerosis_T2',
        'NCI_psoriatic arthropathy_T2',
        'NCI_Post Surgical Pain_T2',
        'NCI_parkinsons disease_T2',
        'NCI_disc disorder_T2',
        'NCI_peripheral vascular disease_T2'
           ]

# 2. Path Settings
home_dir = '/lustre06/project/6055423/UK_Biobank/Files_CSV/'

# 3. Functions
def load_and_impute(csv_path, strategy="median"):
    data = pd.read_csv(csv_path)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputed_data = imp_mean.fit_transform(data)
    return pd.DataFrame(imputed_data, columns=data.columns)

# 4. Data Loading
confounds = pd.read_csv(home_dir + 'UKB_brain_confounds_T2.csv')
mri_t1 = load_and_impute(home_dir + 'UKB_Named_T1_42k.csv')
mri_dti = pd.read_csv(home_dir + 'UKB_Named_DTI_42k.csv').dropna()  # Same 5k subjects missing all DWI data
brain_fc = pd.read_csv(home_dir + 'DCC_full' + '.csv')  # FC method 
edges = pd.read_csv(home_dir + 'Fan_Cluster_279_Edges_Details_V2.csv')
bone = load_and_impute(home_dir + 'UKB_bone.csv')
psychosocial = pd.read_csv(home_dir + 'NonBio_T2.csv')  # With sleep
blood_data = load_and_impute(home_dir + 'UKB_New_Blood_Immuno_T0.csv')
PRS = pd.read_csv(home_dir + 'PRS_NCI_T2.csv')
UKB = pd.read_csv(home_dir + 'UKB_NoBrain_500K_V5.csv', usecols=['eid', 'Age_T2', 'Sex_T0', 'UKBiobankSite_T2'])
NCI = pd.read_csv(home_dir + 'NCI_combined.csv')
brain_t3 = pd.read_csv(home_dir + 'UKB_Stacked_Brain_T3.csv')

# 5. Data Processing
# Merging & Transformations
mri = mri_dti.merge(mri_t1, on='eid', how='outer')
brain = mri.merge(brain_fc, on='eid', how='outer')
brain = clean_data(brain) # Remove confound outliers

brain_trunc = brain_fc.rename(columns=dict(zip(edges.names, edges.long_names)))
brain_trunc = brain_trunc.groupby(brain_trunc.columns, axis=1).mean()
brain = brain.merge(brain_trunc, on='eid', how='outer')

brain = brain.merge(bone, on='eid', how='outer')
brain = brain.merge(psychosocial, on='eid', how='outer')

blood_data[blood_data.columns[1:]] = (blood_data[blood_data.columns[1:]] + 1).apply(np.log)
blood_data = blood_data[blood_data.eid.isin(brain.eid)]
brain = brain.merge(blood_data, on='eid', how='outer')

brain = brain.merge(PRS, on='eid', how='outer').sort_values('eid').reset_index(drop=True)

UKB[['site_1', 'site_2', 'site_3', 'site_4']] = pd.get_dummies(UKB.UKBiobankSite_T2)
NCI = pd.concat([NCI.filter(like='T2'), NCI.eid], axis=1)
UKB = UKB.merge(confounds, on='eid', how='outer')
UKB = UKB.merge(NCI, on='eid', how='outer')
UKB = UKB.dropna(subset='UKBiobankSite_T2')
brain_t3.columns = [i.replace('T3', 'T2') for i in brain_t3.columns]

# Log-normalize motion
UKB['fMRIMotion_T2'] = np.log(UKB['fMRIMotion_T2'])
    
# 6. Modalities
T1_cols = mri.columns[614:].to_list()
dwi_cols = mri.columns[1:614].to_list()
fc_cols = brain_fc.columns[:-1].to_list()
trunc_fc_cols = brain_trunc.columns[:-1].to_list()
stacked_brain = T1_cols + dwi_cols + trunc_fc_cols
bone_cols = bone.columns[1:].to_list()
prs_cols = PRS.columns[1:].to_list()
blood_cols = blood_data.columns[1:].to_list()
psychosocial_cols = psychosocial.columns[1:].to_list()

modalities = [T1_cols, dwi_cols, fc_cols, stacked_brain, bone_cols, psychosocial_cols]
mod_names = ['T1', 'DWI', 'fMRI', 'Stacked', 'Bone', 'Psychosocial']

# 7. Final Execution
clean_brain = brain.merge(UKB[confound_cols + NCI_cols], on='eid', how='outer').dropna(subset='NCI_T2')
num_iter = int(sys.argv[1]) # Number of model iterations to run

print('Data prepped...')
print('Final data shape: ', clean_brain.shape)

run_model(data=clean_brain, clf='LR_lin', outcome='diagnoses', modalities=modalities,
          mod_names=mod_names, iterations=num_iter, lng_brain=brain_t3)