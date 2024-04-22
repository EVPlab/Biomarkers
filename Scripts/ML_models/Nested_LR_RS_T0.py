"""
# UK Biobank Data Setup and Logistic Regression Classification

This script sets up the UK Biobank dataset containing participants with data from blood immunoassays, polygenic risk scores, and psychosocial assessments. 
It instantiates a logistic regression machine learning model with nested cross-validation and randomized hyperparameter search.

## Overview

The script aims to classify participants reporting a pain-associated diagnosis from diagnosis-free controls based on biological and psychosocial features.

## Features Included

- Blood immunoassays
- Polygenic risk scores
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, matthews_corrcoef,average_precision_score
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

def p_match(data,NCI,modality):
    
    # Get top 5 medications taken by NCI group + statins
    data = data.reset_index(drop=True)
    
    if modality in ['Blood', 'Inflammatory', 'Metabolic', 'CBC']:
        top_meds = data[(data[NCI] == 1)][med_cols].sum().sort_values().tail(5).index.tolist() + ['Med_C10A_T0']
        top_meds = list(set(top_meds))
        c = top_meds+['Sex_T0']
        c2 = c+['Age_T0']
        n_comp = 5
        fc = ['FC'+str(i) for i in range(1,n_comp+1)]
        confounds = fc
    else:
        c = ['Sex_T0']
        c2 = c+['Age_T0']
        n_comp = 2
        fc = ['FC'+str(i) for i in range(1,n_comp+1)]
        confounds = fc
    
    # Compute top n components by factor analysis
    famd = prince.FAMD(
    n_components=n_comp,
    n_iter=10,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
)

    # Preparing data structures for FAMD
    data[c] = np.where(data[c] == 0,'False','True') # Recode binaries to fit with prince API
    famd_df = data[c2]
    famd = famd.fit(famd_df)
    fc_comps = famd.transform(famd_df)
    data[fc] = fc_comps
    
    # Propensity match on components of top 10 meds, sex and Age
    match_vars = ['eid',
              NCI
             ] + confounds
    confounds_ = ['eid',
                 NCI
                ] + confounds
    
    pymatch = data[match_vars].dropna()

    pscore = pymatch[confounds_]
    case = pscore[pscore[NCI] == 1]
    control = pscore[pscore[NCI] == 0]
    m = Matcher(case, control, yvar=NCI, exclude=['eid'])

    np.random.seed(20170925)
    
    m.fit_scores(balance=True, nmodels=100)

    m.predict_scores()

    m.tune_threshold(method='random')
    m.match(method="min", nmatches=1, threshold=0.2, with_replacement=False)
    m.record_frequency()
    m.assign_weight_vector()
    matched = m.matched_data.sort_values("match_id")
    
    # Create matched df with variables of interest
    clean_matched = data[data.eid.isin(matched.eid)]
    clean_matched[c] = np.where(clean_matched[c] == 'True',1,0) # Recode binary strings to int
    
    return clean_matched

##################################################################################################################
    
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef, average_precision_score
import numpy as np
import pandas as pd

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

    if mod_name in ['PRS', 'Blood', 'Psychosocial']:
        roc_curve = np.stack([eid_test, y_test, prob_test])
        roc_curve_other = np.stack([eid_o, y_o, prob_o_test])
    else:
        roc_curve = [np.NaN]
        roc_curve_other = [np.NaN]
        
    results_dic = {
        'weights': feature_imp,
        'results_df': norm_df,
        'roc_curve': roc_curve,
        'roc_curve_other': roc_curve_other,
        'random_state': rs,
        'best_params': model.best_params_,
        'signature': signature,
        'signature_other': signature_o
    }

    # Additional computations for 'Blood' or 'Psychosocial' models
    if mod_name in ['Blood', 'Psychosocial']:
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
                                'clf__regularizer': np.logspace(-3,5,100),
        }
                
        # Flat params
        if clf == 'LR_rbf':
            clf = SnapLR(max_iter=10000,kernel='rbf',normalize=True,class_weight='balanced',n_jobs=1,use_gpu=True)
            params_full = {
                                'clf__regularizer': np.logspace(-2,6,75),
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
            clf = SnapSVM(max_iter=100000,normalize=False,class_weight='balanced',n_jobs=1,use_gpu=True)
            params_full = {
                            'clf__loss' : ['hinge', 'squared_hinge'],
                            'clf__regularizer': np.logspace(-2,6,100),
    }

        # Full model
        scale = StandardScaler()
        pipe = Pipeline(steps=[
        ('scaler',scale),
        ('clf',clf)])
    
        return pipe, params_full
    
##################################################################################################################


def train_model(targ,clf,data,var,mod_name,modality,meta_dic,iterations,iter_l,lng_blood,lng_psych):
    
    # Get classifier and parameters for gridsearch
    pipe,params = get_clf(clf)

    # Instantiate gridsearch estimator
    model = RandomizedSearchCV(estimator=pipe,
                                    n_iter=30,
                                    param_distributions=params,
                                    # scoring=make_scorer(matthews_corrcoef),
                                    scoring=make_scorer(roc_auc_score,needs_proba=True),
                                    cv=5,
                                    return_train_score=False,
                                    n_jobs=15,
                                    pre_dispatch=30,
                                    verbose=0,
                                    refit=True,
                                    random_state=173
                                    )
        
    # Remove modality specific null values (alternative to the above loop)
    y_var = var
    model_vars = ['eid']+modality+[y_var]+confound_cols
    clean_eid = data[model_vars].dropna()['eid']
    data_prep = data[data.eid.isin(clean_eid)]
    
    # cases = Dx and controls = Dx free, PRS using T2 NCI variables
    data_tmp = data_prep.dropna(subset='NCI_T0')
    data_tmp = pd.concat([data_tmp[data_tmp['NCI_T0'] == 0],data_tmp[data_tmp[y_var] == targ]])
        
    # Propensity matching each NCI based on confounds for blood
    if mod_name in ['Blood','Inflammatory','Metabolic','CBC']:
        data_tmp = p_match(data_tmp,y_var,mod_name)        
    else:
        data_tmp = data_tmp
        
    # Predict subjects removed for CV
    data_other = data_prep[~data_prep.eid.isin(data_tmp.eid)]

    # Create feature, target, and confound matrices
    X = data_tmp[modality]
    y = data_tmp[y_var]
    C = data_tmp[confound_cols]
    eid = data_tmp['eid']

    # Other vars
    X_o = data_other[modality]
    y_o = data_other[y_var]
    C_o = data_other[confound_cols]
    eid_o = data_other['eid']
    
    # Longitudinal
    if mod_name in ['Blood', 'Inflammatory', 'Metabolic', 'CBC']:
        X_l = lng_blood[[i.replace('T0','T1') for i in modality]]
        C_l = lng_blood[['Age_T1','Sex_T0']]
        eid_l = lng_blood['eid']
    elif mod_name == 'Psychosocial':
        X_l = lng_psych.drop(columns=['eid','Age_T1','Sex_T0'])
        C_l = lng_psych[['Age_T1','Sex_T0']]
        eid_l = lng_psych['eid']
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
            if mod_name in ['Blood', 'Inflammatory', 'Metabolic', 'CBC', 'Psychosocial']:
                X_train = resid.transform(X_train, C_train)
                X_test = resid.transform(X_test, C_test)
                X_o_test = resid.transform(X_o, C_o)
                X_l_test = resid.transform(X_l, C_l)
            else:
                X_train = X_train.values
                X_test = X_test.values
                X_o_test = X_o.values

            # Save Dx name for file name
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

def run_model(data,clf,mod_names,modalities,outcome,iterations,lng_blood,lng_psych):
    
    # Dictionary for appending results
    meta_dic = {}
    iter_l = []

    # Loop through set number of iterations and train classifier on pain group of interest
    for mod_ind,modality in enumerate(modalities):

            if outcome == 'diagnoses':

                    pain_NCI = [
                'NCI_fibromyalgia_T0',
                'NCI_polymyalgia rheumatica_T0',
                'NCI_cervical spondylosis_T0',
                'NCI_joint pain_T0',
                'NCI_back pain_T0',
                'NCI_spine arthritis/spondylitis_T0',
                'NCI_trapped nerve/compressed nerve_T0',
                'NCI_sciatica_T0',
                'NCI_hernia_T0',
                'NCI_irritable bowel syndrome_T0',
                'NCI_gastro-oesophageal reflux (gord) / gastric reflux_T0',
                'NCI_arthritis (nos)_T0',
                
                'NCI_osteoarthritis_T0',
                'NCI_osteoporosis_T0',
                'NCI_rheumatoid arthritis_T0',
                'NCI_migraine_T0',
                'NCI_headaches (not migraine)_T0',
                'NCI_carpal tunnel syndrome_T0',
                'NCI_angina_T0',
                'NCI_endometriosis_T0',
                'NCI_gout_T0',
                'NCI_chronic fatigue syndrome_T0',
                'NCI_ankylosing spondylitis_T0',
                'NCI_trigemminal neuralgia_T0',
                'NCI_crohns disease_T0',
                
                'NCI_spinal stenosis_T0',
                'NCI_peripheral neuropathy_T0',
                'NCI_ulcerative colitis_T0',
                'NCI_pulmonary embolism +/- dvt_T0',
                'NCI_chronic obstructive airways disease/copd_T0',
                'NCI_stroke_T0',
                'NCI_multiple sclerosis_T0',
                'NCI_psoriatic arthropathy_T0',
                'NCI_Post Surgical Pain_T0',
                'NCI_parkinsons disease_T0',
                'NCI_disc disorder_T0',
                'NCI_peripheral vascular disease_T0'
    ]
                    targ = 1
                    for var in pain_NCI:
                    
                        if mod_names[mod_ind] == 'PRS':
                            var_str = var[:-3]
                            modality_new = PRS.filter(like=var_str).columns[:-1].to_list()
                        else:
                            modality_new = modality
                            
                    ####### Train model and get results #######
                        meta_dic, iter_l = train_model(targ=targ,clf=clf,data=data,var=var,mod_name=mod_names[mod_ind],
                                        modality=modality_new,iterations=iterations,iter_l=iter_l,meta_dic=meta_dic,
                                        lng_blood=lng_blood,lng_psych=lng_psych)

            iter_df = pd.concat(iter_l)
    
        # Save results dictionary
    path_out = '/lustre06/project/6055423/UK_Biobank/Slurm/mfill/UKB_CP_Classification/Phenotypes/PickledModels/'
    with open(path_out+f'T0_{outcome}_clf-{clf}_Pt2.pickle', 'wb') as file:
        pickle.dump(meta_dic, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return iter_df, meta_dic

################################################################ Imputation and log transform of blood features ###############################################################################

def transform_blood_data(blood, log=True):
    """
    Impute missing values with the median and apply log transformation to all columns except the first one,
    using log1p if zeros are present in the feature.
    
    Parameters:
    - blood (pd.DataFrame): Input DataFrame with blood data.
    
    Returns:
    - pd.DataFrame: Transformed blood data.
    """
    
    # Impute missing values with median
    imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
    blood_data = pd.DataFrame(imp_med.fit_transform(blood), columns=blood.columns)
    
    # Apply log transformation as required
    if log:
        for column in blood_data.columns[1:]:  # Exclude the first column (subject ID)
            if (blood_data[column] == 0).any():  # Check if there are zero values in the column
                blood_data[column] = np.log1p(blood_data[column])
            else:
                # Avoid taking log of zero or negative values by adding 1
                blood_data[column] = np.log(blood_data[column])
    
    return blood_data

################################################################ Run gridsearch classification and save outputs ###############################################################################

home_dir = '/lustre06/project/6055423/UK_Biobank/Files_CSV/'

confound_cols = [
            'Age_T0', 
            'Sex_T0',
                ]

UKB = pd.read_csv(home_dir + 'UKB_NoBrain_500K_V5.csv',low_memory=False, usecols=['eid','Age_T0','Age_T1','Sex_T0']) #UKB demographic data

# Load medications to regress out from blood
med = pd.read_csv(home_dir + '0_UKB_ATC_Medications.csv').iloc[:,1:]
med_cols = med.filter(like='T0').columns.tolist()
med[med[med_cols] > 1] = 1
UKB = UKB.merge(med,on='eid')

# Load Blood Data
# T0
blood = pd.read_csv(home_dir + 'UKB_New_Blood_Immuno_T0.csv')
threshold = len(blood.columns) / 2 
blood = blood.dropna(thresh=threshold) # Drop subjects missing >= 50% of features
blood_data = transform_blood_data(blood)
# T1
blood_t1 = pd.read_csv(home_dir + 'UKB_New_Blood_Immuno_T1.csv').dropna(thresh=threshold) # Drop subjects missing >= 50% of features
blood_t1 = transform_blood_data(blood_t1)

# Align column names
column_names = [col.replace('T0', 'T1') for col in blood_data.columns]
blood_t1 = blood_t1[column_names]
blood_t1 = blood_t1.merge(UKB[['eid','Age_T1','Sex_T0']],on='eid')

# Additional variables
merge_cols = ['eid',
'Age_T0',
'Sex_T0'
] + med_cols
blood_data = blood_data.merge(UKB[merge_cols],on='eid',how='outer')

# NCI Data
NCI = pd.read_csv(home_dir + 'NCI_combined.csv')
NCI = pd.concat([NCI.filter(like='T0'),NCI.filter(like='T2'),NCI.eid],axis=1)
nci_cols = NCI.filter(like='NCI').columns

# Merge NCI variables
blood_data = blood_data.merge(NCI,on='eid',how='outer')

# Risk Score with biology removed
psychosocial = pd.read_csv(home_dir + 'NonBio_T0.csv') # With sleep
# T1 Risk scores
psychosocial_t1 = pd.read_csv(home_dir + 'NonBio_T1.csv') # With sleep
missing = ['METWalking_T0',
 'METVigorousActivity_T0',
 'METModerateActivity_T0',
 'SummedDaysActivity_T0',
 'HighIPAQActivityGroup_T0',
 'LowIPAQActivityGroup_T0',
 'AboveModerateVigorousWalkingRecommendation_T0']
means = psychosocial[missing].mean()
t1_cols = [i.replace('T0','T1') for i in missing]
psychosocial_t1.loc[:, t1_cols] = means.values
psychosocial_t1 = transform_blood_data(psychosocial_t1,log=False)
# Align column names
column_names = [col.replace('T0', 'T1') for col in psychosocial.columns]
psychosocial_t1 = psychosocial_t1[column_names]
psychosocial_t1 = psychosocial_t1.merge(UKB[['eid','Age_T1','Sex_T0']],on='eid')

# Polygenic Risk Score
PRS = pd.read_csv(home_dir + 'PRS_NCI_T2.csv')

########################################################### Modality columns #########################################################################

prs_cols = PRS.columns[1:].to_list()
blood_cols = blood.columns.drop(['Sodium_T0', 'Potassium_T0'])[1:].to_list() # Drop urine assays

inf = [
    'CReactiveProtein_T0', 
    'EosinophillCount_T0', 
    'BasophileCount_T0', 
    'MonocyteCount_T0', 
    'NeutrophillCount_T0', 
    'WhiteBloodCellLeukocyteCount_T0', 
    'BasophilePercentage_T0',
    'EosinophillPercentage_T0',
    'MonocytePercentage_T0',
    'NeutrophillPercentage_T0',
    'LymphocyteCount_T0',
    'LymphocytePercentage_T0'
]
met = [
    'Glucose_T0', 
    'AlanineAminotransferase_T0', 
    'AspartateAminotransferase_T0', 
    'AlkalinePhsophate_T0', 
    'TotalBilirubin_T0', 
    'Urea_T0', 
    'Creatinine_T0',
    'Albumin_T0',
    'Calcium_T0', 
    'TotalProtein_T0', 
    # 'Sodium_T0', # Urine
    # 'Potassium_T0', # Urine
    'AlipoproteinA_T0', 
    'AlipoproteinB_T0', 
    'GammaGlutamyltransferase_T0', 
    'Triglyceride_T0', 
    'VitaminD_T0', 
    'Testosterone_T0', 
    'IGF1_T0', 
    'LDLDirect_T0', 
    'Urate_T0', 
    'HDLCholesterol_T0', 
    'CystatinC_T0', 
    'GlycatedHaemoglobin_T0'
]
cbc = [
    'HaematocritPercentage_T0', 
    'HaemoglobinConcentration_T0', 
    'PlateletDistributionWidth_T0', 
    'MeanPlateletThrombocyteVolume_T0', 
    'PlateletCrit_T0', 
    'PlateletCount_T0',
    'RedBloodCellErythrocyteCount_T0', 
    'RedBloodCellErythrocyteDistributionWidth_T0', 
    'HighLightScatterReticulocyteCount_T0', 
    'HighLightScatterReticulocytePercentage_T0', 
    'ReticulocyteCount_T0', 
    'ReticulocytePercentage_T0', 
    'ImmatureReticulocyteFraction_T0', 
    'MeanReticulocyteVolume_T0', 
    'MeanSpheredCellVolume_T0', 
    'NucleatedRedBloodCellCount_T0', 
    'NucleatedRedBloodCellPercentage_T0'
]
psychosocial_cols = psychosocial.columns[1:].to_list()

modalities = [blood_cols,inf,met,cbc,prs_cols,psychosocial_cols]
mod_names = ['Blood','Inflammatory','Metabolic','CBC','PRS','Psychosocial']

########################################################### Finalize data structuring #########################################################################

num_iter = int(sys.argv[1])

# Merge features into one dataframe
model_df = pd.merge(blood_data,PRS,on='eid',how='outer').sort_values('eid').reset_index(drop=True)
model_df = model_df.merge(psychosocial,on='eid',how='outer').sort_values('eid').reset_index(drop=True)

print('Data prepped...')
print('Final data shape: ',model_df.shape)

########################################################### Run gridsearch classification and save outputs #########################################################################

run_model(data=model_df,clf='LR_lin',outcome='diagnoses',modalities=modalities,
          mod_names=mod_names,iterations=num_iter,lng_blood=blood_t1,lng_psych=psychosocial_t1)