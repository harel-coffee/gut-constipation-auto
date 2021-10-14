import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, SelectFromModel, chi2, f_classif, mutual_info_classif
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

def features_select(Xtrain, ytrain, Xtest, method):
    if method in ['T-test', 'Mann-Whitney', 'Wilcox-test']:
        disease_group = Xtrain.loc[ytrain == 0,]
        norm_group = Xtrain.loc[ytrain == 1,]
        features_index = []

        for i in range(Xtrain.shape[1]):
            if method == 'T-test':
                fs = stats.ttest_ind(disease_group.iloc[:,i], norm_group.iloc[:,i])
            elif method == 'Mann-Whitney':
                if len(set(disease_group.iloc[:,i])) & len(set(norm_group.iloc[:,i])) != 1:
                    fs = stats.mannwhitneyu(disease_group.iloc[:,i], norm_group.iloc[:,i])
                else:
                    continue
            elif method == 'Wilcox-test':
                fs = stats.ranksums(disease_group.iloc[:,i], norm_group.iloc[:,i])

            if fs.pvalue <= 0.05:
                features_index.append(i)
    
    elif method == 'Origin':
        return Xtrain, Xtest
    
    else:
        if method in ['Chi2', 'F-test', 'Mutual information']:
            if method == 'Chi2':
                sel = SelectPercentile(chi2, percentile=10)
            elif method == 'F-test':
                sel = SelectPercentile(f_classif)
            elif method == 'Mutual information':
                sel = SelectPercentile(mutual_info_classif, percentile=10)
            features_new = sel.fit_transform(Xtrain, ytrain)
            
        elif method in ['Logistics', 'LASSO', 'Random Forest']:
            if method == 'Logistics':
                clf = LogisticRegression(random_state=625)
            elif method == 'LASSO':
                clf = LogisticRegression(random_state=625, solver='saga', penalty='l1')
            elif method == 'Random Forest':
                clf = RandomForestClassifier(random_state=625)
            clf = clf.fit(Xtrain, ytrain)
            sel = SelectFromModel(clf, prefit=True)
            features_new = sel.transform(Xtrain)
        
        features_index = list(np.where(sel.get_support() == True)[0])

    train_fs = Xtrain.iloc[:,features_index]
    valid_fs = Xtest.iloc[:,features_index]
    return train_fs, valid_fs

def data_prec(X, y, cv_index, selector):
    scaler = StandardScaler()
    Xtrain = X.iloc[cv_index[0], :]
    Xtest = X.iloc[cv_index[1], :]
    ytrain = y[cv_index[0]]
    ytest = y[cv_index[1]]
    Xtrain_selected, Xtest_selected = features_select(Xtrain, ytrain, Xtest, selector)

    Xtrain_selected = scaler.fit_transform(Xtrain_selected)
    Xtest_selected = scaler.transform(Xtest_selected)
    return Xtrain_selected, ytrain, Xtest_selected, ytest

def model_report(X, y, model, selector, cv_index):
    prec_data = data_prec(X, y, cv_index, selector)
    train_auc = cross_val_score(model, prec_data[0], prec_data[1], cv=RepeatedStratifiedKFold(n_repeats=8, n_splits=5), n_jobs=-1, scoring='roc_auc').mean()
    
    model.fit(prec_data[0], prec_data[1])
    pred_test = model.predict(prec_data[2])
    test_acc = accuracy_score(prec_data[-1], pred_test)
    return train_auc, test_acc

# data load
data = pd.read_csv('Data/otu_table_L6_constipation.txt', index_col=0)

# data pre-process
genus = data.iloc[:, :-1]
target = LabelEncoder().fit_transform(data.iloc[:, -1])

# base classifiers
clf_names = ['Log', 'Lasso', 'NB', 'kNN', 'SVM', 'DT', 'AdA', 'RF', 'GBRT']
classifiers = [LogisticRegression(), LogisticRegression(solver = "saga",penalty = "l1"), GaussianNB(), KNeighborsClassifier(), 
               SVC(probability=True), DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()]

# feature selection methods
selectors = ['Origin', 'T-test', 'Mann-Whitney', 'Wilcox-test', 'Chi2', 'F-test', 'Mutual information', 'Logistics', 'LASSO', 'Random Forest']

# main
train_aucs, test_accs = [], []
cv_indices = [[a,b] for a,b in RepeatedStratifiedKFold(n_repeats=1, n_splits=5).split(genus, target)]

for clf in classifiers:
    clf_train_aucs = []
    clf_test_accs = []
    for selector in selectors:
        selector_train_aucs = []
        selector_test_accs = []
        for cv_index in cv_indices:
            aucs = model_report(genus, target, clf, selector, cv_index)
            selector_train_aucs.append(aucs[0])
            selector_test_accs.append(aucs[1])
        clf_train_aucs.append(np.mean(selector_train_aucs))
        clf_test_accs.append(np.mean(selector_test_accs))
    train_aucs.append(clf_train_aucs)
    test_accs.append(clf_test_accs)

train_aucs_df = pd.DataFrame(train_aucs, index=clf_names, columns=selectors).T
test_accs_df = pd.DataFrame(test_accs, index=clf_names, columns=selectors).T

# plot
sns.heatmap(train_aucs_df, cmap='RdBu_r', annot=True)
sns.heatmap(test_accs_df, cmap='RdBu_r', annot=True)