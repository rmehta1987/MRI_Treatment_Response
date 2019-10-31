import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import glob
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
#from visdom import Visdom
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from itertools import compress
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn import feature_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
import seaborn as sns
import sys
from sklearn.utils import resample
from sklearn.impute import SimpleImputer 



#Global Variables

#Classifier Names
names = ["Linear_SVM", "RBF_SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "AdaBoost",
         "Naive_Bayes", "Gradient_Boosted"]

#Pretty classifier Names
c_names = ["Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes", "Gradient Boosted"]


    
#Feature Names, a10 represents 10th percentile and so on, note that 50th percentile is equivalent to median of histogram
fnames = ["a10", "a25", "a50", "a75", "aquartile", "amean", "avar", "akurt", 
"askew",  "b10", "b25", "b50", "b75", "bquartile", "bmean", "bvr", "bkurt", 
"bskew", "d10", "d25", "d50", "d75", "dquartile","dmean",  "dvra", "dkurt", "dskew"]         

#Names with CTRW and IVIM
f2names = ["a10", "a25", "a50", "a75", "aquartile", "amean", "avar", "akurt", 
"askew",  "b10", "b25", "b50", "b75", "bquartile", "bmean", "bvar", "bkurt", 
"bskew", "d10", "d25", "d50", "d75", "dquartile","dmean", "dvar", "dkurt", "dskew",
"diff10", "diff25", "diff50", "diff75", "diffquartile", "diffmean", "diffvar", "diffkurt", "diffskew",
"perf10", "perf25", "perf50", "perf75", "perfquartile", "perfmean", "perfvar", "perfkurt", "perfskew",
"f10", "f25", "f50", "f75", "fquartile", "fmean", "fvar", "fkurt", "fskew"]       

print (len(f2names))

#This function gets gets all the maps and loads them into a list, can be modified if more maps are added later
def getFiles(file_path, name):
    '''@file_path: where files are stored
        @name: name of the individual maps, maps should be .npy format and file names are mapname*.npy'''

    afiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[0])))
    bfiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[1])))
    dfiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[2])))
    difffiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[3])))
    perffiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[4])))
    ffiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[5])))
    lafiles = []
    lbfiles = []
    ldfiles = []
    ldifffiles = []
    lperffiles = []
    lffiles = []
    print ("obtaining files in the getfiles function")
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles,bfiles,dfiles,difffiles,perffiles,ffiles)):
        lafiles.append(np.load(a))
        lbfiles.append(np.load(b))
        ldfiles.append(np.load(d))
        ldifffiles.append(np.load(e))
        lperffiles.append(np.load(f))
        lffiles.append(np.load(g))
    
    return lafiles, lbfiles, ldfiles, ldifffiles, lperffiles, lffiles


#IVIM FEATURE MATRIX, creates a feature matrix of all the maps, more can be added if necessary
def createFeatMat3(afiles, bfiles, dfiles, diff_files, perf_files, f_files):
    '''creates a feature matrix from the different set of files
    @param afiles = features of amaps  - arranged as the map (0), label (1), patient name (2), histogram features (3), histogram in a numpy array (4)
    @param bfiles = features of bmaps
    @param dfiles = features of dmaps'''

    xtrain = np.zeros((len(afiles),len(afiles[0][3])*6)) #Feature Matrix [a-features, b-features, ddc-features, diff-features, perf-features, f-features]
    ytrain = np.zeros((len(afiles),),dtype=np.int) #label matrix
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles,bfiles,dfiles,diff_files, perf_files, f_files)):
        xtrain[i] = np.hstack((a[3], b[3], d[3], e[3], f[3], g[3]))
        ytrain[i] = a[1]

    print ("finished creating Feature and Label Matrix createFeatMat3, such that original and IVIM features are together")
    return xtrain, ytrain


#Starts visdom, the visualizer for graphs
#viz = Visdom()

#Initialize Classifier hyperparameters
classifiers = [
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(max_depth=5, n_estimators=10, max_features=1)]


classifiers2 = [
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(max_depth=5, n_estimators=10, max_features=1)]

#Repeated Cross Validation
rkf = RepeatedKFold(n_splits=3, n_repeats=2)

#Get all the Files
print ("Getting Files containing features of original maps")
file_path = 'maxminFeat'
name = ['apad_feat','bpad_feat','dpad_feat', 'diffpad_feat', 'perfpad_feat', 'fpad_feat']
afiles, bfiles, dfiles, diff_files, perf_files, f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Crop 1")
name = ['cropaug1_alpha_feat','cropaug1_beta_feat','cropaug1_ddc_feat', 'cropaug1_diff_feat', 'cropaug1_perf_feat', 'cropaug1_f_feat']
c1afiles, c1bfiles, c1dfiles, c1diff_files, c1perf_files, c1f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Crop 2")
name = ['cropaug2_alpha_feat','cropaug2_beta_feat','cropaug2_ddc_feat', 'cropaug2_diff_feat', 'cropaug2_perf_feat', 'cropaug2_f_feat']
c2afiles, c2bfiles, c2dfiles, c2diff_files, c2perf_files, c2f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Crop 3")
name = ['cropaug3_alpha_feat','cropaug3_beta_feat','cropaug3_ddc_feat', 'cropaug3_diff_feat', 'cropaug3_perf_feat', 'cropaug3_f_feat']
c3afiles, c3bfiles, c3dfiles, c3diff_files, c3perf_files, c3f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Original Augmented Rotated")
name = ['ogaug_alpha_feat','ogaug_beta_feat','ogaug_ddc_feat', 'ogaug_diff_feat', 'ogaug_perf_feat', 'ogaug_f_feat']
augafiles, augbfiles, augdfiles, augdiff_files, augperf_files, augf_files = getFiles(file_path,name)


def runClassifiers(xtrain, ytrain):
    '''@xtrain feature matrix
        @ytrain label matrix'''

    #Importance of features obtained from RFE Linear SVM, RBF_SVM, and Random Forest -- 
    importances = [None] * 3
    for i in range(0,3):
        importances[i] = list()
    
    all_models = [0] * len(classifiers2)
    min = [0] * len(classifiers2)
    #standard deviation of importance obtained from random forest
    std = list()

    #scores of accuracy and f1 for each classifier
    accscore = np.empty((len(classifiers),0)).tolist()
    fonescore = np.empty((len(classifiers),0)).tolist()
    tprs = np.empty((len(classifiers),0)).tolist()
    aucs = np.empty((len(classifiers),0)).tolist()
    
    mean_fpr = np.linspace(0,1,100)
    for train, test in rkf.split(xtrain):

        txtrain = xtrain[train]
        tytrain = ytrain[train]
        txtest = xtrain[test]
        tytest = ytrain[test]
    
        # Temporalrily until fixed NAN values:
        # imr = SimpleImputer(missing_values=np.nan, strategy='median')
        # imr = imr.fit(txtrain)
        # txtrain = imr.transform(txtrain)
        # imr = imr.fit(txtest)
        # txtest = imr.transform(txtest)
        print ("One Iteration of CV")
        # iterate over classifiers
        for i, (name, clf) in enumerate(zip(names, classifiers2)):
            
            '''
            #creating training matrix 
            if name == "Linear_SVM":
                #estimator = RFE(clf, 10, step=1)
                estimator = clf
                estimator = estimator.fit(txtrain, tytrain.ravel())
                ypred = estimator.predict(txtest)
                probROC = estimator.predict_proba(txtest)
                #importances[0].append(estimator.get_support(indices=True))
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
                
            elif name == "RBF_SVM":
                #rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
                #ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
                #probROC = clf.predict_proba(txtest[:,estimator.get_support(indices=True)])
                estimator = clf
                estimator = estimator.fit(txtrain, tytrain.ravel())
                ypred = estimator.predict(txtest)
                probROC = estimator.predict_proba(txtest)
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
            elif name == 'Random_Forest':
                probsz = clf.fit(txtrain, tytrain.ravel())
                timportance = clf.feature_importances_
                importances[2].append(timportance)
                std.append(np.std([tree.feature_importances_ for tree in clf.estimators_],
                axis=0))
                ypred = clf.predict(txtest)
                probROC = clf.predict_proba(txtest)
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
            else:
                probsz = clf.fit(txtrain, tytrain.ravel())
                ypred = clf.predict(txtest)
                probROC = clf.predict_proba(txtest)
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
                
            '''
            probsz = clf.fit(txtrain, tytrain.ravel())
            ypred = clf.predict(txtest)
            probROC = clf.predict_proba(txtest)
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
            fpr, tpr, thresholds = roc_curve(tytest.ravel(),probROC[:,1])
            tprs[i].append(interp(mean_fpr,fpr,tpr))
            tprs[i][-1][0]=0.0
            aucs[i].append(auc(fpr,tpr))
            accscore[i].append(theaccscore)
            if accscore[i][-1] > min[i]:
                all_models[i] = clf
                min[i] = theaccscore
            fonescore[i].append(thef1score)
        
    return accscore, fonescore, std, importances, tprs, aucs, all_models

def visualizeResults(accscore, fonescore, std, importances, allnames, thetitle):
    '''@accscore is the accuracy score of the individual classifiers
        @fonescore is the F1 score of the individual classifiers
        @std standrad deviation of feature importances **CURRENTLY NOT IMPLEMENTED**
        @importances importances of classifiers that can return feature importances, only index 0 and 2 are currently implemetned
        where importances[0] is SVM RFE Linear and importances[2] is Random Forest Importance'''

    #find average of feature importances from random forest
    avgimport = np.average(np.array(importances[2]),0)
    indices = np.argsort(avgimport)[::-1]
    avgstd = np.average(np.array(std),0)


    print ("RFE Importance ", end = " ")
    print ([allnames[i] for i in importances[0][-1]])


    print ("Random Forest Importance", end = " ")
    print ([(allnames[i],avgimport[i]) for i in indices])

    print ("F1 Scores")

    r = len(fonescore) #number of classifiers
    c = len(fonescore[0])#number of folds 
    X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
    Y = np.ones((c,r))
    Y = Y*np.array(fonescore).transpose()
    newtitle = 'F1 Score ' + thetitle
    f1win = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='F-1 score',title=newtitle,legend=names))

    for i,scores in enumerate(fonescore):
        #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=orgf1score,opts=dict(xlabel='Fold',ylabel='F1',title='{} Original F1 score'.format(names[i])))
        #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=orgf1score,opts=dict(xlabel='Fold',ylabel='F1'),update='append')
        print ("Classifier name and F1 Score: %s : "%(names[i]), end = " ")
        print (scores) 

    print ("Accuracy Scores")

    for i,scores in enumerate(accscore):
        #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=accwin, opts=dict(xlabel='Fold',ylabel='F1',title='{} Original Accuracy score'.format(names[i])))
        #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=accwin, opts=dict(xlabel='Fold',ylabel='Accuracy'),update='append')
        #viz.line(X=np.arange(1,len(scores)+1).reshape(-1,1),Y=np.array(scores).reshape(-1,1),name=names[i],win=accwin, opts=dict(xlabel='Fold',ylabel='Accuracy',legend=[names[i]]), update='append')
        print ("Classifier name and Accuracy Score: %s : "%(names[i]), end=" ")
        print (scores) 
    
    #Stack Visualizations of each accuracy
    r = len(accscore) #number of classifiers
    c = len(accscore[0])#number of folds 
    X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
    Y = np.ones((c,r))
    Y = Y*np.array(accscore).transpose()
    newtitle = 'Accuracy ' + thetitle
    accwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title=newtitle,legend=names))   

def visualizeResults2(accscore, fonescore, tprs, aucs, allnames, thetitle):
    '''@accscore is the accuracy score of the individual classifiers
        @fonescore is the F1 score of the individual classifiers
        @std standrad deviation of feature importances **CURRENTLY NOT IMPLEMENTED**
        @importances importances of classifiers that can return feature importances, only index 0 and 2 are currently implemetned
        where importances[0] is SVM RFE Linear and importances[2] is Random Forest Importance
        @allnames is feature names 
        @thetile is the title of the plots
    '''
        

    print ("Calculating F1")

    # Average F1 Score for every classifier
    #avg_f1 = [np.mean(scores) for scores in fonescore]
    #avg_acc = [np.mean(scores) for scores in fonescore]
    #avg_auc = [np.mean(scores) for scores in fonescore]
    #print (sns.utils.ci(aucs[-1],which=95))
    #print (sns.utils.ci(fonescore[-1],which=95))
    #print (sns.utils.ci(accscore[-1],which=95))
    sns.set()
    f1 = plt.figure(1)
    ax = sns.boxplot(data=fonescore,palette="vlag",showfliers=False)
    sns.swarmplot(data=fonescore, color=".2", alpha=0.3)
    ax.set(xticklabels=c_names[2:])
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_title('F1 Score of all Classifiers in {}'.format(thetitle))
    print ("Accuracy")
    f2 = plt.figure(2)
    ax2 = sns.boxplot(data=accscore,palette="vlag",showfliers=False)
    sns.swarmplot(data=accscore, color=".2", alpha=0.3)
    ax2.set(xticklabels=c_names[2:])
    ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
    ax2.set_title('Accuracy of all Classifiers in {}'.format(thetitle))
    print ("AUC")
    f3 = plt.figure(3)
    ax3 = sns.boxplot(data=aucs,palette="vlag",showfliers=False)
    sns.swarmplot(data=aucs, color=".2", alpha=0.3)
    ax3.set(xticklabels=c_names[2:])
    ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
    ax3.set_title('Area Under Curve of all Classifiers in {}'.format(thetitle))

    f1.savefig("F1_Classifier_test2.svg" ,bbox_inches='tight')
    f2.savefig("Acc_Classifier_test2.svg",bbox_inches='tight')
    f3.savefig("AUC_Classifier_test2.svg",bbox_inches='tight')

    #f4 = plt.figure(4)
    #ax4 = sns.lineplot(data=tprs, hue="region", style="event", markers=True, dashes=False)
    
    #plt.show()

def runGridSearch(xtrain, ytrain, allnames, grid_classifier=GradientBoostingClassifier()):
    
    # ### Pipeline for GridSearch 


    # #****** GRID SEARCH FOR OPTIMAL PARAMATERS for Gradient Boost Classifier   ********
    print ("GRID SEARCH")

    tuned_parameters = {"loss": ["deviance"], "learning_rate": [0.01, .025, .05, .075, 0.1, 0.15, 0.2], 
                        "min_samples_split": np.linspace(0.1,0.5,8),"min_samples_leaf": np.linspace(0.1,0.5,8),
                        "max_depth": [3,5,8],"max_features":["log2", "sqrt"],"criterion": ["friedman_mse"],
                        "subsample": [0.5, 0.618, 0.8, 0.85, 1.0], "n_estimators":[100,250,500]}

    #*******************
    

    cv_grid = GridSearchCV(grid_classifier, tuned_parameters, cv=10, scoring='roc_auc', n_jobs=-1, verbose=True)
    cv_grid.fit(xtrain, ytrain.ravel())

    print ("CV_Grid score: ", end = " ") 
    print (cv_grid.score(xtrain, ytrain.ravel()))

    print ("CV_Grid pest parameters: ", end = " ") 
    print (cv_grid.best_params_)

    b_estimator = cv_grid.best_estimator_
    b_estimator.fit(xtrain,ytrain.ravel())

    b_feats = b_estimator.feature_importances_
    np.save("bestfeats_gbc.npy", [b_feats, allnames])
    np.save("bestparams.npy", cv_grid.best_params_)
    #viz.bar(X=b_feats,opts=dict(stacked=False,rownames=allnames))

    return b_estimator, cv_grid.best_params_

def oneModel(themodel, xtrain, ytrain):


    tprs = []
    accscore = []
    fonescore = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    b_feats = []

    for train, test in rkf.split(xtrain):

        txtrain = xtrain[train]
        tytrain = ytrain[train]
        txtest = xtrain[test]
        tytest = ytrain[test]

        probz = themodel.fit(txtrain,tytrain.ravel()).predict_proba(txtest)
        fpr, tpr, thresholds = roc_curve(tytest.ravel(),probz[:,1])
        tprs.append(interp(mean_fpr,fpr,tpr))
        tprs[-1][0]=0.0
        aucs.append(auc(fpr,tpr))
        ypred = themodel.predict(txtest)
        fonescore.append(f1_score(tytest.ravel(),ypred))
        accscore.append(accuracy_score(tytest.ravel(),ypred))
        #b_feats.append(themodel.feature_importances_)

    return tprs, aucs, mean_fpr, accscore, fonescore, themodel, b_feats

def runScriptBro(xtrain, ytrain, tuned_parameters):
    
   
    modulename = 'boruta' 
    if modulename not in sys.modules:
        #print ('You have not imported the {} module for feature selection').format(modulename)
        print ("package not installed, install BorutaPy")
    
    else:
        rkf2 = RepeatedKFold(n_splits=5, n_repeats=50) 
        bro_feats = []
        bro_imp_mean = []
        bro_imp_med = []
        i = 0
        print ("Feature Importance")
        #after_tuned = {'max_depth': 8, 'n_estimators': 100}
        for train, _ in rkf2.split(xtrain):
            borutamodel = GradientBoostingClassifier(**tuned_parameters)
            model_feat = BorutaPy(borutamodel,perc=80,max_iter=20,verbose=2)
            model_feat.fit(xtrain[train], ytrain[train].ravel())
            bro_feats.append(model_feat.ranking_) # model_feat.ranking_: array of shape [n_features]
            tmp_mean, tmp_median = model_feat.check_imp_history()
            bro_imp_mean.append(tmp_mean)
            bro_imp_med.append(tmp_median)
            i+=1
            print ("Total Iteration: {}".format(i))
             
        #visualize best features from boruta package
        file_name = 'ivim_best_feats_list_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_feats)
        file_name = 'ivim_best_feats_imp_mean_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_imp_mean)
        file_name = 'best_feats_imp_med_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_imp_med)

def createAll():

    #create Feature matrix for the original and augmented files
    print ("Getting feature matrix of original maps")
    xtrain, ytrain = createFeatMat3(afiles, bfiles, dfiles, diff_files, perf_files, f_files)

    print ("Getting feature matrix of Crop 1 maps")
    c1xtrain, c1ytrain = createFeatMat3(c1afiles, c1bfiles, c1dfiles, c1diff_files, c1perf_files, c1f_files)
    print ("Getting feature matrix of Crop 2 maps")
    c2xtrain, c2ytrain = createFeatMat3(c2afiles, c2bfiles, c2dfiles, c2diff_files, c2perf_files, c2f_files)
    print ("Getting feature matrix of Crop 3 maps")
    c3xtrain, c3ytrain = createFeatMat3(c3afiles, c3bfiles, c3dfiles, c3diff_files, c3perf_files, c3f_files)
    print ("Getting feature matrix of Augmented orignal maps")
    augxtrain, augytrain = createFeatMat3(augafiles, augbfiles, augdfiles, augdiff_files, augperf_files, augf_files)

    #full augmented feature matrix
    full_aug_xtrain = np.vstack([c1xtrain,c2xtrain,c3xtrain,augxtrain])
    full_aug_ytrain = np.vstack([c1ytrain.reshape(-1,1),c2ytrain.reshape(-1,1),c3ytrain.reshape(-1,1),augytrain.reshape(-1,1)])

    #Random Shuffle of Augmentations

    randindx = np.random.choice(np.arange(len(full_aug_xtrain)),size=int(len(full_aug_xtrain)*0.9),replace=True)   # Chooses 80% of Augmented images

    xtrain = np.vstack((xtrain,full_aug_xtrain[randindx]))
    ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain[randindx]))

    return xtrain, ytrain

def runScript1():

    xset, yset = createAll()

    trainvalind = np.arange(0,int(len(xset)*0.75))

    xtrain = xset[trainvalind]
    ytrain = yset[trainvalind]
    xtest = xset[trainvalind[-1]:] # get the remaining items from the dataset
    ytest = yset[trainvalind[-1]:] # get the remaining items from the dataset
    #imr = SimpleImputer(missing_values=np.nan, strategy='median')
    #imr = imr.fit(xtest)
    #xtest = imr.transform(xtest)
    np.save("xtrain_80_ivim_ctrw.npy",xset)
    np.save("ytrain_80_ivim_ctrw.npy",yset)
    np.save("xtest_ivim_ctrw.npy",xtest)
    np.save("ytest_ivim_ctrw.npy",ytest)

    atitle = "Org and IVIM Maps Cross Validation"
    accscore, fonescore, std, importances, tprs, aucs, all_models = runClassifiers(xtrain, ytrain)
    print ("Finished Classification and now plotting")
    visualizeResults2(accscore, fonescore, tprs, aucs, fnames, atitle)
    xtestlen = len(xtest)
    for i in range(len(all_models)):
        model = all_models[i]
        print (classifiers2[i])
        testacc = []
        testauc = []
        testf1 = []
        for i in range(500):
            # prepare train and test sets
            indicies = resample(np.arange(xtestlen), n_samples=xtestlen-2, replace=False)
            # evaluate model
            ypred = model.predict(xtest[indicies])
            probz = model.predict_proba(xtest[indicies,:])
            fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
            testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
            testauc.append(auc(fpr,tpr))
            testf1.append(f1_score(ytest[indicies].ravel(),ypred))

        print (np.std(testf1), np.std(testacc), np.std(testauc))   
        print (np.mean(testf1), np.mean(testacc), np.mean(testauc)) 
        print (np.median(testf1), np.median(testacc), np.median(testauc))

# Run Hypertuning
def runScript2():
    
    xtrain = np.load("xtrain_80_ivim_ctrw.npy")
    ytrain = np.load("ytrain_80_ivim_ctrw.npy")
    xtest = np.load("xtest_ivim_ctrw.npy")
    ytest = np.load("ytest_ivim_ctrw.npy")
    
    #### COMMENT OUT ONCE NANS ARE FIXED
    #imr = SimpleImputer(missing_values=np.nan, strategy='median')
    #imr = imr.fit(xtrain)
    #xtrain = imr.transform(xtrain)
    
    b_estimator, best_params_ = runGridSearch(xtrain, ytrain, f2names, grid_classifier=GradientBoostingClassifier())
    
# Run Boruta
def runScript3():    
    
    xtrain = np.load("xtrain_80_ivim_ctrw.npy")
    ytrain = np.load("ytrain_80_ivim_ctrw.npy")
    xtest = np.load("xtest_ivim_ctrw.npy")
    ytest = np.load("ytest_ivim_ctrw.npy")
    bestparams = np.load("bestparams.npy")
    
    #### COMMENT OUT ONCE NANS ARE FIXED
    #imr = SimpleImputer(missing_values=np.nan, strategy='median')
    #imr = imr.fit(xtrain)
    #xtrain = imr.transform(xtrain)
    
    runScriptBro(xtrain, ytrain, bestparams)


#Run using created trained and test files
def runScript4(): 
    
    xtrain = np.load("xtrain_80_ivim_ctrw.npy")
    ytrain = np.load("ytrain_80_ivim_ctrw.npy")
    xtest = np.load("xtest_ivim_ctrw.npy")
    ytest = np.load("ytest_ivim_ctrw.npy")

    atitle = "Org and IVIM Maps Cross Validation"
    accscore, fonescore, std, importances, tprs, aucs, all_models = runClassifiers(xtrain, ytrain)
    print ("Finished Classification and now plotting")
    visualizeResults2(accscore, fonescore, tprs, aucs, fnames, atitle)
    xtestlen = len(xtest)
    for i in range(len(all_models)):
        model = all_models[i]
        print (classifiers2[i])
        testacc = []
        testauc = []
        testf1 = []
        for i in range(500):
            # prepare train and test sets
            indicies = resample(np.arange(xtestlen), n_samples=xtestlen-2, replace=False)
            # evaluate model
            ypred = model.predict(xtest[indicies])
            probz = model.predict_proba(xtest[indicies,:])
            fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
            testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
            testauc.append(auc(fpr,tpr))
            testf1.append(f1_score(ytest[indicies].ravel(),ypred))

        print (np.std(testf1), np.std(testacc), np.std(testauc))   
        print (np.mean(testf1), np.mean(testacc), np.mean(testauc)) 
        print (np.median(testf1), np.median(testacc), np.median(testauc))

runScript1()
print ("COMPLETED WOO")

