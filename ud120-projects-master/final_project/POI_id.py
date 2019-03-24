
# coding: utf-8

# # Main code for the project "Identify Fraud from Enron Mail"

# ## Preparation

# In[1]:

# Import needed libraries
import sys
import pickle
sys.path.append('../tools/')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
import pprint


# In[2]:

# Import feature_format and tester function. 
# They were modified so that to run with Python 3.7. Can be found in attachment.
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# ## Step 1a: Overview of  the initial dataset

# In[3]:

### Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'rb') as data_file:
    data_dict = pickle.load(data_file)


# In[4]:

### Store to my_dataset for easy export below.
my_dataset = data_dict


# In[5]:

# Format of the data?
type(my_dataset)


# In[6]:

# Nb of samples?
len(my_dataset)


# There are 146 people (i.e. samples in the dataset). Let's have a look at one sample:

# In[7]:

# Get the value of 1 key of the dictionary
next(iter(my_dataset.keys()))


# In[8]:

# Get the value associated to this key
next(iter(my_dataset.values()))


# In[9]:

# The features of each samples are stored in a dictionary.
# Nb of features?
len(next(iter(my_dataset.values())))


# We can check that, as stated in the project details page, each sample has:
# - 1 label (poi)
# - 14 financial features
# - 6 email features

# In[10]:

# How many POI in the dataset?
nb_NaN = 0
nb_POI = 0
nb_non_POI = 0

for k in my_dataset.keys():
    if my_dataset[k]['poi'] == 'NaN':
        nb_NaN +=1
    elif my_dataset[k]['poi']:
        nb_POI +=1
    elif not my_dataset[k]['poi']:
        nb_non_POI +=1


# In[11]:

print('Status on samples labels. NaN: {} , POI: {}  , Not POI: {}'.format(nb_NaN,nb_POI,nb_non_POI))


# Good piece of news is that we have the POI label for all the datapoint. 
# Bad piece of news is taht the classes are strongly imbalanced. This will make us use a stratified shuffling method while validating the classifiers we propose. 

# ## Step 1b: Further exploration of the features
# 
# To go further with the exploration of the available features (missing values, outliers...) we create to visualize them:

# In[12]:

def feature_explore (dataset_dict,feature,unit):
    '''Function that take the dataset dictionary, one of its numerical features, and its unit as inputs
    and returns the following, for the POI and non POI sub-groups:
    - count of NaN
    - min value
    - max value
    - mean value
    The box plot and histogram are also plotted for the whole data (POI and non POI together)'''
    # initialize list of values, when existing
    val = []
    # initialize assiated list POI (True --> 1; False --> 0)
    val_POI = []
    # initialize count of NaN, for POI and non POI
    NaN_POI_nb = 0
    NaN_nonPOI_nb = 0 
    
    for k in my_dataset.keys():
        if my_dataset[k][feature] == 'NaN':
            if my_dataset[k]['poi'] == True:
                NaN_POI_nb += 1
            else:
                NaN_nonPOI_nb += 1
        else:
            val.append(int(my_dataset[k][feature]))
            if my_dataset[k]['poi'] == True:
                val_POI.append(1)
            else:
                val_POI.append(0)
    
    # Convert to numpy arrays for later calculations
    val = np.array(val)
    val_POI = np.array(val_POI)
    
    # Compute and display count of NaN, min, max, mean for POI and non POI
    print('Feature: {} ({})'.format(feature,unit))
    print('Count of NaN - POI: {:d} \t\t non-POI: {:d}'.format(NaN_POI_nb,NaN_nonPOI_nb))
    print('Min value    - POI: {:d} \t non-POI: {:d}'.format(min(val[val_POI==1]),min(val[val_POI==0])))
    print('Max value    - POI: {:d} \t non-POI: {:d}'.format(max(val[val_POI==1]),max(val[val_POI==0])))
    print('Mean value   - POI: {:.0f} \t non-POI: {:.0f}'.format(np.mean(val[val_POI==1]),np.mean(val[val_POI==0])))
    
    # Box plot and histogram
    green_diamond = dict(markerfacecolor='g', marker='D')
    plt.figure(figsize=(10,5))            
    ax1 = plt.subplot(121)
    ax1.boxplot([val[val_POI==1],val[val_POI==0]], flierprops=green_diamond)
    plt.xticks([1, 2], ['POI','non POI'])
    plt.ylabel(feature+' '+unit)
    
    ax2 = plt.subplot(122)
    ax2.hist([val[val_POI==1],val[val_POI==0]],alpha=0.8,bins=25)
    plt.xlabel(feature+' '+unit)
    plt.ylabel('count')
    plt.legend(['POI','non POI'])
    


# Now we use this function on the financial features, beginning with the salary:

# In[13]:

feature_explore (my_dataset,'salary','$')


# There is an obvious outlier, which correspond to the salary of a non POI. To be able to find the name of outliers, we create another simple function:

# In[14]:

def display_records(dataset_dict,feature,threshold,above = True):
    '''Function that take the dataset dictionary, one of its numerical features, and a threshold as inputs
    and returns the samples (people records) whose feature is above (above = True) or below (above = False) the 
    given threshold'''
    if above:
        for k in my_dataset.keys():
            if my_dataset[k][feature] != 'NaN' and int(my_dataset[k][feature]) >= threshold:
                print(k)
                pprint.pprint(my_dataset[k])
                print('')
    else:
        for k in my_dataset.keys():
            if my_dataset[k][feature] != 'NaN' and int(my_dataset[k][feature]) <= threshold:
                print(k)
                pprint.pprint(my_dataset[k])
                print('')


# We use this function to find the non POI salary outlier: 

# In[15]:

display_records(my_dataset,'salary',2.5e7)


# We find that the identified outlier is not a person but a sample that contains the total for the financial features. We pop it from the dictionary and repeat the process.

# In[16]:

print(len(my_dataset))
my_dataset.pop('TOTAL')
print(len(my_dataset))


# ### "Promising" financial features

# In[17]:

feature_explore (my_dataset,'salary','$')


# Several interesting points:
# - one third of the salary data is missing, but at least we have it for all the POI except 1
# - the median salary of the POI sugroup is higher than that of the non POI subgroup. Intersting feature for our classifier!
# - 3 clear outliers: 2 POI and 1 non POI
# - the 'shape' of the salary distribution seems to be quite similar for POI and non POI 
# 
# Let's have a look at the 3 clear outliers mentioned above:

# In[18]:

display_records(my_dataset,'salary',1e6)


# The 2 POIs are Kenneth Lay and Jeffrey Skilling, respectively Chairman of the Board of Directors and COO of Enron when the scandal occured.
# The non POI is Mark Frevert, Enron vice chairman.
# 
# Now let's look at the other financial features:

# In[19]:

feature_explore (my_dataset,'bonus','$')


# In[20]:

feature_explore (my_dataset,'restricted_stock','$')


# In[21]:

feature_explore (my_dataset,'exercised_stock_options','$')


# In[22]:

feature_explore (my_dataset,'long_term_incentive','$')


# The 'salary', 'bonus', 'restricted stock', 'exercised_stock_options' and 'long_term_incentive' features do not have too many missing values, and they seem to be able to help distinguish between POIs and non-POIs.

# ### 'expenses' and 'other'

# In[23]:

feature_explore (my_dataset,'expenses','$')


# In[24]:

feature_explore (my_dataset,'other','$')


# For the 'other' feature, the 2 main outliers are again Kenneth Lay (POI) and Mark Frevert (non POI).
# 
# The 'expenses' and 'other' financial features do not seem to bring any useful information to distinguish between POIs and non POIs.

# ### Features with many missing values

# In[25]:

feature_explore (my_dataset,'deferred_income','$')


# Too many missing data, in particular for the POIs.

# In[26]:

feature_explore (my_dataset,'deferral_payments','$')


# Same comment for 'deferral_payments'...

# In[27]:

feature_explore (my_dataset,'loan_advances','$')


# Only 1 POI has a value for 'loan_advances': we cannot generalize with this feature.

# In[28]:

# feature_explore (my_dataset,'director_fees','$')


# In[29]:

# feature_explore (my_dataset,'restricted_stock_deferred','$')


# The feature_explore function gives a ValueError with the 'director_fees' and 'restricted_stock_deferred' financial features because we have no value for theses feature for all the POIs ! They will definitely not be useful for our classifier.

# Too few data (notably for the POIs) for the following financial features: loan_advances, director_fees, deferral_payments, deferred_income, restricted_stock_deferred .

# ### 'Total' financial features

# In[30]:

feature_explore (my_dataset,'total_payments','$')


# There is a very clear outlier, from the POI subgroup: Kenneth Ley. 

# In[31]:

feature_explore (my_dataset,'total_stock_value','$')


# For the financial features, **we will test 2 strategies**:
# - **strategy A**: we keep only the 2 overall features, i.e. 'total_payments' and 'total_stock_value'. By summing the different financial features, we ensure that we have a value for a maximum number of people.
# - **strategy B**: we disregard:
#     - the 'total' features, because by summing different kinds of financial quantities we believe they might 'dilute' the information we need to distinguish between POIs and non POIs.
#     - the features for which there were too many missing values: 'loan_advances', 'director_fees', 'deferral_payments', 'deferred_income', 'restricted_stock_deferred'
#     - the features for which we saw that there was not a significant difference between POIs and non-POIs: 'other',  'expenses'
# 
# Whatever the strategy, **we decide to keep the outliers** of the selected features. Indeed, we saw that there were at least two very emblematic POIs, Kenneth Lay and Jeffrey Skilling, who presented outliers for one or several financial features. Given the limited number of POIs in the dataset, we don't want to further reduce their proportion in the data that will be used to train the algorithm. 

# ### Email features

# In[32]:

feature_explore (my_dataset,'from_messages','')


# In[33]:

feature_explore (my_dataset,'to_messages','')


# In[34]:

feature_explore (my_dataset,'from_this_person_to_poi','')


# In[35]:

feature_explore (my_dataset,'from_poi_to_this_person','')


# In[36]:

feature_explore (my_dataset,'shared_receipt_with_poi','')


# In the frame of our POI identifier algorithm, there is no need to use the 'email_address' feature.
# 
# At first sight, the 'shared_receipt_with_poi' email feature looks promising. 
# 
# On the other hand, the 'from_this_person_to_poi' and 'from_poi_to_this_person' features might not be so expressive if we don't 'normalize' them by the total 'from_message' and 'to_message' respectively. Indeed, if we make the assumption that the fraud system was established by a limited number of people who know each others, then it is not the absolute number of mails send to or received from a POI that matters, but the ration to the total mailbox. If A and B send 30% and 5% of their emails to C respectively, then C is likely to be closer to A than B. Later on, we will check this intuition by using the original poi email features, then replacing them by the 'relative' poi email features, and comparing the erformance of teh POI identifier algorithm.
# 

# ## Step 2: Feature selecion and engineering
# 
# ### Create relative POI email features
# As explained above, we use the piece of code below to compute, for each person of the dataset:
# - the ratio of emails sent to a POI over total number of sent emails
# - the ratio of emails received from a POI over the total number of received emails

# In[37]:

for k in my_dataset.keys():
    if my_dataset[k]['from_poi_to_this_person'] != 'NaN' and my_dataset[k]['to_messages'] != 'NaN':
        my_dataset[k]['relative_from_poi_to_this_person'] =         float(my_dataset[k]['from_poi_to_this_person'])/float(my_dataset[k]['to_messages'])
    else: 
        my_dataset[k]['relative_from_poi_to_this_person'] = 'NaN'
        
    if my_dataset[k]['from_this_person_to_poi'] != 'NaN' and my_dataset[k]['from_messages'] != 'NaN':
        my_dataset[k]['relative_from_this_person_to_poi'] =         float(my_dataset[k]['from_this_person_to_poi'])/float(my_dataset[k]['from_messages'])
    else:
        my_dataset[k]['relative_from_this_person_to_poi'] = 'NaN'


# Now let's have a look at these new features by making a scatter plot:

# In[38]:

# Create matrix X with the 2 'relative' POI email features, and vector y of labels
X = []
y = []

for k in my_dataset.keys():
    if my_dataset[k]['relative_from_poi_to_this_person'] != 'NaN' and     my_dataset[k]['relative_from_this_person_to_poi'] != 'NaN':
        X.append([my_dataset[k]['relative_from_poi_to_this_person'],my_dataset[k]['relative_from_this_person_to_poi']])
        if my_dataset[k]['poi']:
            y.append(1)
        else:
            y.append(0)

X = np.array(X)
y = np.array(y)


# We then use the short scatter plot function below:

# In[39]:

def plot2Dscatter(X,y,features,labels):
    plt.scatter(X[y == labels[0],0], X[y == labels[0],1], color = 'b', label=str(labels[0]))
    plt.scatter(X[y == labels[1],0], X[y == labels[1],1], color = 'r', label=str(labels[1]))
    plt.legend()
    plt.xlabel(features[0])
    plt.ylabel(features[1])


# In[40]:

features = ['relative_from_poi_to_this_person','relative_from_this_person_to_poi']
labels = [1,0]
plot2Dscatter(X,y,features,labels)


# Quite interestingly all the POIs have a ration of emails sent to other POI equal or above about 0.2, that is they all sent 1 email in 5 to another POI.

# ### Rescaling or not rescaling...

# In the investigation, we want to try different kinds of algorithms for our classifier. Some of them, like SVM or k-means clustering, would behave differently whether applied to scaled or unscaled features. In our case, we decided, as explained above, not to remove the outliers in the financial features. Because of this choice, min/max scaling would not be a good approach for scaling the data. 
# 
# We decide to **start by using the unscaled features**. If we cannot manage to get the desired performance for our classifier, we will go back to this step and see which scaling method could be used.

# ### Feature selection
# 
# As dicussed in the previous sections, we will test 3 sets of features:
# - the lists 1 and 3 differ by the used financial features ('total' vs. 'handpicked' features)
# - the lists 2 and 3 differ by the used email features ('absolute' vs. 'relative' POI mail features')
# 
# We will also create, as a reference list of features, a list of 5 features among the original features, selected using SelectKBest class.

# In[41]:

### The first feature must be "poi".
# List 1: 5 features
features_list1 = ['poi','total_payments','total_stock_value','shared_receipt_with_poi',                  'relative_from_poi_to_this_person','relative_from_this_person_to_poi'] 
# List 2: 8 features
features_list2 = ['poi','bonus','exercised_stock_options','long_term_incentive','restricted_stock','salary',                  'shared_receipt_with_poi','from_poi_to_this_person','from_this_person_to_poi']
# List 3: 8 features as well
features_list3 = ['poi','bonus','exercised_stock_options','long_term_incentive','restricted_stock','salary',                  'shared_receipt_with_poi','relative_from_poi_to_this_person','relative_from_this_person_to_poi'] 


### Extract features and labels from dataset for local testing
data1 = featureFormat(my_dataset, features_list1, sort_keys = True)
labels1, features1 = targetFeatureSplit(data1)

data2 = featureFormat(my_dataset, features_list2, sort_keys = True)
labels2, features2 = targetFeatureSplit(data2)

data3 = featureFormat(my_dataset, features_list3, sort_keys = True)
labels3, features3 = targetFeatureSplit(data3)


# In[42]:

# List 4: 'best' 5 features, in teh sense of higher f-score
# Select all features except email_address
features_list4 = ['poi','bonus','deferral_payments','director_fees','exercised_stock_options','expenses',                  'from_messages','from_poi_to_this_person','from_this_person_to_poi','loan_advances',                  'long_term_incentive','other','restricted_stock','restricted_stock_deferred','salary',                  'shared_receipt_with_poi','to_messages','total_payments','total_stock_value'] 
# Extract features and labels from dataset
data4 = featureFormat(my_dataset, features_list4, sort_keys = True)
labels4, features4 = targetFeatureSplit(data4)
# Use SelectKBest to select the features having the 5 highest f-score
from sklearn.feature_selection import SelectKBest
Kbest_selector = SelectKBest(k=5)
features4_new = Kbest_selector.fit_transform(features4, labels4)


# Let's see which features have been selected. We compare feature4_new and feature4 for the first sample:

# In[43]:

print(features4_new[0])
print(features4[0])


# We see that the features index 0 ('bonus'), 3 ('exercised_stock_options'), 9 ('long_term_incentive'), 13 ('salary') and 17 ('total_stock_value') have been selected. Note that there is no email features in this selection. Let's have a look at the f-score for these features: 

# In[44]:

Kbest_selector.scores_


# The selected features have a f-score between 10 and 25.

# These lists do not contain a very high number of features, and therefore we do not see the need, as a first approach, to use a dimensionality reduction technique like PCA. If the performance of the classifier is below expectetation we will go back to this step and try to use a PCA on a the list number 3.

# Now we want to use a standard Naive Bayes classifier to have an idea of the difference of performance between the 4 features lists. We write a simple function to do that:

# In[45]:

def NB_clf_classif_report(features, labels):
    '''Function that takes a dataset (features and labels) as input, splits it into training and test subsets
    (in stratified manner because we are dealing with imbalanced classes), creates a standard Naive Bayes classifier,
    fits it to the training data, and then displays the classification report using the test data.'''
    
    # split features and labels into training and testing sets
    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.5, random_state=42, stratify=labels)
    
    # import NB classifier from sk learn library
    from sklearn.naive_bayes import GaussianNB
    # create classifier
    clf = GaussianNB()
    
    # fit classifier using training dataset
    clf = clf.fit(features_train, labels_train)
    
    # compute recall, precision and f1-score, then display them
    from sklearn.metrics import classification_report
    labels_pred = clf.predict(features_test)
    print(classification_report(labels_test, labels_pred))# compute accuracy score


# In[46]:

print('Performance with feature list nb1: ')
NB_clf_classif_report(features1, labels1)
print('')
print('Performance with feature list nb2: ')
NB_clf_classif_report(features2, labels2)
print('')
print('Performance with feature list nb3: ')
NB_clf_classif_report(features3, labels3)
print('')
print('Performance with feature list nb4: ')
NB_clf_classif_report(features4_new, labels4)


# When interpreting these results, we are mainly focusing in the recall and also precision for the POI class. Indeed, we are in a case of very imbalanced classification, so we want to ensure that our algorithm will be able to spot the rare positive samples in the dataset (i.e "high" recall). Given the context of this analysis, we want to give the priority at the recall over the precision. Indeed, the algorithm could be used as a first filter to detect people for which further investigation may lead to findings. The precision may not be very high, but at least we will minimize the missing of true POIs. 
# 
# Going back to the results, we see that the third list of features, containing the five financial features that seemed particularly relevant following our manual exploration of the dataset, the shared receipt email feature and also the two created email features. From now on, we use this list of features.  

# In[47]:

print('Using this list of 8 features, the dataset contains {} samples among which there are {} POIs.'.     format(len(features3),sum(labels3)))
print('The POI class represents only {:.1%} of the dataset.'.format(sum(labels3)/len(features3)))


# ## Step 3: Algorithm choice and tuning
# 
# Here we will test 4 classifiers: Naive Bayes (NB), Support Vector Machine (SVM), k-nearest neighbors (kNN) and Decision Tree (DT). In the case of SVM, kNN and DT, we will perform a grid search to find the best parameters for these classifiers. 
# 
# To assess the performance of the classifier pending on its parameters, we will look at three metrics: recall score, precision score, and f1-score.

# In[48]:

# split features and labels into training (50%) and testing (50%) sets, 
# in stratified manner because we are dealing with imbalanced classes
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features3, labels3, test_size=0.5, random_state=42, stratify=labels3)


# ### Naive Bayes
# 
# We do not perform any specific tuning of the parameters here.

# In[49]:

# import NB classifier from sk learn library
from sklearn.naive_bayes import GaussianNB
# create classifier
clf = GaussianNB()
    
# fit classifier using training dataset
clf = clf.fit(features_train, labels_train)
    
# compute recall, precision and f1-score, then display them
from sklearn.metrics import classification_report
labels_pred = clf.predict(features_test)
print(classification_report(labels_test, labels_pred))


# The recall of POIs is below target (0.3).

# ### SVM
# 
# In this case we perform a grid search, trying:
# - two different kernel types: rbf and sigmoid. Based on the visualization we did before, we think it would be difficult for a linear kernel to be able to separate well between POIs and non-POIs.
# - three different values for the gamma kernel coefficient: 1, 0.1 and 0.01
# - four different values for the penalty parameter C: 10, 100, 1000 and 10000

# In[50]:

# Dictionary of the parameters, and the possible values they may take. 
# Based on the visualization we did before, we think it difficult for a linear kernel to 
# be able to separate well between POIs and non-POIs. We therefore limit our grid to a rbf kernel.
# We try 3 diferent C (possible choices are 10, 100 and 1000), and 
# 2 different values for gamma (0.1 and 0.01)
parameters = {'kernel':['rbf','sigmoid'], 'gamma': [1, 0.1, 0.01], 'C': [10, 100, 1000, 10000]}


# In[51]:

# We first create only the 'algorithm object, that is, the classifier and the possible sets of parameters
# we set the class_weight parameters to 'balanced' to give more weight to the unfrequent class (POIs)
from sklearn.svm import SVC
svc = SVC(class_weight = 'balanced')


# In[52]:

# We pass the algorithm (svc) and the dictionary of parameters to try (parameters) and 
# it generates a grid of parameter combinations to try.
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svc, parameters, scoring = 'recall')


# In[53]:

# The fit function now tries all the parameter combinations, 
# and returns a fitted classifier that's automatically tuned to the optimal parameter combination.
clf.fit(features_train,labels_train)


# Now, for the display of the grid search results, we define the function below:

# In[54]:

def grid_results_display(clf, features_test, labels_test):
    '''Function that takes a gridsearch fitted classifier, the test features and labels as input, 
    and displays the best combination of parameters, the grid scores for the different combinations of the grid
    and finally the classification report for the best combination (using the test data).
    # For this function, we re-used and adapted an example found in the sklearn documentation (ref in references.txt)'''

    print('Best parameters set found on training set:')
    print()
    print(clf.best_params_)
    print()
    print('Grid scores on training set:')
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('{:.3f} (+/-{:.03f}) for {}'.format(mean, std * 2, params))
    print()
    print('Detailed classification report:')
    print()
    print('The model is trained on the full training set.')
    print('The scores are computed on the full test set.')
    print()
    labels_pred = clf.predict(features_test)
    print(classification_report(labels_test, labels_pred))
    print()


# In[55]:

grid_results_display(clf, features_test, labels_test)


# The only tested parameter that has an effect in our classification task seems to be the kernel: sigmoid kernel works better than rbf kernel. Nevertheless, the precision of POI detection is very low and below target (0.3).  

# ### kNN
# 
# In this case also we will perform a grid search, trying:
# - five different numbers of neighbors: 2, 3, 4, 5, 6 
# - three different algorithms: ‘ball_tree’, ‘kd_tree’, ‘brute’

# In[56]:

# dictionary of parameters
parameters = {'n_neighbors': [2, 3, 4, 5, 6], 'algorithm': ['ball_tree', 'kd_tree', 'brute']}


# In[57]:

# create algorithm object
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()


# In[58]:

# create classifier from algorithm and dictionary of parameters using GridSearchCV
clf = GridSearchCV(neigh, parameters, scoring = 'recall')


# In[59]:

# Fit the classifier (tries all the parameter combinations)
clf.fit(features_train,labels_train)


# In[60]:

grid_results_display(clf, features_test, labels_test)


# The kNN classifier works with reasonable recall and precision when tuned with a number of neighbors of 3, with no notable difference between the different algorithm we tested. 

# ### Decision Tree
# 
# In this case we also perform a grid search, trying:
# - two kinds of criterions: 'gini' and 'entropy'
# - four different max_depth parameters: 1, 2, 4 and 8
# - three different min_samples_split settings: 2, 4 and 8 

# In[61]:

# dictionary of parameters
parameters = {'criterion': ['gini','entropy'], 'max_depth': [1, 2, 4, 8], 'min_samples_split': [2, 4, 8]}


# In[62]:

# create algorithm object
# we set the class_weight parameters to 'balanced' to give more weight to the unfrequent class (POIs)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(class_weight = 'balanced')


# In[63]:

# create classifier from algorithm and dictionary of parameters using GridSearchCV
clf = GridSearchCV(dt, parameters, scoring = 'recall')


# In[64]:

# Fit the classifier (tries all the parameter combinations)
clf.fit(features_train,labels_train)


# In[65]:

grid_results_display(clf, features_test, labels_test)


# We achieve the best performance so far with a decision tree classifier with a maximum_depth parameter of 1. With such value for maximum_depth, the type of criterion (gini or entropy) and the min_samples_split parameters seem to have little effect. 

# We propose to submit this algorithm (Decision Tree classifier, {'criterion': 'gini', 'max_depth': 1, 'min_samples_split': 2}) for test using the tester code. Note that we also set the class_weight setting to 'balanced' because we want to give more weight to the POIs samples (see sklearn DT classifier documentation).

# In[68]:

clf = DecisionTreeClassifier(criterion='gini', max_depth=1, min_samples_split=2, class_weight='balanced')


# ## Step4: validation and evaluation
# 
# We start by dumping the classifier, the selected features and corresponding data:

# In[69]:

dump_classifier_and_data(clf, my_dataset, features_list3)


# Now we use the tester.py (modified so that to work with model_selection.StratifiedShuffleSplit instead of the previous version cross_validation.StratifiedShuffleSplit). As stated in the starter code, "because of the small size of the dataset, the script uses stratified shuffle split cross validation.

# In[70]:

get_ipython().magic('run tester.py')


# After this evaluation based on 1000 folds of the initial dataset, the proposed set of features together with the tuned classifier reach a recall of 0.6 and a precision slightly above 0.3.
