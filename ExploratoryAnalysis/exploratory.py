# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from surprise import SVDpp, Reader, Dataset
from surprise.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readCSV(path):
    with open(path, 'r') as f:
        f.readline()  # Skip the header line if there's one
        for l in f:
            fields = l.strip().split(',')
            if len(fields) == 9:  # Make sure there are exactly 9 fields
                gender, age, r, field4, field5, field6, field7, field8, field9 = fields
                r = int(r)  # Assuming r is the one that should be converted to an int
                yield gender, age, r, field4, field5, field6, field7, field8, field9
            else:
                print(f"Skipping invalid line: {l.strip()}") 

# %%
allPatients = [] #all of the data 
for l in readCSV("diabetes_prediction.csv"):
    allPatients.append(l)

# %%
allPatients[0]

# %%
df = pd.read_csv('diabetes_prediction.csv')
stats = df.describe()
print(stats)

# %%
#average bmi being around 27 indicates they are mostly overweight 
#average age is 41, slightly skewed towards female as ~58k are female, and ~41k are male
gender_num = df['gender'].value_counts()
print(gender_num)
diabetes_count = df['diabetes'].value_counts(); 
print(diabetes_count)

# %%
correlation = df.select_dtypes(include=['number']).corr()['HbA1c_level']

#sort the correlations 
correlation_with_target = correlation.sort_values(ascending=False)


print(correlation_with_target)

# %%
len(allPatients)

# %%
# Filter entries where HbA1c_level > 6.5%
hba1c_filtered = df[df['HbA1c_level'] > 6.5]
diabetes_percentage_hba1c = (hba1c_filtered['diabetes'].sum() / len(hba1c_filtered)) * 100
print(f"Percentage of entries with diabetes (HbA1c > 6.5%): {diabetes_percentage_hba1c:.2f}%")

#filtering entries where the bmi is greater than 25
bmi_filtered = df[df['bmi'] > 25]
diabetes_bmi = (bmi_filtered['diabetes'].sum()/ len(bmi_filtered)) * 100
print(f"Percentage of entries with diabetes (bmi): {diabetes_bmi:.2f}%")

#blood glucose, 126: 
blood_glucose_filtered = df[df['blood_glucose_level'] > 126]
diabetes_percentage_bloodGlucose = (blood_glucose_filtered['diabetes'].sum()/len(blood_glucose_filtered)) * 100 
print(f"Percentage of entries with diabetes blood glucose > 126: {diabetes_percentage_bloodGlucose:.2f}%")




# %%
hypertension_filtered = df[df['hypertension'] == 1]
diabetes_percentage_hypertension = (hypertension_filtered['diabetes'].sum()/len(hypertension_filtered)) * 100
print(f"Percentage of people with hypertension who also have diabetes: {diabetes_percentage_hypertension:.2f}%")

diabetes_filtered = df[df['diabetes'] == 1]
hypertension_percentage_diabetes = (diabetes_filtered['hypertension'].sum() / len(diabetes_filtered)) * 100
print(f"Percentage of people with diabetes who also have hypertension: {hypertension_percentage_diabetes:.2f}%")

#insight into how some of these factors connect to one another
#example) age -> hba1c levels 



# %%
def accuracy(predictions, y):
    correct = predictions == y # Binary vector indicating which predictions were correct
    return sum(correct) / len(correct)

# %%
def BER(pred, y):
    TP = np.sum([(p and l) for (p,l) in zip(pred, y)])
    FP = np.sum([(p and not l) for (p,l) in zip(pred, y)])
    TN = np.sum([(not p and not l) for (p,l) in zip(pred, y)])
    FN = np.sum([(not p and l) for (p,l) in zip(pred, y)])
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    BER = 1 - 1/2 * (TPR + TNR)
    return BER

# %%
file_path_train = "diabetes_train.csv"
file_path_test = "diabetes_test.csv"
dataTrain = pd.read_csv(file_path_train) #, index_col = "Unnamed:0")
dataTest = pd.read_csv(file_path_test)
dataTest.rename(columns = {"Unnamed: 0": "index"}, inplace = True)
dataTrain.rename(columns = {"Unnamed: 0": "index"}, inplace = True)

# %%
dataTrain.columns

# %%
#define a feature for HbA1c levels and blood_glucose levels
def feature(row, dataTrain = dataTrain):
    feat1 = dataTrain.loc[row]['HbA1c_level']
    feat2 = dataTrain.loc[row]['blood_glucose_level']
    return [1] + [feat1] + [feat2]

# %%
X_train = [feature(row) for row in range(len(dataTrain))]
y_train = [dataTrain.loc[row]['diabetes'] for row in range(len(dataTrain))]
X_test = [feature(row) for row in range(len(dataTest))]
Y_test = [dataTest.loc[row]['diabetes'] for row in range(len(dataTest))]

# %%
mod = linear_model.LogisticRegression(C=1)
mod.fit(X_train,y_train)
pred = mod.predict(X_test)

# %%
BER(pred, Y_test)

# %%
best_accuracy = 0 
best_threshold = 0
for threshold in np.arange(6.5, 6.7, 0.01):
    ypred = [1 if dataTrain.loc[row]['HbA1c_level'] > threshold else 0 for row in dataTrain.index]
    correct = [1 if dataTrain.loc[row]['diabetes'] == pred else 0 for pred,row in zip(ypred, dataTrain.index)]
    accuracy = sum(correct)/len(correct)
    if accuracy > best_accuracy:
        best_accuracy = accuracy 
        best_threshold = threshold 
    print(best_threshold)

# %%
ypred = [1 if dataTest.loc[row]['HbA1c_level'] > best_threshold else 0 for row in dataTest.index]
correct = [1 if dataTest.loc[row]['diabetes'] == pred else 0 for pred,row in zip(ypred, dataTest.index)]
y = [dataTest.loc[row]['diabetes'] for row in dataTest.index]
accuracy = sum(correct)/len(correct)
print(accuracy)

# %%
##accuracy function 
def accuracy(predictions, y):
    TP = np.sum([1 for p, l in zip(predictions, y) if p == l == True])
    FP = np.sum([1 for p, l in zip(predictions, y) if p == 1 and l == False])
    TN = np.sum([1 for p, l in zip(predictions, y) if p == l == False])
    FN = np.sum([1 for p, l in zip(predictions, y) if p == 0 and l == True])

    
    return (TP + TN) / (TP + FP + TN + FN) #returning accuracy 

# %%
##Trying out Naive Bayes 
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
bayes_accuracy = accuracy(predictions, Y_test)
bayes_accuracy

# %%
model2 = RandomForestClassifier(n_estimators=76)
model2.fit(X_train, y_train)
predictions2 = model.predict(X_test)
randomForest_accuracy = accuracy(predictions2, Y_test)
randomForest_accuracy

# %%
X_train = np.array(X_train)  
smote = SMOTE(random_state=42)

##applying SMOTE
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:")
print(pd.Series(y_train).value_counts())
print("After SMOTE:")
print(pd.Series(y_train_res).value_counts())

# %%
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_res, y_train_res)

# 3. Predict using the test set
y_pred = rf_clf.predict(X_test)

# 4. Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy after SMOTE (random classifier):", accuracy)

# %%
# Initialize the Naive Bayes classifier
nb_clf = GaussianNB()

# Train the Naive Bayes model on the resampled data (or original data if you don't want to use SMOTE)
nb_clf.fit(X_train_res, y_train_res)
y_pred = nb_clf.predict(X_test)
accuracySMOTE= accuracy_score(Y_test, y_pred)
print(f"Accuracy naivey bayes (SMOTE): {accuracySMOTE}")

# %%
##SMOTE logistic regression
linearSMOTE = linear_model.LogisticRegression(C=1)
linearSMOTE.fit(X_train_res, y_train_res)
linearPrediction = linearSMOTE.predict(X_test)
accuracyLinearSMOTE = accuracy_score(linearPrediction, Y_test)
accuracyLinearSMOTE

# %%
# Using the existing or new threshold to predict diabetes
ypred_smote = [1 if row['HbA1c_level'] > best_threshold else 0 for _, row in dataTest.iterrows()]

# Calculate accuracy
correct_smote = [1 if actual == pred else 0 for actual, pred in zip(dataTest['diabetes'], ypred_smote)]
accuracy_smote = sum(correct_smote) / len(correct_smote)

print(f"Accuracy after SMOTE: {accuracy_smote}")


