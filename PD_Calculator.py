#Copyright 2020, Gregory Happ, All rights reserved.
print("Copyright 2020, Gregory Happ, All rights reserved.")
print()
#Probability of Default (PD) calculator!!!
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import sklearn
from sklearn import svm
from sklearn import metrics
import pickle

# LOAD MODEL
pickle_in = open(r"C:\Users\Greg Happ\Desktop\Python Data Science\PD Project\pd_svm_model.pickle", "rb")
#print(pickle_in)
PD_SVM = pickle.load(pickle_in)
#print()
#print(PD_SVM)

#print('\n'*10)

print('Probability of default calculator for peer to peer lending!') 
print('The calculator uses a support vector machine (SVM) algorithm.')
print('The SVM was trained on more than 1300 peer-to-peer loans.')
print('When tested on more than 1.3 million peer-to-peer loans, it was 80 percent accurate.')
print('\n'*2)

def get_grade():
    grade_bool = True
    while grade_bool:
        LoanGrade = str(input("Please enter the FICO letter grade of the potential borrower: "))
        LoanGrade = LoanGrade.upper()
        if LoanGrade=='A' or LoanGrade=='B' or LoanGrade=='C' or LoanGrade=='D' or LoanGrade=='E' or LoanGrade=='F' or LoanGrade=='G':
            grade_bool = False
        else:
            print("Please enter an A, B, C, D, E, F or G for the FICO letter grade")
            
    return LoanGrade

def get_dti():
    dti_bool = True
    #user_input = input("Please enter the debt-to-income ratio of the potential borrower: ")
    while dti_bool:
        user_input = input("Please enter the debt-to-income ratio of the potential borrower: ")
        try:
            dti = int(user_input)
            dti_bool = False
        except ValueError:
            try:
                dti = float(user_input)
                dti_bool = False
            except ValueError:
                print("Please enter a number or the debt-to-income ratio")
            
    return dti

#Get loan data from user
grade = get_grade()
dtiratio = get_dti()
print('\n'*2)
print('FICO score grade:', grade)
print()
print('Debt-to-income ratio:', dtiratio)
print()


df = pd.DataFrame()
df['dti'] = [dtiratio]
df['grade'] = [grade]


#prepare data
df["A"] = df.grade == 'A'
df["B"] = df.grade == 'B'
df["C"] = df.grade == 'C'
df["D"] = df.grade == 'D'
df["E"] = df.grade == 'E'
df["F"] = df.grade == 'F'
df["G"] = df.grade == 'G'
df["A"] = df["A"]*1
df["B"] = df["B"]*1
df["C"] = df["C"]*1
df["D"] = df["D"]*1
df["E"] = df["E"]*1
df["F"] = df["F"]*1
df["G"] = df["G"]*1

#print(df)

#print()

x = df[['dti', 'A', 'B', 'C', 'D', 'E', 'F', 'G']]
predictions = PD_SVM.predict(x) # Gets a list of all predictions
prob_predictions = PD_SVM.predict_proba(x)

#print(predictions, prob_predictions, x)
print()
print('Probability of Default:', prob_predictions[0, 1])

print('\n'*2)
endinput = input('(Press enter to close)')