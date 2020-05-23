#Copyright 2020, Gregory Happ, All rights reserved.
print("Copyright 2020, Gregory Happ, All rights reserved.")
print()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import sklearn
from sklearn import svm
from sklearn import metrics
import pickle

#######################################################################################################################
#Load the data (note: make sure to enter the full file path (if the file isn't in your main directory): )
df = pd.read_csv(r"loan.csv", usecols=["loan_status", "dti", "grade"])
########################################################################################################################
###################################################################################################################

print("DF's Shape:")
print(df.shape)
print()

############################
print("Check for null values")
#check for null values
print(df.isnull())
# Total missing values for each feature
print(df.isnull().sum())
# Any missing values?
print(df.isnull().values.any())
# Total number of missing values
print(df.isnull().sum().sum())
##############################
print()
print("Clean data")
df = df.dropna()
print()
print("Check for null values")
#check for null values
print(df.isnull())
# Total missing values for each feature
print(df.isnull().sum())
# Any missing values?
print(df.isnull().values.any())
# Total number of missing values
print(df.isnull().sum().sum())
print()

print("DF's Shape:")
print(df.shape)
print()

print("Prepare Data:")
#drop 'current' loans
df = df[df.loan_status != 'Current']
#Make New columns
df["default"] = df.loan_status == 'Default'
df["charged_off"] = df.loan_status == 'Charged Off'
df["dnmcp_charged_off"] = df.loan_status == 'Does not meet the credit policy. Status:Charged Off'
#Convert new boolean columns into integers:
df["default"] = df["default"]*1
df["charged_off"] = df["charged_off"]*1
df["dnmcp_charged_off"] = df["dnmcp_charged_off"]*1
#Add columns together, to make a column that simply says if the loan defaulted or not (defaulted and charged off are basically the same thing):
Defaulted = df["default"] + df["charged_off"] + df["dnmcp_charged_off"]
df['Defaulted'] = Defaulted
#delete unnessecary columns:
del df['loan_status']
del df['default']
del df['charged_off']
del df['dnmcp_charged_off']

print()
print('DF Shape:')
print(df.shape)
print('\n'*10)



#########################################################################3
#Basic Data Analysis and Data Visualization:

#Lending Club Average Default Risk
print('P2P Loan Average Default Risk:')
print(df['Defaulted'].mean())
print()

#Data Vizualization:

my_data = df['Defaulted'].value_counts()
colors = ["#ff9999", "#99ff99"]
#colors = ["#f15854", "#60bd68"]
#['#ff9999','#66b3ff','#99ff99','#ffcc99']
#colors = ["red", "green"]
#colors = ["red", "lime"]
mylabels = 'Defaulted', 'Did not default'
sizes = [my_data[1], my_data[0]]
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
#plt.pie(sizes, labels=mylabels, shadow=True, colors=colors, explode=(0, 0), startangle=90, autopct='%1.1f%',)
fig1, ax1 = plt.subplots()
patches, texts, autotexts = ax1.pie(sizes, explode=(0,0.1), labels=mylabels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=colors)
for texts in texts:
    texts.set_color('grey')
for autotexts in autotexts:
    autotexts.set_color('grey')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()

############################################################################
#Debt-to-Income Ratio Summary Statistics
print('Debt-to-Income Ratio Summary Statistics')
print(df['dti'].describe())
print()
#Data Vizualization

labelA, labelB = ['Debt-to-Income Ratios without outliers'], ['Debt-to-Income Ratios with outliers']
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# rectangular box plot
bplot1 = ax1.boxplot(df['dti'],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labelA,
                     showfliers=False)  # will be used to label x-ticks
ax1.set_title('')
# notch shape box plot
bplot2 = ax2.boxplot(df['dti'],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labelB,
                     showfliers=True)  # will be used to label x-ticks
ax2.set_title('')
# fill with colors
colors = ['lightblue']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
# adding horizontal grid lines
for ax in [ax1, ax2]:
    ax.yaxis.grid(True)
    ax.set_ylabel('Debt-to-Income Ratio')
plt.show()

#########################################################################################


#Credit Risk Grade Relative Frequency Table
print('Credit Risk Grade Relative Frequency Table')
my_tab = pd.crosstab(index=df['grade'],  # Make a crosstab
                              columns="Percent",      # Name the count column
                              normalize="columns")      
print(my_tab)
print()
#Data Vizualization:
my_data = df['grade'].value_counts()
#colors = ["#ff9999", "#99ff99"]
#colors = ["#f15854", "#60bd68"]
#['#ff9999','#66b3ff','#99ff99','#ffcc99']
#colors = ["red", "green"]
#colors = ["red", "lime"]
#colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#colors = ['lightyellow', 'lightcyan', 'lightpink', 'plum']
colors = ['lightgreen', 'aquamarine', 'lightblue', 'lightyellow', 'yellow', 'orange', 'pink']
mylabels = 'A', 'B', 'C', 'D', 'E', 'F', 'G'
sizes = [my_data['A'], my_data['B'], my_data['C'], my_data['D'], my_data['E'], my_data['F'], my_data['G']]
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
#plt.pie(sizes, labels=mylabels, shadow=True, colors=colors, explode=(0, 0), startangle=90, autopct='%1.1f%',)
fig1, ax1 = plt.subplots()
patches, texts, autotexts = ax1.pie(sizes, explode=(0,0,0,0,0,0,0), labels=mylabels, autopct='%1.0f%%',
        shadow=False, startangle=90, colors=colors)
for texts in texts:
    texts.set_color('grey')
for autotexts in autotexts:
    autotexts.set_color('grey')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()
###########################################################################
#Basic Data Analysis by Groups

#Lending Club Average Default Risk by Credit Risk Grade
print('Probability of Default by Credit Risk Grade:')
print(df.groupby('grade')['Defaulted'].mean())
print()
#Data Visualization
defaultbygrade_list = df.groupby('grade')['Defaulted'].mean().tolist()
dbg_list = [round(num, 2) for num in defaultbygrade_list]
men_means = dbg_list
whole_list = [1, 1, 1, 1, 1, 1, 1]
women_means = np.subtract(whole_list, dbg_list)
ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, men_means, width, color='#ff9999',
                label='Default')
rects2 = ax.bar(ind + width/2, women_means, width, color='#99ff99',
                label='No default')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probability')
ax.set_title('Probability of Default by Credit Risk Score Grade')
ax.set_xticks(ind)
ax.set_xticklabels(('A', 'B', 'C', 'D', 'E', 'F', 'G'))
ax.legend()
def autolabel(rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')
autolabel(rects1, "left")
autolabel(rects2, "right")
fig.tight_layout()
plt.ylim(0, 1)
plt.show()
#############################################################################



#Debt-to-Income Ratio Summary Statistics by Default
print('Debt-to-Income Ratio Summary Statistics by Defaulted')
print(df.groupby('Defaulted')['dti'].describe())
print()

#Data Vizualization
#Group dti by Defaulted
#yes
dti_yes = df.loc[df['Defaulted'] == 1, 'dti']
dti_yes = pd.DataFrame(data=dti_yes)
dti_yes = dti_yes.dti.tolist()
dti_yes = np.asarray(dti_yes)
#no
dti_no = df.loc[df['Defaulted'] == 0, 'dti']
dti_no = pd.DataFrame(data=dti_no)
dti_no = dti_no.dti.tolist()
dti_no = np.asarray(dti_no)
labels = ['Defaulted', 'Did Not Default']
labelA, labelB = 'Debt-to-Income Ratios without outliers', 'Debt-to-Income Ratios with outliers'
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# box plot
bplot1 = ax1.boxplot([dti_yes, dti_no],
                     vert=True,  # vertical box alignment
                     patch_artist=True,
                     labels = labels,
                     showfliers=False)  # will be used to label x-ticks
ax1.set_title(labelA)
# box plot
bplot2 = ax2.boxplot([dti_yes, dti_no],
                     vert=True,  # vertical box alignment
                     patch_artist=True,
                     labels = labels,
                     showfliers=True)  # will be used to label x-ticks
ax2.set_title(labelB)
# fill with colors
colors = ['pink', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
# adding horizontal grid lines
for ax in [ax1, ax2]:
    ax.yaxis.grid(True)
    ax.set_ylabel('Debt-to-Income Ratio')
plt.show()



#Debt-to-Income Ratio Summary Statistics by Grade
print('Debt-to-Income Ratio Summary Statistics by Credit Risk Grade')
print(df.groupby('grade')['dti'].describe())
print()
#Data Vizualization
#Group dti by Grade
#A
dti_A = df.loc[df['grade'] == 'A', 'dti']
dti_A = pd.DataFrame(data=dti_A)
dti_A = dti_A.dti.tolist()
dti_A = np.asarray(dti_A)
#B
dti_B = df.loc[df['grade'] == 'B', 'dti']
dti_B = pd.DataFrame(data=dti_B)
dti_B = dti_B.dti.tolist()
dti_B = np.asarray(dti_B)
#C
dti_C = df.loc[df['grade'] == 'C', 'dti']
dti_C = pd.DataFrame(data=dti_C)
dti_C = dti_C.dti.tolist()
dti_C = np.asarray(dti_C)
#D
dti_D = df.loc[df['grade'] == 'D', 'dti']
dti_D = pd.DataFrame(data=dti_D)
dti_D = dti_D.dti.tolist()
dti_D = np.asarray(dti_D)
#E
dti_E = df.loc[df['grade'] == 'E', 'dti']
dti_E = pd.DataFrame(data=dti_E)
dti_E = dti_E.dti.tolist()
dti_E = np.asarray(dti_E)
#F
dti_F = df.loc[df['grade'] == 'F', 'dti']
dti_F = pd.DataFrame(data=dti_F)
dti_F = dti_F.dti.tolist()
dti_F = np.asarray(dti_F)
#G
dti_G = df.loc[df['grade'] == 'G', 'dti']
dti_G = pd.DataFrame(data=dti_G)
dti_G = dti_G.dti.tolist()
dti_G = np.asarray(dti_G)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
labelA, labelB = 'Debt-to-Income Ratios without outliers', 'Debt-to-Income Ratios with outliers'
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# box plot
bplot1 = ax1.boxplot([dti_A, dti_B, dti_C, dti_D, dti_E, dti_F, dti_G],
                     vert=True,  # vertical box alignment
                     patch_artist=True,
                     labels = labels,
                     showfliers=False)  # will be used to label x-ticks
ax1.set_title(labelA)
# box plot
bplot2 = ax2.boxplot([dti_A, dti_B, dti_C, dti_D, dti_E, dti_F, dti_G],
                     vert=True,  # vertical box alignment
                     patch_artist=True,
                     labels = labels,
                     showfliers=True)  # will be used to label x-ticks
ax2.set_title(labelB)
# fill with colors
colors = ['lightgreen', 'aquamarine', 'lightblue', 'lightyellow', 'yellow', 'orange', 'pink']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
# adding horizontal grid lines
for ax in [ax1, ax2]:
    ax.yaxis.grid(True)
    ax.set_ylabel('Debt-to-Income Ratio')
plt.show()
####################################################################################



#################
#Statistical Tests:
################

#############
#Chi Squared:
# Defaulted vs Grade chi square test
#
print('\n'*5)
print("Chi Squared: Defaulted vs Grade chi square test")
crosstab = pd.crosstab(df['Defaulted'], df['grade'])
print(stats.chi2_contingency(crosstab))

#############
#Z Test:
# Defaulted vs D-to-I z-test
#
print('\n'*5)

print(np.std(dti_yes))
print(np.std(dti_no))
print()
print("Z-Test: Defaulted vs D-to-I z-test:")
print(sm.stats.ztest(dti_yes, dti_no))



print('\n'*10)
###############################################################################
#############################################################################
###########################################################################
#THE MODEL
#
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
#######################################################################################
#SVM
#
##########################################################################################
x = df[['dti', 'A', 'B', 'C', 'D', 'E', 'F', 'G']]
y = df.Defaulted


accuracy_list = []
svm_list = []

print("Accuracies of 9 SVM models:")
for i in range(0, 9):
    print(i)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.999)
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, predictions)
    print(acc)
    accuracy_list.append(acc)
    svm_list.append(clf)

print()
print(accuracy_list)
print()
print(svm_list)
print()
acc_df = pd.DataFrame(accuracy_list)
print(acc_df)
print()
print(acc_df.describe())
print()
print(acc_df.median())

print()
mdn_acc = np.median(np.asarray(accuracy_list))
print(mdn_acc)
print()
for i in range(0, 9):
    if accuracy_list[i] == mdn_acc:
        model_number = i

print()
print(accuracy_list[model_number])
print(model_number)
print(svm_list[model_number])
PD_SVM = svm_list[model_number]
print('\n')
print("model we will save:")
print(PD_SVM)



#Save the model!!!!! (note: make sure to enter the full file path of where you want it saved: )
with open(r"pd_svm_model.pickle", "wb") as f:
    pickle.dump(PD_SVM, f)
    
