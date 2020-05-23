# Probability_of_Default-SVM
Made a probability of default calculator for peer-to-peer lending by training a support vector machine algorithm on over 1300 peer-to-peer loans.
The independent variables used were credit scores and debt-to-income ratios. When tested on 1.3 million loans, it was 80% accurate.

The goal of the project was to make a calculator that estimates the probability of default for a peer-to-peer loan.
The independent variables used in the estimation are the borrower's credit score, and the borrower's debt-to-income ratio.
A support vector machine (SVM) model was trained on over 1300 peer-to-peer loans.
After it was trained, it was then tested on over 1.3 million loans. It was 80% accurate when tested.

The PD_calculator.py is the probability of default calculator.

The pickle file is the SVM model. The PD_calculator needs the pickle file, as the calulator loads the pickle file.

The .docx file is both a summary of my findings analyzing peer-to-peer loans, and a brief description of the SVM model's testing, training, and accuracy (80%).
The .docx file is filled with data visualizations and statistical tables.
These visualizations and tables alone are very useful in estimating the probabilities of default for peer-to-peer loans with different credit scores and debt-to-income ratios.

The PD_project file creates, trains and tests the SVM model and then saves it as the pickle file. 
PD_project also produces the data visualization and statistical tables found in the .docx report.


NOTE: The loan data csv file easily exceeds GitHubs file size limit, as it contains over 1.3 million rows and over 100 columns.
However, you can download the file here: https://www.kaggle.com/wendykan/lending-club-loan-data?select=loan.csv
