
# Predicting the Sale Price of Bulldozers using ML

## Problem definition

Predict the auction sale price for a piece of heavy equipment to create a "blue book" for bulldozers.

## Data

> The data is downloaded from kaggle.com "Blue Book for Bulldozers competition.
The data for this competition is split into three parts:

Train.csv is the training set, which contains data through the end of 2011.  
Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.  
Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.

> https://www.kaggle.com/c/bluebook-for-bulldozers/overview

## Evaluation

The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

## Features

Features are shown here in a spreadsheet:  
https://docs.google.com/spreadsheets/d/1P0vZ7VYhBpJc-4-4mM4nvYoTzgnr-AF1F5eIMvcVd7Y/edit?usp=sharing # # #

> *It's a finished competition.*

* Best RMSLE score on validation data from Kaggle public leaderboard is :
*0.22909*
* My best RMSLE score on validation data for now is :
*0.24457*

Since there were a lot of categorical values, categorical features were encoded with Label Encoding rather than one hot.

## To-DO List
- [x] OneHotEncoding and LabelEncoding separately
- [x] Transform some categorical feature values into numerical values
- [x] Feature Selection and Extraction
- [ ] Try XGBoost also
- [x] More visualization to get an emphasis on data

One Hot Encoding won't be used due to large number of feature columns. 


> *In bulldozerv2, some changes were made.*  

 Some "Object" type features as "Tire Size", "Undercarriage Pad Width", "Stick Length" were transformed into numerical features.  
 Some ordinal features as "Usage Band", "Product Size", "Blade Width", "Enclosure Type", "Grouser Type" were mapped by meaningful values like "Low" = 1, "Medium" = 2, "High" = 3.
 
 After changes were made RMSLE is changed to *0.24592*. Thus, we can tell there are no improvements, RMSLE has suffered from this changes.
 
 Correlation of Numerical Features before changes were made:  
 
 <img src="img/Figure_1.png" width="40%">
 
 
 Frequency of Sale Prices:  
 
 <img src="img/Figure_3.png" width="40%">
 
   
 Distribution of Sale Prices due to Sale Year:  
 
 <img src="img/Figure_4.png" width="40%">
 
   
 Product Size versus Sale Prices:  
  
 <img src="img/Figure_5.png" width="40%">
 
 
 Product Size versus Tire Sizes (Train Set on the left and Valid Set on the right hand side):  
 
 <img src="img/Figure_6.png" alt="Train Set" width="40%">                           <img src="img/Figure_7.png" alt="Valid Set" width="40%">  

 
 
 Blade Width versus Tire Sizes (Train Set on the left and Valid Set on the right hand side):  
 
 <img src="img/Figure_8.png" alt="Train Set" width="40%">                           <img src="img/Figure_9.png" alt="Valid Set" width="40%"> 
   
 
 ## Feature Selection
 
 > 2 procedures were followed while extracting features;  
 - Manually deleting unimportant features  
 - Backward Elimination method
 
 In Backward Elimination, features which p values' are higher than significance level (0.05) were deleted. 2 features were deleted through this procedure and the scores afterwards:  
 Model's performance on validation data after backwards elimination 0.87622
 Model's performance on train data after backwards elimination 0.92786
 RMSLE:0.24485
 
 About the other procedure, manual procedure, After fitting Random Forest Regressor, feature importances were 
 
 
 
