# bluebook-for-bulldozers

https://www.kaggle.com/c/bluebook-for-bulldozers/overview

> *It's a finished competition.*

* Best RMSLE score on validation data from Kaggle public leaderboard is :
*0.22909*
* My RMSLE score on validation data for now is :
*0.24547*

Since there were a lot of categorical values, categorical features were encoded with Label Encoding rather than one hot.

## To-DO List
- [x] OneHotEncoding and LabelEncoding separately
- [x] Transform some categorical feature values into numerical values
- [ ] Try XGBoost also
- [x] More visualization to get an emphasis on data

One Hot Encoding won't be used due to large number of feature columns. 


> *In bulldozer_v2, some changes were made.*  

 Some "Object" type features as "Tire Size", "Undercarriage Pad Width", "Stick Length" were transformed into numerical features.  
 Some ordinal features as "Usage Band", "Product Size", "Blade Width", "Enclosure Type", "Grouser Type" were mapped by meaningful values like "Low" = 1, "Medium" = 2, "High" = 3.
 
 Correlation of Numerical Features before changes were made:  
 
 ![Correlation Matrix](/img/Figure_1.png)
