import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import statsmodels.api as sm




df = pd.read_csv("TrainAndValid.csv", 
                     low_memory = False)


corr_base = df.corr()

plt.figure(dpi=300)
sns.set(font_scale=0.5)
sns.heatmap(corr_base, fmt=".2f", square=True, annot=True, annot_kws={"size": 5})
plt.show()


    
def preprocessing1(df):
    #predicting bulldozer price
    
    df["saledate"] = pd.to_datetime(df["saledate"]) # parsing dates
    df.sort_values(by=["saledate"], inplace = True, ascending=True)
         
        
    #adding some new features derived from saledate
    
    df["saleYear"] = df["saledate"].dt.year
    df["saleMonth"] = df["saledate"].dt.month
    df["saleDay"] = df["saledate"].dt.day
    df["saleDayOfWeek"] = df["saledate"].dt.dayofweek
    df["saleWeekOfYear"] = df["saledate"].dt.weekofyear
    
    df.drop("saledate", axis=1, inplace=True)
        
    
    usageband_dict = {"Medium" : 2,
                  "Low" : 1,
                  "High" : 3,
                  }
    productsize_dict = {"Small" : 2,
                    "Mini" : 1,
                    "Compact" : 3,
                    "Medium" : 4,
                    "Large / Medium" : 5,
                    "Large" : 6,
                    }
    bladewidth_dict = {"<12'" : 1,
                    "12'" : 2,
                    "13'" : 3,
                    "14'" : 4,
                    "16'" : 5,
                    "None or Unspecified" : 0,
                    }
    enclosuretype_dict = {"High Profile" : 2,
                      "Low Profile" : 1,
                      "None or Unspecified" : 0,
                      }
    grousertype_dict =   {"Double" : 2,
                      "Single" : 1,
                      "Triple" : 3,
                      }
    
    df.UsageBand = df.UsageBand.map(usageband_dict).fillna(0)#ordinal
    df.ProductSize=df.ProductSize.map(productsize_dict).fillna(0) #ordinal
    df.Blade_Width=df.Blade_Width.map(bladewidth_dict).fillna(0) #ordinal
    df.Enclosure_Type=df.Enclosure_Type.map(enclosuretype_dict).fillna(0) #ordinal
    df.Grouser_Type=df.Grouser_Type.map(grousertype_dict).fillna(0) #ordinal
    
    
    sel_ord_cols = ["UsageBand", "ProductSize", "Blade_Width", "Enclosure_Type", "Grouser_Type"]
    
    
    df.Tire_Size.value_counts()#numerical #10 inch will be removed, " tags will be removed
    df.Undercarriage_Pad_Width.value_counts() #numerical #inch will be removed
    df.Stick_Length.value_counts() #numerical #feet to inch -> 12*feet = new_inch value = new_inch + inch
    
    tire_size_new = df.Tire_Size.str.split('"|\s').str[0].replace("None", np.nan).astype(float)  
    df.Tire_Size = tire_size_new
    
    under_carriage_new = df.Undercarriage_Pad_Width.str.split().str[0].replace("None", np.nan).astype(float)  
    df.Undercarriage_Pad_Width = under_carriage_new
    

    
    stick_length_new = df.Stick_Length.replace("None or Unspecified", np.nan)
    stick_length_feet = stick_length_new.str.split("'").str[0]
    stick_length_feet = stick_length_feet.astype(float)
    
    
    stick_length_inches=stick_length_new.str.split().str[1].str.split('"').str[0]
    stick_length_inches=stick_length_inches.astype(float)
    
    stick_length_new = (stick_length_feet * 12) + stick_length_inches
    
    df.Stick_Length = stick_length_new

    
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content):
            df[label] = content.astype("category").cat.as_ordered()   
            
    
    return df


def preprocessing2(df):
#Imputation and converting cat to numeric values
     
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if df[label].isna().sum() > 0:
                
                imputer = SimpleImputer(missing_values=np.nan, strategy = "median")
                imputer = imputer.fit(df[label].values.reshape(-1,1))
                df[label] = imputer.transform(df[label].values.reshape(-1,1))
                
        else:
                            
                df[label] = pd.Categorical(content).codes + 1
             
    return df


df = preprocessing1(df)

corr_next = df.corr()


plt.figure(dpi=300)
sns.set(font_scale=0.5)
sns.heatmap(corr_next, fmt=".1f", square=True, annot=True, annot_kws={"size": 5})
plt.show()

plt.figure(dpi=300)
plt.hist(df.SalePrice,bins=45,color="darkorange") #frequency of saleprices
plt.show()

plt.figure(dpi=300)
ax = sns.boxplot(x = "saleYear", y = "SalePrice", data = df)
plt.show()

plt.figure(dpi=300)
ax = sns.boxplot(x = "ProductSize", y = "SalePrice", data = df)
plt.show()


#splitting train and valid datasets
df_train = df[df["saleYear"]!=2012]
df_valid = df[df["saleYear"]==2012]

df_train.saleYear.value_counts()
df_valid.saleYear.value_counts()




#preprocessing datasets
df_train = preprocessing2(df_train)
df_valid = preprocessing2(df_valid)#


corr_train = df_train.corr()
corr_valid = df_valid.corr()#since only 1 value for both backhoemounting and salesyear there is no corr. for them

corr_train_high = corr_valid[corr_valid > .4]
corr_valid_high = corr_valid[corr_valid > .4]


#box plot
for df_temp in [df_train,df_valid]: #box plot for both train and valid sets
    plt.figure(dpi=300) #20.5
    ax = sns.boxplot(x = "ProductSize", y = "Tire_Size", data = df_temp)
    plt.show()

    plt.figure(dpi=300)
    ax = sns.boxplot(x = "Blade_Width", y = "Tire_Size", data = df_temp)
    plt.show()
    
    

### Train - Valid Split

X_train = df_train.drop("SalePrice", axis = 1)
y_train = df_train["SalePrice"] #natural logarithmic of target var. can be discussed

X_valid = df_valid.drop("SalePrice", axis = 1)
y_valid = df_valid["SalePrice"] #natural logarithmic of target var. can be discussed


from sklearn.ensemble import RandomForestRegressor
#n_estimators doesn't cause overfitting or underfitting more estimators more stable and powerful model
#min_samples_leaf A smaller leaf makes the model more prone to capturing noise in train data. -> overfitting


model = RandomForestRegressor(n_estimators=100,
                              min_samples_leaf=10,
                              n_jobs=-1,
                              max_features=0.5,
                              random_state=0)
#max_depth:Longest path between the root node and the leaf node, very high max_depth can cause overfitting
#min_samples_split: minimum required number of observations in any given node in order to split it. too low values can cause overfitting
###too high can cause underfit though.



model.fit(X_train, y_train)
y_preds = model.predict(X_valid)
model.score(X_train, y_train)
model.score(X_valid, y_valid)

print(f"Model's performance on validation data before feature extraction {model.score(X_valid, y_valid)}")
print(f"Model's performance on train data before feature extraction {model.score(X_train, y_train)}")

'''
Model's performance on validation data before feature extraction 0.8761005065934953
Model's performance on train data before feature extraction 0.9294303560425823
rmsle:0.24592169141754905
'''


### evaluation

from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_preds))
        

feature_importances = model.feature_importances_


features_dict = dict(zip(X_train.columns, list(feature_importances)))
features_dict



df_importance = pd.DataFrame(features_dict, index=[0])
df_importance = df_importance.sort_values(by=[0], axis=1, ascending=False)
df_importance.iloc[:,:21].T.plot.barh(title="Feature Importances", legend=False)


remove_index = list(df_importance.columns).index("Grouser_Tracks") #old -> Grouser_Tracks, Ripper, Thumb

'''
thumb and leftmost features deleted

Model's performance on validation data after feature extraction 0.8771164153081379
Model's performance on train data after feature extraction 0.9291259385098817

rmsle:0.24555071476572543
'''
'''
ripper and leftmost features deleted
Model's performance on validation data after feature extraction 0.8762069338563853
Model's performance on train data after feature extraction 0.922278069942806
rmsle:0.24457256667477129
'''
'''
grouser and leftmost features deleted
Model's performance on validation data after feature extraction 0.8736448109456616
Model's performance on train data after feature extraction 0.9156582692510319
rmsle:0.24460054865717812
'''


new_columns = list(df_importance.columns)[:remove_index]


new_X_train = X_train[new_columns]
assert list(new_X_train.columns) == new_columns
new_X_valid = X_valid[new_columns]
assert list(new_X_valid.columns) == new_columns

model.fit(new_X_train, y_train)
y_preds = model.predict(new_X_valid)
model.score(new_X_train, y_train)
model.score(new_X_valid, y_valid)


rmsle_new = np.sqrt(mean_squared_log_error(y_valid, y_preds)) 


print(f"Model's performance on validation data after feature extraction {model.score(new_X_valid, y_valid)}")
print(f"Model's performance on train data after feature extraction {model.score(new_X_train, y_train)}")

#backward elimination

X_opt_Train = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)
X_opt_Valid = np.append (arr=np.ones([X_valid.shape[0],1]).astype(int), values = X_valid, axis = 1)

sl = 0.05
regressor_ols = sm.OLS(y_train, X_opt_Train).fit()
print(regressor_ols.summary())
X_opt_Train = np.delete(X_opt_Train,56,1)
regressor_ols = sm.OLS(y_train, X_opt_Train).fit()
print(regressor_ols.summary())
X_opt_Train = np.delete(X_opt_Train,55,1)
regressor_ols = sm.OLS(y_train, X_opt_Train).fit()
print(regressor_ols.summary())

X_opt_Valid = np.delete(X_opt_Valid,56,1)
X_opt_Valid = np.delete(X_opt_Valid,55,1)

model.fit(X_opt_Train, y_train)
y_preds = model.predict(X_opt_Valid)
model.score(X_opt_Train, y_train)
model.score(X_opt_Valid, y_valid)


print(f"Model's performance on validation data after backwards elimination {model.score(X_opt_Valid, y_valid)}")
print(f"Model's performance on train data after backwards elimination {model.score(X_opt_Train, y_train)}")

rmsle_backwards = np.sqrt(mean_squared_log_error(y_valid, y_preds)) 

'''
Model's performance on validation data after backwards elimination 0.8762201635814332
Model's performance on train data after backwards elimination 0.9278648009228353
rmsle:0.24484635702900687
'''


'''


### test data


df_test = pd.read_csv("Test.csv", 
                 low_memory = False)

df_test = preprocessing1(df_test)
df_test = preprocessing2(df_test)


test_pred = model.predict(df_test)

column_dict = {"SalesID": df_test.SalesID,
               "SalesPrice": test_pred}

final_df = pd.DataFrame(column_dict)
'''

