import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import r2_score


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
    
    df.dtypes
    df.isna().sum()
           
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
    df.Tire_Size.value_counts()
    
    under_carriage_new = df.Undercarriage_Pad_Width.str.split().str[0].replace("None", np.nan).astype(float)  
    df.Undercarriage_Pad_Width = under_carriage_new
    df.Undercarriage_Pad_Width.value_counts()
    

    
    stick_length_new = df.Stick_Length.replace("None or Unspecified", np.nan)
    stick_length_feet = stick_length_new.str.split("'").str[0]
    stick_length_feet = stick_length_feet.astype(float)
    
    
    stick_length_inches=stick_length_new.str.split().str[1].str.split('"').str[0]
    stick_length_inches=stick_length_inches.astype(float)
    
    stick_length_new = (stick_length_feet * 12) + stick_length_inches
    stick_length_new.value_counts()
    
    df.Stick_Length = stick_length_new
    
    df.Stick_Length.value_counts()
    
    
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


#44 cat, 13 numerical, 44 -> 5 ordinal, 3 numerical

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

#preprocessing datasets
df_train = preprocessing2(df_train)
df_valid = preprocessing2(df_valid)#


corr_train = df_train.corr()
corr_valid = df_valid.corr()#since only 1 value for both backhoemounting and salesyear there is no corr for them

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
y_train = df_train["SalePrice"] #natural logarithmic of target var. can be discussed to improve model's accuracy

X_valid = df_valid.drop("SalePrice", axis = 1)
y_valid = df_valid["SalePrice"] #natural logarithmic of target var. can be discussed to improve model's accuracy



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



### evaluation

from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_preds))
        

feature_importances = model.feature_importances_


features_dict = dict(zip(X_train.columns, list(feature_importances)))
features_dict



df_importance = pd.DataFrame(features_dict, index=[0])
df_importance = df_importance.sort_values(by=[0], axis=1, ascending=False)
df_importance.iloc[:,:21].T.plot.barh(title="Feature Importances", legend=False)


