import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sns.set()

df = pd.read_csv("TrainAndValid.csv", 
                     low_memory = False)



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
    
    '''
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content):
            print(df[label].value_counts())
            print()
            time.sleep(3)
    '''
    #converting string types features to category type
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content):
            df[label] = content.astype("category").cat.as_ordered()
    
    return df

df.dtypes
df = preprocessing1(df)



def plotting(df):
    fig, ax = plt.subplots()
    ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000])
    plt.show(fig)
    plt.close(fig)
    
    df.SalePrice.hist()



def preprocessing2(df):
#Imputation and converting cat to numeric values
     
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if df[label].isna().sum():
                
                imputer = SimpleImputer(missing_values=np.nan, strategy = "median")
                imputer = imputer.fit(df[label].values.reshape(-1,1))
                df[label] = imputer.transform(df[label].values.reshape(-1,1))
                
        else:
                            
                df[label] = pd.Categorical(content).codes + 1
                
    return df

#splitting train and valid datasets
df_train = df[df["saleYear"]!=2012]
df_valid = df[df["saleYear"]==2012]

#preprocessing datasets
df_train = preprocessing2(df_train)
df_valid = preprocessing2(df_valid)


def correlationPlot(df):
    corr = df.corr()
    sns.set(font_scale=1)
    plt.figure(figsize=(30,30))
    sns.heatmap(corr, annot = True, fmt = '.2f', square = True)
    plt.show()
    plt.close()
 

### Train - Valid Split

X_train = df_train.drop("SalePrice", axis = 1)
y_train = df_train["SalePrice"]

X_valid = df_valid.drop("SalePrice", axis = 1)
y_valid = df_valid["SalePrice"]


from sklearn.ensemble import RandomForestRegressor
#n_estimators doesn't cause overfitting or underfitting more estimators more stable and powerful model
#min_samples_leaf A smaller leaf makes the model more prone to capturing noise in train data. -> overfitting


model = RandomForestRegressor(n_estimators=100,
                                   min_samples_leaf=10,
                                   n_jobs=-1,
                                   max_features=0.5)
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
        

### test data


df_test = pd.read_csv("Test.csv", 
                 low_memory = False)

df_test = preprocessing1(df_test)
df_test = preprocessing2(df_test)


test_pred = model.predict(df_test)

column_dict = {"SalesID": df_test.SalesID,
               "SalesPrice": test_pred}

final_df = pd.DataFrame(column_dict)

