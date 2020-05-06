import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sns.set()

#predicting bulldozer price


df = pd.read_csv("TrainAndValid.csv", 
                 low_memory = False)

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

def plotting(df):
    fig, ax = plt.subplots()
    ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000])
    plt.show(fig)
    plt.close(fig)
    
    df.SalePrice.hist()



def preprocessing(df):
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


df_train = preprocessing(df_train)
df_valid = preprocessing(df_valid)


#preprocessing datasets
'''
df_train = preprocessing(df_train)
df_valid = preprocessing(df_valid)
'''


def correlationPlot(df):
    corr = df.corr()
    sns.set(font_scale=1)
    plt.figure(figsize=(30,30))
    a = sns.heatmap(corr, annot = True, fmt = '.2f', square = True)
    plt.show()
    plt.close()
 
    '''
correlationPlot(df_train)
'''


### Train - Valid Split

X_train = df_train.drop("SalePrice", axis = 1)
y_train = df_train["SalePrice"]

X_valid = df_valid.drop("SalePrice", axis = 1)
y_valid = df_valid["SalePrice"]


# Scaling
'''
def scale(arr):
    sc = StandardScaler()
    arr = sc.fit_transform(arr)
    return arr

scaled_train = scale(X_train)
scaled_valid = scale(X_valid)

#Ridge

from sklearn.linear_model import Ridge

model = Ridge()

model.fit(scaled_train, y_train)

y_preds = model.predict(scaled_valid)

model.score(scaled_valid, y_valid)
    
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

'''
grid = {"n_estimators": [40,100,150],
        "min_samples_leaf":[25, 35, 40],
        "min_samples_split":[4,8,16],
        "max_features":["auto", "sqrt"],
        "max_samples":[None,0.7]}
'''
model = RandomForestRegressor(n_estimators=40,
                                   min_samples_leaf=1,
                                   min_samples_split=14,
                                   n_jobs=-1,
                                   max_samples=None,
                                   random_state=42)


'''
rs_model = RandomizedSearchCV(estimator=model,
                      param_distributions=grid,
                      n_iter=5,
                      cv=5,#5 fold cross-validation
                      verbose=2)
'''


model.fit(X_train, y_train)
y_preds = model.predict(X_valid)
model.score(X_train, y_train)
model.score(X_valid, y_valid)

### evaluation

from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_preds))
        

