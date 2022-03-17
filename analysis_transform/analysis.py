import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import date
import pickle

# plotting continuos variables
def plot_continous(df):
    for item in df.columns:
        sns.displot(x=item, data = df, kde=True)
    plt.show()
    
# plotting discrete variables    
def plot_discrete(df):
    r = math.ceil(df.shape[1]/2)
    c = 2
    fig, ax = plt.subplots(r,c, figsize=(15,40))
    i = 0
    j = 0
    for item in df.columns:
        sns.histplot(x=item, data = df, ax = ax[i, j])
        if j == 0:
            j = 1
        elif j == 1:
            j = 0
            i = i + 1
    plt.show()
    
# plotting boxplot    
def boxplot_continous(df):
    r = math.ceil(df.shape[1]/2)
    c = 2
    fig, ax = plt.subplots(r,c, figsize=(15,20))
    i = 0
    j = 0

    for item in df.columns:
        sns.boxplot(x=item, data=df, ax=ax[i, j])
        if j == 0:
            j = 1
        elif j == 1:
            j = 0
            i = i + 1
    plt.show()    
    
# transform variables and plot the transformed variables
def plot_transformer(df): 
    data_log = pd.DataFrame()
    data_log = log_it(df)
    data_bc, data_yj = power_transform(df)
    r = df.shape[1]
    c = 4
    fig, ax = plt.subplots(r, c, figsize=(30,30))
    i = 0
    data = ""
    for item in df.columns:
        for j in range(c):
            if j == 0:
                data = df
                head = "original"
            elif j == 1:
                data = data_log
                head = "log"
            elif j == 2:
                data = data_yj
                head = "yeo-johnson"
            elif j == 3:
                data = data_bc
                head = "box-cox"
            ax[0,j].set_title(head)
         
            if item in data.columns:
                sns.distplot(a = data[item], ax = ax[i, j]) 
        i = i + 1
    plt.show()
    
# perform log transform        
def log_it(df):
    data_log = pd.DataFrame()
    for item in df.columns:
        data_log[item] = df[item].apply(__log_transform_clean)
    return data_log

def __log_transform_clean(x):
    if np.isfinite(x) and x!=0:
        return np.log(x)
    else:
        return np.NAN
    
def __df_box_cox(df):
    df1 = pd.DataFrame()
    for item in df.columns:
        if df[item].min() > 0:
            df1[item] = df[item]
    return df1

#perform power transform
def power_transform(df):
    df_f_bc = __df_box_cox(df)
    pt_bc = PowerTransformer(method="box-cox")
    pt_bc.fit(df_f_bc)
    df_bc = pd.DataFrame(pt_bc.transform(df_f_bc), columns = df_f_bc.columns)
    
    pt_yj = PowerTransformer()
    pt_yj.fit(df)
    df_yj = pd.DataFrame(pt_yj.transform(df), columns = df.columns)
    
    return df_bc, df_yj

# remove outliers
def remove_outliers(df_num, df_cat):
    df = pd.concat([df_num, df_cat], axis=1)
    display(df)
    old_rows = df.shape[0]
    for item in df_num.columns:
        iqr = np.nanpercentile(df[item],75) - np.nanpercentile(df[item],25)
        upper_limit = np.nanpercentile(df[item],75) + 1.5*iqr
        lower_limit = np.nanpercentile(df[item],25) - 1.5*iqr
        df = df[(df[item] < upper_limit) & (df[item] > lower_limit)]
        
    rows_removed = old_rows - df.shape[0]
    rows_removed_percent = (rows_removed/old_rows)*100
        
    print("{} rows have been removed, {}% in total".format(rows_removed, rows_removed_percent))
    df_continous = df[["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15", "price"]]
    df_discrete = df.drop(columns=df_continous.columns)
  
    return df_continous, df_discrete

# checks for unique values
def unique(df):
    for item in df.columns:
        print(item)
        print(df[item].unique())
        print("---------------")

# builds model and saves it        
def regression_automation(X_test, y_test, filename, X_train = None, y_train = None, train = True):
    if train:
        knn_models = __search_k(X_train, y_train, X_test, y_test)
        var = int(input("Please enter k:"))
        files = []
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        filename_lr = filename + "_linear.sav"
        pickle.dump(lr, open("models/"+filename_lr, 'wb'))
        print("-----------------------------")
        print("------Linear Regression------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("-----------------------------")
        print("------Linear Regression------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        print("Filename Linear: " + filename_lr)
        files.append(filename_lr)
        
        knn_models[var-2].score(X_test, y_test)
        y_pred_train = knn_models[var-2].predict(X_train)
        y_pred_test = knn_models[var-2].predict(X_test)
        filename_knn = filename + "_knn.sav"
        pickle.dump(knn_models[var-2], open("models/"+filename_knn, 'wb'))
        print("--------------KNN------------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("Filename knn: " + filename_knn)
        print("-----------------------------")
        print("--------------KNN------------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        files.append(filename_knn)
        
        return files
    if train == False:
        loaded_linear = pickle.load(open("models/"+filename[0], 'rb'))
        y_pred = loaded_linear.predict(X_test)
        print("------Linear Regression------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred))
        print("MSE:",mean_squared_error(y_test , y_pred))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred)))
        print("MAE:",mean_absolute_error(y_test , y_pred))
        print("-----------------------------")
        
        loaded_knn = pickle.load(open("models/"+filename[1], 'rb'))
        y_pred1 = loaded_knn.predict(X_test)
        print("--------------KNN------------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred1))
        print("MSE:",mean_squared_error(y_test , y_pred1))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred1)))
        print("MAE:",mean_absolute_error(y_test , y_pred1))
        print("-----------------------------")
        
        
def __search_k(X_train, y_train, X_test, y_test):
    knn_models = []
    scores = []
    for k in range(2,15):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        knn_models.append(model)
        scores.append(model.score(X_test, y_test))
    plt.figure(figsize=(10,6))
    plt.plot(range(2,15),scores,color = 'blue', linestyle='dashed',
    marker='o', markerfacecolor='red', markersize=10)
    plt.title('R2-scores vs. K Value')
    plt.xticks(range(1,16))
    plt.gca().invert_yaxis()
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()
    return knn_models

def min_max(X,  filename, fit = True):
    X_cont = X[["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15"]]
    X_disc = X.drop(columns=X_cont).reset_index(drop=True)
    if fit:
        scaler = MinMaxScaler()
        scaler.fit(X_cont)
        filename = filename + ".sav"
        pickle.dump(scaler, open("scaler/"+filename, 'wb'))
        X_scaled_cont = scaler.transform(X_cont)
        X_scaled_cont_df = pd.DataFrame(X_scaled_cont, columns=X_cont.columns)
        X_scaled_df = pd.concat([X_scaled_cont_df, X_disc], axis=1)
        return X_scaled_df, filename
    if fit == False:
        loaded_model = pickle.load(open("scaler/"+filename, 'rb'))
        X_scaled_cont = loaded_model.transform(X_cont)
        X_scaled_cont_df = pd.DataFrame(X_scaled_cont, columns=X_cont.columns)
        X_scaled_df = pd.concat([X_scaled_cont_df, X_disc], axis=1)
        return X_scaled_df
    
def one_hot(X, filename, y=None, fit = True):
    X = X.copy()
    X["bedrooms"] = X["bedrooms"].apply(__bedrooms)
    X["bedrooms"] = X["bedrooms"].astype(object)
    X["bathrooms"] = X["bathrooms"].apply(__bathrooms)
    X["bathrooms"] = X["bathrooms"].astype(object)
    X["floors"] = X["floors"].astype(object)
    if fit:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
        X_num_train = X_train.select_dtypes(np.number).reset_index(drop=True)
        X_cat_train = X_train.select_dtypes(np.object)
        X_num_test = X_test.select_dtypes(np.number).reset_index(drop=True)
        X_cat_test = X_test.select_dtypes(np.object)
        X = X.select_dtypes(np.object)
        col_list = [X[col].unique() for col in X.columns]
        encoder = OneHotEncoder(handle_unknown='error', drop='first', categories=col_list)
        encoder.fit(X_cat_train)
        filename = filename + ".sav"
        pickle.dump(encoder, open("encoder/"+filename, 'wb'))
        categoricals_encoded_train = encoder.transform(X_cat_train).toarray()
        categoricals_encoded_test = encoder.transform(X_cat_test).toarray()
        categoricals_encoded_train_df = pd.DataFrame(categoricals_encoded_train, columns = encoder.get_feature_names_out())
        categoricals_encoded_test_df = pd.DataFrame(categoricals_encoded_test, columns = encoder.get_feature_names_out())
        df_train_onehot = pd.concat([categoricals_encoded_train_df, X_num_train], axis=1)
        df_test_onehot = pd.concat([categoricals_encoded_test_df, X_num_test], axis=1)
        return df_train_onehot, df_test_onehot, y_train, y_test, filename
    if fit == False:
        loaded_encoder = pickle.load(open("encoder/"+filename, 'rb'))
        categoricals_encoded = loaded_encoder.transform(X).toarray()
        categoricals_encoded = pd.DataFrame(categoricals_encoded, columns = loaded_encoder.get_feature_names_out())
        return categoricals_encoded

def __bedrooms(x):
    if x > 6:       
        return "many"
    else:
        return x
def __bathrooms(x):
    x = int(round(x, 0))
    if x == 0:
        return 1
    else:
        return x

def months (x):
    months = (date.today().year - x.year) * 12 + date.today().month - x.month
    return months

def sep_cont_disc(df):
    data_continous = df[["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15", "price"]]
    data_discrete = df.drop(columns=data_continous.columns)
    return data_continous, data_discrete

def power_e(x):
    return np.e**x
