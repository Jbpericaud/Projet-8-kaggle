
# coding: utf-8

# # Store Sales Eda, Modelisation And Prediction Test-set

# In[1]:


# I thank the authors of these notebooks for their knowledge and help on this kaggle competition
# (Store Sales - Time Series Forecasting).
#
# This is my first Kaggle, and I wanted to excercise myself in the field of time series prediction,
# as I am a beginner in this field.
#
# This notebook will be presented in 3 parts.
# An exploratory analysis on my different variables (allowing to explain the relations
# of different components on my variable Target "sales"°)
# Several modeling analyses
# And a prediction analysis of my Test samples.
#
# https://www.kaggle.com/code/ilyakondrusevich/54-stores-54-models
#
# https://www.kaggle.com/code/andrej0marinchenko/hyperparamaters

# In[345]:


# Librairie
# importation librarie
import matplotlib as plt
import matplotlib.pyplot as plt  # For use legend, xlabel, ylabel on plt
import numpy as np
import scipy as sp
import sklearn
from IPython.display import display
import pandas as pd

import seaborn as sns
import missingno as msno
import pandas as pd
import calendar

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.express as px
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from statistics import mean


# In[346]:


download_dir = Path(r"C:\Users\JbPer\Openclassroom\Projet 8\stores")


# In[347]:


holidays_events = pd.read_csv(download_dir / "holidays_events.csv", sep=",")
oil = pd.read_csv(download_dir / "oil.csv", sep=",")
stores = pd.read_csv(download_dir / "stores.csv", sep=",")
test = pd.read_csv(download_dir / "test.csv", sep=",")
train = pd.read_csv(download_dir / "train.csv", sep=",")


# We have 5 files, we will try to merge these files together.
# By grouping oil, store (a categorical variable that represents the store that sold the product),
# vacations and public holidays (holidays_events) on training and test data.
# (The transaction values are unknown on the Test set data (this is logical). Moreover the Target variable "sales" is
# too dependent on them. This Transaction variable is not a good variable for our prediction analysis).

# Our goal is to predict the sales of the different product families sold in the different Favorita stores.
# There are 54 Favorita stores in this dataset.
# There are 54 stores and my idea is to train the model on the data of each store separately.

# # Processing of training data

oil_train = train.merge(oil, how="left", on="date")
oil_train_st = oil_train.merge(stores, how="left", on="store_nbr")
oil_train_st_h = oil_train_st.merge(holidays_events, how="left", on="date")

oil_train_st_tr_h = oil_train_st_h.rename(
    columns={"type_x": "store_type", "type_y": "ho_or_nt"}
)

#  Only the number of orders is kept to avoid duplicates (because for the same day you can get several events)

dt_Train = oil_train_st_tr_h.drop_duplicates(subset=["id"])

dt_Train.date = pd.to_datetime(dt_Train.date)
dt_Train["year"] = dt_Train["date"].dt.year
dt_Train["month"] = dt_Train["date"].dt.month
dt_Train["week"] = dt_Train["date"].dt.isocalendar().week
dt_Train["quarter"] = dt_Train["date"].dt.quarter
dt_Train["day_of_week"] = dt_Train["date"].dt.day_name()


# Here, we observe that some variables are missing (Nan value),
# notably the Holiday_or_not and Dcoilwtico columns.

# ## For event variables

dt_Train[["locale", "locale_name", "description"]] = dt_Train[
    ["locale", "locale_name", "description"]
].replace(np.nan, "")
dt_Train["ho_or_nt"] = dt_Train["ho_or_nt"].replace(np.nan, "Work Day")
dt_Train["transferred"] = dt_Train["transferred"].replace(np.nan, False)
dt_Train["ho_or_nt"] = dt_Train["ho_or_nt"].replace(np.nan, "Work Day")

# ### For missing oil data, we will use interpolation techniques to find the missing values
oil["date"] = pd.to_datetime(oil["date"])

oil = (
    oil.set_index("date")["dcoilwtico"].resample("D").sum().reset_index()
)  # add missing dates and fill NaNs with 0

# Interpolate
oil["dcoilwtico"] = np.where(
    oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"]
)  # replace 0 with NaN
oil[
    "dcoil_interpolated"
] = oil.dcoilwtico.interpolate()  # fill NaN values using an interpo method

oil.drop(["dcoilwtico"], axis=1, inplace=True)
dt_Train.drop(["dcoilwtico"], axis=1, inplace=True)
dt_Train = dt_Train.merge(oil, how="left", on="date")
dt_Train = dt_Train.dropna(subset=["sales"]) # Just remove unnecessary lines

# We will do the same thing for the Test set data

oil = pd.read_csv(download_dir / "oil.csv", sep=",")
oil_test = test.merge(oil, how="left", on="date")
oil_test_st = oil_test.merge(stores, how="left", on="store_nbr")
oil_test_st_h = oil_test_st.merge(holidays_events, how="left", on="date")
oil_test_st_h = oil_test_st_h.rename(
    columns={"type_x": "store_type", "type_y": "ho_or_nt"}
)
dt_test = oil_test_st_h.drop_duplicates(subset=["id"])

dt_test = oil_test_st_h
dt_test.date = pd.to_datetime(dt_test.date)
dt_test["year"] = dt_test["date"].dt.year
dt_test["month"] = dt_test["date"].dt.month
dt_test["week"] = dt_test["date"].dt.isocalendar().week
dt_test["quarter"] = dt_test["date"].dt.quarter
dt_test["day_of_week"] = dt_test["date"].dt.day_name()

dt_test[["locale", "locale_name", "description"]] = dt_test[
    ["locale", "locale_name", "description"]
].replace(np.nan, "")

dt_test["ho_or_nt"] = dt_test["ho_or_nt"].replace(np.nan, "Work Day")
dt_test["transferred"] = dt_test["transferred"].replace(np.nan, False)

oil["date"] = pd.to_datetime(oil["date"])
oil = (
    oil.set_index("date")["dcoilwtico"].resample("D").sum().reset_index()
)  # add missing dates and fill NaNs with 0

# Interpolate
oil["dcoilwtico"] = np.where(
    oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"]
)  # replace 0 with NaN
oil[
    "dcoil_interpolated"
] = oil.dcoilwtico.interpolate()  # fill NaN values using interpolation method
oil.drop(["dcoilwtico"], axis=1, inplace=True)
dt_test.drop(["dcoilwtico"], axis=1, inplace=True)
dt_test = dt_test.merge(oil, how="left", on="date")

# Save DataFrame test
dt_test.to_csv("TestFinish.csv", index=False)

# # Let's move on to the modeling stage

# Before predicting the test values, we will build several models to improve the accuracy of the model.
# The metric used will be the RMSE
#
# Why RMSE ?
#
# It is often used to predict time series and has many advantages. It is robust to outliers
# (we have not treated outliers). It calculates the relative error between the predicted and actual values
# and penalises underestimation of the actual value more severely than overestimation. In general, unlike RMSE
# (Root Mean Squared Error), it performs well with labels that can take on values over several orders of magnitude.
#
# We can see document of RMSE:
# https://www.kaggle.com/general/215997 of Satish Gunjal
# Or Very good Notebook on Model Fit Metrics
# https://www.kaggle.com/code/residentmario/model-fit-metrics/notebook of Aleksey Bilogur
#
#
#

# We will use the functions of the course (Kaggle Time Series) to build our model

# In[399]:


simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[414]:


def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update(
            {f"sin_{freq}_{i}": np.sin(i * k), f"cos_{freq}_{i}": np.cos(i * k)}
        )
    return pd.DataFrame(features, index=index)


# Compute Fourier features to the 4th order (8 new features) for a
# series y with daily observations and annual seasonality:
#
# fourier_features(y, freq=365.25, order=4)

# annotations: https://stackoverflow.com/a/49238256/5769929

# To build our model we will only use the year 2017.
# Because we will predict that 15 days of data following the Train set 
# 
# For our model, we will apply a foursquare feature to capture the overall shape of the curves 
# seasonal, quarterly, monthly and daily.


dt_Train["store_nbr"] = pd.Categorical(dt_Train["store_nbr"])
dt_Train["day_of_week"] = pd.Categorical(dt_Train["day_of_week"])
dt_Train["city"] = pd.Categorical(dt_Train["city"])
dt_Train["cluster"] = pd.Categorical(dt_Train["cluster"])
dt_Train["store_type"] = pd.Categorical(dt_Train["store_type"])
dt_Train["state"] = pd.Categorical(dt_Train["state"])
dt_Train["city"] = pd.Categorical(dt_Train["city"])
dt_Train["family"] = pd.Categorical(dt_Train["family"])
dt_Train["ho_or_nt"] = pd.Categorical(dt_Train["ho_or_nt"])
dt_Train["description"] = pd.Categorical(dt_Train["description"])

# Save Data train
dt_Train.to_csv("TrainFinish.csv", index=False)


download_dir2 = Path(r"C:\Users\JbPer\Openclassroom\Projet 8")

# Load Data for created X_variable (Dataframe of explanatory variable)
store_sales1 = pd.read_csv(
    download_dir2 / "TrainFinish.csv",
    usecols=[
        "store_nbr",
        "family",
        "date",
        "day_of_week",
        "city",
        "ho_or_nt",
        "description",
        "dcoil_interpolated",
        "year",
        "month",
        "week",
        "quarter",
        "sales",
    ],
    dtype={
        "store_nbr": "float32",
        "family": "category",
        "sales": "float32",
        "day_of_week": "category",
        "city": "category",
        "ho_or_nt": "category",
        "description": "category",
    },
    parse_dates=["date"],
    infer_datetime_format=True,
)
store_sales1 = store_sales1.dropna(subset=["sales"])
store_sales1["store_nbr"] = store_sales1["store_nbr"].astype(float).astype(int)
store_sales1["store_nbr"] = pd.Categorical(store_sales1["store_nbr"])

store_sales1["date"] = store_sales1.date.dt.to_period("D")

store_sales1 = store_sales1.set_index(["store_nbr",
                                       "family", "date"]).sort_index()


# In[161]:


s = store_sales1.reset_index()
s = s.loc[(s["date"] >= "2017-01-01")]
s.drop(
    ["store_nbr", "family", "sales", "city", "description", "year"],
    axis=1,
    inplace=True,
)
s = s.drop_duplicates(subset=["date"])

transformer = make_column_transformer(
    (OneHotEncoder(sparse=False), ["ho_or_nt", "day_of_week"]), remainder="passthrough",
)

transformed = transformer.fit_transform(s)
transformed_df = pd.DataFrame(transformed,
                              columns=transformer.get_feature_names())


# I have to set my variables to float 

X_variable = transformed_df.set_index(["date"])
X_variable = X_variable.astype(float)


# Model Construction by store

stores = pd.read_csv(download_dir / "stores.csv", sep=",")

stores = stores.set_index(["store_nbr"]).sort_index()


# # Let's move on to the Prediction data 

# with the best model (ridge regression)

dt_test = pd.read_csv(
    download_dir2 / "TestFinish.csv",
    usecols=["store_nbr", "family",
             "date", "onpromotion",
             "dcoil_interpolated"],
    dtype={"family": "object",
           "onpromotion": "int32",
           "store_nbr": "category"},
    parse_dates=["date"],
    infer_datetime_format=True,
)


dt_Train = pd.read_csv(
    download_dir2 / "TrainFinish.csv",
    usecols=[
        "store_nbr",
        "family",
        "date",
        "onpromotion",
        "sales",
        "dcoil_interpolated",
    ],
    dtype={"family": "object",
           "onpromotion": "int32",
           "store_nbr": "category"},
    parse_dates=["date"],
    infer_datetime_format=True,
)


dt_Train["date"] = dt_Train.date.dt.to_period("D")
dt_test["date"] = dt_test.date.dt.to_period("D")

dt_Train["store_nbr"] = pd.Categorical(dt_Train["store_nbr"])
dt_test["store_nbr"] = pd.Categorical(dt_test["store_nbr"])

train_rdy = dt_Train.set_index(["store_nbr", "family", "date"]).sort_index()
train_rdy = train_rdy.dropna(subset=["sales"])

test_rdy = dt_test.set_index(["store_nbr", "family", "date"]).sort_index()

Va_test = pd.read_csv(
    download_dir2 / "TestFinish.csv",
    usecols=[
        "ho_or_nt",
        "month",
        "date",
        "week",
        "dcoil_interpolated",
        "quarter",
        "day_of_week",
    ],
    dtype={"day_of_week": "category", "holiday_or_not": "category"},
    parse_dates=["date"],
    infer_datetime_format=True,
)

s = Va_test.reset_index()
s.drop(["index"], axis=1, inplace=True)

s["date"] = s.date.dt.to_period("D")
s = s.drop_duplicates(subset=["date"])

# Creation of X_variable_test

transformer = make_column_transformer(
    (OneHotEncoder(sparse=False), ["ho_or_nt",
                                   "day_of_week"]),
    remainder="passthrough"
)

transformed = transformer.fit_transform(s)
transformed_df = pd.DataFrame(transformed,
                              columns=transformer.get_feature_names())

X_variable_test = transformed_df.set_index(["date"])
X_variable_test = X_variable_test.astype(float)


# The data_test is missing 3 features:
# onehotencoder__x0_Additional
# onehotencoder__x0_Event
# onehotencoder__x0_Transfer
# They need to be added.
# 

X_variable_test = X_variable_test.assign(
    onehotencoder__x0_Additional=0,
    onehotencoder__x0_Transfer=0,
    onehotencoder__x0_Event=0,
)
X_variable_test = X_variable_test[
    [
        "onehotencoder__x0_Additional",
        "onehotencoder__x0_Event",
        "onehotencoder__x0_Holiday",
        "onehotencoder__x0_Transfer",
        "onehotencoder__x0_Work Day",
        "onehotencoder__x1_Friday",
        "onehotencoder__x1_Monday",
        "onehotencoder__x1_Saturday",
        "onehotencoder__x1_Sunday",
        "onehotencoder__x1_Thursday",
        "onehotencoder__x1_Tuesday",
        "onehotencoder__x1_Wednesday",
        "month",
        "week",
        "quarter",
        "dcoil_interpolated",
    ]
]

X_variable_test = X_variable_test.astype(float)


sdate = "2017-01-01"  # start and end of training date
edate = "2017-08-15"
y_arr = []
onpromotion_train = []
onpromotion_test = []

for nbr in stores.index:
    # y_arr
    temp = train_rdy.loc[str(nbr), "sales"]
    y_arr.append(temp.unstack(["family"]).loc[sdate:edate])

    # onpromotion_Train
    onpromotion = train_rdy.loc[str(nbr), "onpromotion"]
    onpromotion = onpromotion.unstack(["family"]).loc[sdate:edate].sum(axis=1)
    onpromotion.name = "onpromotion"
    onpromotion_train.append(onpromotion)

    # onpromotion_Test
    test_onpromotion = test_rdy.loc[str(nbr), "onpromotion"]
    test_onpromotion = test_onpromotion.unstack(["family"]).sum(axis=1)
    test_onpromotion.name = "onpromotion"
    onpromotion_test.append(test_onpromotion)


# In[291]:


fourier = CalendarFourier(freq="W", order=4)
X_arr = []
X_test_r = []

for y, onpromotion_Tr, onpromotion_Te in zip(
    y_arr, onpromotion_train, onpromotion_test
):

    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=False,
        additional_terms=[fourier],
        drop=True,
    )

    X = dp.in_sample()
    X = X.merge(onpromotion_Tr, how="left",
                left_index=True, right_index=True)
    X = X.merge(X_variable, how="left", on="date")

    X_arr.append(X)

    X_test = dp.out_of_sample(steps=16)
    X_test = X_test.merge(onpromotion_Te,
                          how="left", left_index=True, right_index=True)
    X_test = X_test.reset_index()
    X_test = X_test.rename(columns={"index": "date"})  # Indexprocessing stage
    # X_test.set_index(['date'])
    X_test = X_test.merge(X_variable_test, how="left", on="date")
    X_test = X_test.set_index(["date"])
    X_test_r.append(X_test)


ridge = make_pipeline(RobustScaler(), Ridge(alpha=10))

y_pred_r = []
y_fit_r = []
y_test_r = []
result_test = []

for X_test, X, y in zip(X_test_r, X_arr, y_arr):

    model = ridge.fit(X, y)

    y_fit = pd.DataFrame(model.predict(X),
                         index=X.index,
                         columns=y.columns).clip(0.0)
    y_fit_r.append(y_fit)

    y_test = pd.DataFrame(
        model.predict(X_test),
        index=X_test.index,
        columns=y.columns
    ).clip(0.0)
    y_test_r.append(y_test)
    result_test.append(y_test.stack(["family"]))



# # Data for submission


# Get correct dates for submission
dates = [
    "2017-08-16",
    "2017-08-17",
    "2017-08-18",
    "2017-08-19",
    "2017-08-20",
    "2017-08-21",
    "2017-08-22",
    "2017-08-23",
    "2017-08-24",
    "2017-08-25",
    "2017-08-26",
    "2017-08-27",
    "2017-08-28",
    "2017-08-29",
    "2017-08-30",
    "2017-08-31",
]

# Get correct order for submission
order = list(range(1, len(result_test) + 1))
str_map = map(str, order)
correct_order_str = sorted(list(str_map))
int_minus_one = lambda element: int(element) - 1
correct_order_int = list(map(int_minus_one, correct_order_str))

# Create and fill list with predictions in the correct order
data = []
for date in dates:
    for i in correct_order_int:
        data += result_test[i].loc[date].to_list()

# Create dataframe from the list
result = pd.DataFrame(data, columns=["sales"])


# In[310]:


result


# In[313]:


store_salesTest = pd.read_csv(download_dir / "test.csv", usecols=["id"])


# In[314]:


store_salesTest.shape  # same number of lines as the result


y_submit = store_salesTest.join(result)


y_submit

# Save submission
y_submit.to_csv('submission.csv', index = False)

# Just observation, predict Result


store_salesTest = pd.read_csv(
    download_dir / "test.csv",
    dtype={"store_nbr": "category",
           "family": "category",
           "onpromotion": "uint32"},
    parse_dates=["date"],
    infer_datetime_format=True,
)


pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", None)


Observ_data = store_salesTest.join(result)


food_sales = (
    Observ_data.groupby(by=["family", "date"])
    .agg({"sales": "mean"})
    .reset_index()
    .sort_values(by="date", ascending=True)
)

family_salesx = food_sales.groupby(["family", "date"]).mean().unstack("family")

# We will observe the sales rates of the grocery store (all of  stores)

GROCERY = family_salesx.sales["GROCERY I"]


ax = GROCERY.plot(**plot_params)


# On a 15-day prediction scale, we can see a higher variability in the sales rate on Sunday for GROCERY I 
# 
# Comparison with the real values (the previous week)

fa_sa = (
    dt_Train.groupby(by=["family", "date"])
    .agg({"sales": "mean"})
    .reset_index()
    .sort_values(by="date", ascending=True)
)
mask = (fa_sa["date"] >= "2017-08-01") & (fa_sa["date"] <= "2017-08-15")

fa_sa = fa_sa.loc[mask]
family_sales2 = fa_sa.groupby(["family", "date"]).mean().unstack("family")
food_sales1 = family_sales2.sales["GROCERY I"]


ax = food_sales1.plot(**plot_params)
# Here, we can observe the same variations on the type of product "Grocery". We have well predicted the Test data
