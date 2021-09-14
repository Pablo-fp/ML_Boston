import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Config the page tittle on the browser
st.set_page_config(
    page_title='Residential Price Prediction App')


sidebar = st.sidebar
header = st.container()
dataset = st.container()
model = st.container()
prediction = st.container()

with sidebar:
    # Config a sidebar with link to my sites
    void, picture, void = (st.columns(3))
    profile = Image.open('Data/linkedin.png')
    picture.image(profile)

    st.markdown("""  
    All the code is available for you to use in [Github](https://github.com/Pablo-fp).  
    You can follow me in [LinkedIn](https://www.linkedin.com/in/pablo-fernandez-perez/) for more content.  
    If you want to learn more about Streamlit I recommend you visit the channels of [Data Professor](https://www.youtube.com/c/DataProfessor) and [Misra Turp](https://www.youtube.com/c/Soyouwanttobeadatascientist) in Youtube.
     """)
    st.write('---')


with header:
    # LOAD HEADER
    image = Image.open('Data/residential-price-prediction.jpg')
    st.image(image)
    st.markdown("""
    # **Residential Price Prediction Web App**
    We are going to use a dataset to build a machine learning model that predicts one of the features, in this case the house price.
    """)
    st.write('---')

with dataset:
    st.markdown("""
    # 1.Dataset
    We will be using scikit-learn’s boston dataset. A famous dataset originally published in 1978 by Harrison, D. and Rubinfeld, D.L. *`Hedonic prices and the demand for clean air´*.
     """)
    # Load boston dataset
    boston = datasets.load_boston()
    data = pd.DataFrame(boston.data)
    data.columns = boston.feature_names
    data['PRICE'] = boston.target

    # Load dataset´s head
    st.write(data.head())
    st.markdown("""
    In this dataset, each row describes a boston town or suburb. There are 506 rows and 13 attributes (features) with an additional column, our target (price).
     """)
    # Load dataset´s shape
    st.write('*Dataset´s shape*:', data.shape)
    st.markdown("""  
    **Features:**  
    **CRIM**: per capita crime rate by town  
    **ZN**: proportion of residential land zoned for lots over 25,000 sq.ft.  
    **INDUS**: proportion of non-retail business acres per town  
    **CHAS**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)  
    **NOX**: nitric oxides concentration (parts per 10 million)  
    **RM**: average number of rooms per dwelling  
    **AGE**: proportion of owner-occupied units built prior to 1940  
    **DIS**: weighted distances to five Boston employment centres  
    **RAD**: index of accessibility to radial highways  
    **TAX**: full-value property-tax rate per 10,000usd  
    **PTRATIO**: pupil-teacher ratio by town  
    **B 1000(Bk - 0.63)^2**: where Bk is the proportion of blacks by town  
    **LSTAT**: % lower status of the population  
    **MEDV**: median value of owner-occupied homes in $1000's
    """)

    st.markdown("""
    **Plotting the dataset in Boston´s map:**
    Taking advantage of Streamlit´s visualization capabilities you can plot the dataset in a map using its coordinates.
    (Originally the dataset included two more columns *longitude* and *latitude* of each register. I have cleaned the datasets to eliminate points over water bodies. You can find this edited file in my github account)
     """)
    # Loading dataset with coordinates points
    boston_points = pd.read_csv('Data/Boston_points.csv')
    # Plotting coordinates in map
    st.map(boston_points)

    # Filtering data in the map
    txt_column, map_column = (st.columns(2))
    txt_column.markdown("""
    **Filtering by the PRICE of the houses in the Boston´s map:**
    You can filter the dataset, in this case as an example you can filter by the median value of the homes, to see the relations beetween the towns on the map.
    Remember each number represents the value in 1k, so the slider goes **from 5,000 to 50,000$** in steps of 5,000$ (average prices from 1978 and before).
     """)
    selected_value = txt_column.slider(
        "Select Median Value (PRICE) of homes in $1000's", min_value=5, max_value=50, value=10, step=5)
    filtered = boston_points[(boston_points['MEDV'] >= selected_value) & (
        boston_points['MEDV'] < (selected_value + 5))]
    map_column.map(filtered)
    st.write('---')

with model:
    st.markdown("""
    # 2.Model
    For the prediction model I am going to use a **Random Forest Regressor**.
    In this case I will reuse the code of *Shreayan Chaudhary* in kaggle, you can get more info here: [Boston prediction by Shreayan Chaudhary](https://www.kaggle.com/shreayan98c/boston-house-price-prediction).
     """)
    # Spliting target variable and independent variables
    X = data.drop(['PRICE'], axis=1)
    y = data['PRICE']
    # Splitting to training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4)
    # Create a Random Forest Regressor
    rfr = RandomForestRegressor()
    # Train the model using the training sets
    rfr.fit(X_train, y_train)
    # Model prediction on train data
    y_pred = rfr.predict(X_train)
    # Predicting Test data with the model
    y_test_pred = rfr.predict(X_test)
    # Model Evaluation
    acc_rf = metrics.r2_score(y_test, y_test_pred)
    st.write('**Model Evaluation Metrics of the chosen Ramdom Forest**')
    st.write('**R^2**:', acc_rf, '- It is a measure of the linear relationship between X and Y. It is interpreted as the proportion of the variance in the dependent variable that is predictable from the independent variable.')
    st.write('**Adjusted R^2**:', 1 - (1-metrics.r2_score(y_test, y_test_pred))
             * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1), '- The adjusted R-squared compares the explanatory power of regression models that contain different numbers of predictors.')
    st.write('**MAE**:', metrics.mean_absolute_error(y_test, y_test_pred),
             '- It is the mean of the absolute value of the errors. It measures the difference between two continuous variables, here actual and predicted values of y.')
    st.write('**MSE**:', metrics.mean_squared_error(y_test, y_test_pred),
             '- The mean square error (MSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value.')
    st.write('**RMSE**:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
             '- The root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed.')

    # Visualizing the differences between actual prices and predicted values
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.scatter(y_train, y_pred, color='lightcoral')
    plt.xlabel("Prices")
    plt.ylabel("Predicted prices")
    plt.title("Prices vs Predicted prices")
    st.pyplot()
    st.write('---')

with prediction:

    st.markdown("""
    # 3.Prediction
    Now you can set the features to obtain a predicted price using the parameters in the left column.
     """)
    # Dividing the screen in two columns, input and result
    input_column, result_column = (st.columns(2))

    # Input Parameters in input_column
    input_column.header('Specify Input Parameters')

    # Function with sliders and user inputs
    def user_input_features():
        CRIM = input_column.slider('CRIM: per capita crime rate by town', float(
            X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
        ZN = input_column.slider('ZN: proportion of residential land zoned', float(X.ZN.min()),
                                 float(X.ZN.max()), float(X.ZN.mean()))
        INDUS = input_column.slider('INDUS: proportion of non-retail business acres per town', float(X.INDUS.min()),
                                    float(X.INDUS.max()), float(X.INDUS.mean()))
        CHAS = input_column.slider('CHAS: Charles River dummy variable', float(
            X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
        NOX = input_column.slider('NOX: nitric oxides concentration (parts per 10 million)', float(
            X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
        RM = input_column.slider('RM: average number of rooms per dwelling', float(
            X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
        AGE = input_column.slider('AGE: proportion of owner-occupied units built prior to 1940', float(
            X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
        DIS = input_column.slider('DIS: weighted distances to five Boston employment centres', float(
            X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
        RAD = input_column.slider('RAD: index of accessibility to radial highways', float(
            X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
        TAX = input_column.slider('TAX: full-value property-tax rate per 10,000usd', float(
            X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
        PTRATIO = input_column.slider(
            'PTRATIO: pupil-teacher ratio by town', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
        B = input_column.slider('B: where Bk is the proportion of blacks by town', float(X.B.min()),
                                float(X.B.max()), float(X.B.mean()))
        LSTAT = input_column.slider('LSTAT: % lower status of the population', float(X.LSTAT.min()),
                                    float(X.LSTAT.max()), float(X.LSTAT.mean()))
        data = {'CRIM': CRIM,
                'ZN': ZN,
                'INDUS': INDUS,
                'CHAS': CHAS,
                'NOX': NOX,
                'RM': RM,
                'AGE': AGE,
                'DIS': DIS,
                'RAD': RAD,
                'TAX': TAX,
                'PTRATIO': PTRATIO,
                'B': B,
                'LSTAT': LSTAT}
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

    # Print Result in result_column
    result_column.header('Prediction of MEDV')
    result_column.markdown(
        '(Median value of owner-occupied homes in the 70s):')
    prediction = round(float(rfr.predict(df))*1000, 2)
    prediction_str = str(prediction) + ' $'
    result_column.title(prediction_str)
    result_column.write('---')

    # Print Feature importances plot in result_column
    result_column.header('Feature Importances')
    result_column.markdown("""
    As you may have noticed not all of the features contribute to the model equally or with the same **importance**.  
    Scikit-learn provides a *feature importances* variable with the model, which shows the relative importance of each feature. 
    The scores are scaled down so that the sum of all scores is 1.
     """)

    features = boston['feature_names']
    importances = rfr.feature_importances_
    indices = np.argsort(importances)

    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),
             importances[indices], color='lightcoral', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    result_column.pyplot()

    result_column.markdown("""
    We can remove the features with the less importance to improve the performance of our model. Removing some noise and highly correlated features will increase the accuracy.
     """)

    st.write('---')
