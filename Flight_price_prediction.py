#!/usr/bin/env python
# coding: utf-8

# # Importing all libraries!

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data = pd.read_excel(r"C:\Users\AkshayKumarJShetty\OneDrive - TheMathCompany Private Limited\Desktop\Udemy_Projects\Data_Train.xlsx")


# In[3]:


train_data.head(5)


# In[4]:


train_data.info()


# In[5]:


train_data.isnull().sum()


# In[6]:


# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[7]:


train_data.shape

Importing dataset
1.Since data is in form of excel file we have to use pandas read_excel to load the data
2.After loading it is important to check null values in a column or a row
3.If it is present then following can be done,
    a.Filling NaN values with mean, median and mode using fillna() method
    b.If Less missing values, we can drop it as well
# In[8]:


### getting all the rows where we have missing value
train_data[train_data['Total_Stops'].isnull()]


# #### As we have 1 missing value , I can directly drop these

# In[9]:


train_data.dropna(inplace=True)


# In[10]:


train_data.isnull().sum()


# In[11]:


train_data


# ## Pre-process & Perform Featurization of "Date_of_Journey"
#     ie pre-process it & extract day,month,year from "Date_of_Journey" feature..

# In[12]:


data=train_data.copy()


# In[13]:


data.head(2)


# In[14]:


data.dtypes


# we can see that Date_of_Journey is a object data type,
# Therefore, we have to convert this datatype into timestamp bcz our 
# model will not be able to understand these string values,it just understand Time-stamp..
# 
# For this we require pandas to_datetime to convert object data type to datetime dtype.

# In[15]:


def change_into_datetime(col):
    data[col]=pd.to_datetime(data[col])


# In[16]:


data.columns


# In[17]:


for feature in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
    change_into_datetime(feature)


# In[18]:


data.dtypes


# In[19]:


data['Date_of_Journey'].min()


# In[20]:


data['Date_of_Journey'].max()


# ##  Feature Engineering of "Date_of_Journey" & fetch day,month,year !

# In[21]:


data['day_of_journey'] = data['Date_of_Journey'].dt.day
data['month_of_journey'] = data['Date_of_Journey'].dt.month
data['year_of_journey'] = data['Date_of_Journey'].dt.year


# In[22]:



# year = lambda x: x.year
# month = lambda x: x.month
# day = lambda x: x.day


# In[23]:


data.head()


# In[24]:


data.drop('Date_of_Journey',axis=1,inplace=True)


# In[25]:


data.head()


# ## Clean Dep_Time & Arrival_Time & featurize it..

# In[26]:


def extract_hour_min(df,col):
    df[col+'_hour'] = df[col].dt.hour
    df[col+'_minute'] = df[col].dt.minute
    df.drop(col,axis = 1,inplace = True)
    return df.head(2)


# In[27]:


extract_hour_min(data,'Dep_Time')


# In[28]:


extract_hour_min(data,'Arrival_Time')


# ## Analyse when will most of the flights will take-off

# In[29]:


### Converting the flight Dep_Time into proper time i.e. mid_night, morning, afternoon and evening.

def flight_dep_time(x):
    '''
    This function takes the flight Departure time 
    and convert into appropriate format.
    '''
    if ( x> 4) and (x<=8 ):
        return 'Early mrng'
    
    elif ( x>8 ) and (x<=12 ):
        return 'Morning'
    
    elif ( x>12 ) and (x<=16 ):
        return 'Noon'
    
    elif ( x>16 ) and (x<=20 ):
        return 'Evening'
    
    elif ( x>20 ) and (x<=24 ):
        return 'Night'
    else:
        return 'Late night'


# In[30]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind = 'bar')


# In[31]:


import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[32]:


cf.go_offline()


# In[33]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind = 'bar')


# In[34]:


data.head(10)


# ## Pre-process Duration Feature & extract meaningful features 
Lets Apply pre-processing on duration column,
-->> Once we pre-processed our Duration feature , lets featurize this feature & extract Duration hours and minute from duration..


-->> As my ML model is not able to understand this duration as it contains string values , thats why we have to tell our
Ml Model that this is Duration_hour & this Duration_is minute..
# In[35]:


def preprocess_duration(x):
    if 'h' not in x:
        x='0h '+x
    elif 'm' not in x:
        x=x+' 0m'
    return x
    


# In[36]:


data['Duration']=data['Duration'].apply(preprocess_duration)


# In[37]:


data['Duration']


# In[38]:


data['Duration'][0].split(' ')[0]


# In[39]:


int(data['Duration'][0].split(' ')[0][0:-1])


# In[40]:


int(data['Duration'][0].split(' ')[1][0:-1])


# In[41]:


data['Duration_hours']=data['Duration'].apply(lambda x:int(x.split(' ')[0][0:-1]))


# In[42]:


data['Duration_mins']=data['Duration'].apply(lambda x:int(x.split(' ')[1][0:-1]))


# In[43]:


data.head()


# ##  Analyse whether Duration impacts on Price or not ?

# In[44]:


data['Duration_total_mins']=data['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[45]:


data.head()


# In[46]:


#### It Plot data and regression model fits across a FacetGrid.. (combination of 'regplot` and :class:`FacetGrid)
#### its a extended form of scatter plot..

sns.lmplot(x='Duration_total_mins',y='Price',data=data)


## Conclusion-->> pretty clear that As the duration of minutes increases Flight price also increases.


# ## which city has maximum final destination of flights ?

# In[47]:


data['Destination'].unique()


# In[48]:


data['Destination'].value_counts().iplot(kind='bar')

'''
Insights->> 
Final destination of majority of flights is Cochin. There are two values for Delhi destination which needs to be corrected,

'''
# ## Lets Perform Exploratory Data Analysis(Bivariate Analysis) to come up with some business insights
#     Problem Statement-->> on which route Jet Airways is extremely used? 
#                           Are they economical?

# In[49]:


data['Route']


# In[50]:


data[data['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending=False)


# ## Airline vs Price Analysis
#     ie finding price distribution & 5-point summary of each Airline..

# In[51]:


# plt.figure(figsize=(15,5))
# sns.boxplot(y='Price',x='Airline',data=data)
# plt.xticks(rotation='vertical');


# In[52]:


import plotly.express as px
fig = px.box(data,y='Price',x='Airline',color = 'Airline')
fig.update_layout (showlegend=False)
fig.show()


# Insights--> From graph we can see that Jet Airways Business have the highest Price., Apart from the first Airline almost all are having similar median

# In[53]:


fig = px.violin(data, y='Price',x='Airline', color='Airline', box=True, points='all',hover_data=data.columns)
fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout (showlegend=False)
fig.show()


# ## Lets Perform Feature-Encoding on Data !
#     Applying one-hot on data !

# In[54]:


data.head()


# In[55]:


np.round(data['Additional_Info'].value_counts()/len(data)*100,2)


# In[56]:


# Additional_Info contains almost 80% no_info,so we can drop this column
# we can drop Route as well as we have pre-process that column
## lets drop Duration_total_mins as we have already extracted "Duration_hours" & "Duration_mins"

data.drop(columns=['Additional_Info','Route','Duration_total_mins','year_of_journey'],axis=1,inplace=True)


# In[57]:


data.columns


# In[58]:


data.head(4)


# ## Separate categorical data & numerical data !
#     categorical data are those whose data-type is 'object'
#     Numerical data are those whose data-type is either int of float

# In[59]:


cat_col=[col for col in data.columns if data[col].dtype=='object']


# In[60]:


num_col=[col for col in data.columns if data[col].dtype!='object']


# In[61]:


cat_col


# ## Handling Categorical Data
#     We are using 2 basic Encoding Techniques to convert Categorical data into some numerical format
#     if data belongs to Nominal data (ie data is not in any order) -->> OneHotEncoder is used in this case
#     if data belongs to Ordinal data (ie data is in order ) -->>       LabelEncoder is used in this case

# ### Lets apply one-hot encoding on 'Source' feature !

# In[62]:


data['Source'].unique()


# In[63]:


data['Source']


# In[64]:


data['Source'].apply(lambda x: 1 if x=='Banglore' else 0)


# In[65]:


for category in data['Source'].unique():
    data['Source_'+category]=data['Source'].apply(lambda x: 1 if x==category else 0)


# In[66]:


data.head(3)


# ## Performing Target Guided Mean Encoding !
#     ofcourse we can use One-hot , but if we have more sub-categories , it creates curse of dimensionality in ML..
#     lets use Target Guided Mean Encoding in order to get rid of this..

# In[67]:


airlines=data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[68]:


airlines


# In[69]:


dict1={key:index for index,key in enumerate(airlines,0)}


# In[70]:


dict1


# In[71]:


data['Airline']=data['Airline'].map(dict1)


# In[72]:


data['Airline']


# In[73]:


data.head(2)


# In[74]:


data['Destination'].unique()


# Note: till now , Delhi (Capital of India) has one Airport & its second Airport is yet to build in Greater Noida (Jewar)
#       which is part of NCR , so we will consider New Delhi & Delhi as same ...
# 
# 
#       but in future , these conditions may change..

# In[75]:


data['Destination'].replace('New Delhi','Delhi',inplace=True)


# In[76]:


data['Destination'].unique()


# In[79]:


dest=data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[80]:


dest


# In[81]:


dict2={key:index for index,key in enumerate(dest,0)}


# In[82]:


dict2


# In[83]:


data['Destination']=data['Destination'].map(dict2)


# In[84]:


data['Destination']


# In[85]:


data.head(2)


# ## Perform Manual Encoding on Total_stops feature

# In[86]:


data['Total_Stops'].unique()


# In[87]:


stops={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[88]:


data['Total_Stops']=data['Total_Stops'].map(stops)


# In[89]:


data['Total_Stops']


# ## Performing Outlier Detection !
#     Here the list of data visualization plots to spot the outliers.
# 1. Box and whisker plot (box plot).
# 2. Scatter plot.
# 3. Histogram.
# 4. Distribution Plot.
# 5. QQ plot
CAUSE FOR OUTLIERS
* Data Entry Errors:- Human errors such as errors caused during data collection, recording, or entry can cause outliers in data.
* Measurement Error:- It is the most common source of outliers. This is caused when the measurement instrument used turns out to be faulty.
* Natural Outlier:- When an outlier is not artificial (due to error), it is a natural outlier. Most of real world data belong to this category.
# In[90]:


def plot(df,col):
    fig,(ax1,ax2,ax3)=plt.subplots(3,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
    sns.distplot(df[col],ax=ax3,kde=False)


# In[91]:


plot(data,'Price')

getting a high level over-view of various ways to deal with outliers:Again there are various ways to deal with outliers :


1..Statistical imputation , ie impute it with mean , median or mode of data..

a..Whenever ur data is Gaussian Distributed ,use 3 std dev approach to remove outliers in such case
     ie we will use u+3*sigma & u-3*sigma
        data pts greater than upper_boundary( u+3*sigma) are my outliers 
            & data pts which are less than lower_boundary(u-3*sigma) are my outliers

        Above approach is known as Z-score & it has a extended version known as Robust z-score..
        Robust Z-score is also called as Median absolute deviation method. 
        It is similar to Z-score method with some changes in parameters.


b..If Features Are Skewed We Use the below Technique which is IQR
    Data which are greater than IQR +1.5 IQR and data which are below than IQR - 1.5 IQR are my outliers
     where IQR=75th%ile data - 25th%ile data

     & IQR +- 1.5 IQR  will be changed depending upon the domain ie it may be IQR + 3IQR 


       Extended version of above is WINSORIZATION METHOD(PERCENTILE CAPPING)..
       This method is similar to IQR method. It says -->> 

       Data points that are greater than 99th percentile and data points that are below tha 1st percentile 
       are treated as outliers.



 c..If we have huge high dimensional data , then it is good to perform isolation forest...
     It is a clustering algo which works based on decision tree and it isolate the outliers.
     It classify the data point to outlier and not outliers..
         If the result is -1, it means that this specific data point is an outlier. 
         If the result is 1, then it means that the data point is not an outlier.






So we have tonnes of ways to deal with outliers..
# In[92]:


data['Price']=np.where(data['Price']>=35000,data['Price'].median(),data['Price'])


# In[93]:


plot(data,'Price')


# In[94]:


data.head(2)


# In[95]:


data.drop(columns=['Source','Duration'],axis=1,inplace=True)


# In[96]:


data.head(2)


# In[97]:


data.dtypes


# ## Performing Feature Selection !
 Finding out the best feature which will contribute most to the target variable. 
Lets get a high level overview of most of the frequently used feature selection technique..


Why to apply Feature Selection?
To select important features to get rid of curse of dimensionality ie..to get rid of duplicate features


ways or technqiues to do it if we have regression use-case
a..SelectKBest
    Score function:
    
    For regression: f_regression, mutual_info_regression

    f_regression
    Its backbone is pearson co-relation.. 


    mutual_info_regression 
    Its Backbone is Various statistical test like Chi-sq,Anova & p-value.


b..ExtraTreesClassifier
   This technique gives you a score for each feature of your data,the higher the score more relevant it is


# In[98]:


from sklearn.feature_selection import mutual_info_regression


# In[99]:


X=data.drop(['Price'],axis=1)


# In[100]:


y=data['Price']


# In[101]:


X.dtypes


# In[102]:


mutual_info_regression(X,y)


# In[103]:


imp=pd.DataFrame(mutual_info_regression(X,y),index=X.columns)
imp.columns=['importance']


# In[104]:


imp.sort_values(by='importance',ascending=False)


# In[105]:


from sklearn.model_selection import train_test_split


# In[106]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

what we often do in modelling:
a..Initially ,lets build basic random forest model.
b..then later-on , we will try to improve this model using some parameters..
c..Then we will hyper-tune my model to get optimal value of parameters in order to achieve optimal value of params..
# In[107]:


from sklearn.ensemble import RandomForestRegressor


# In[108]:


ml_model=RandomForestRegressor()


# In[109]:


model=ml_model.fit(X_train,y_train)


# In[110]:


y_pred=model.predict(X_test)


# In[111]:


y_pred


# In[112]:


y_pred.shape


# In[113]:


len(X_test)


# ## How to save ML model into disk
lets try to dump ml model using pickle & joblib..
advantage of dumping--
imagine in future we have new data ,& lets say we have to predict price on this huge data

then just for this new data , we have to execute all the above cells follow the entire pipeline,  then only we are able to predict on this...


so to get rid of such issue , will just dump it to reuse it again & again..
what does this file store??
this save coefficients of our model.. not an entire dataset
# In[117]:


# pip install pickle


# In[118]:


import pickle


# In[119]:


file = open(r'C:\Users\AkshayKumarJShetty\OneDrive - TheMathCompany Private Limited\Desktop\Udemy_Projects/rf_random.pkl','wb')


# In[120]:


pickle.dump(model,file)


# In[121]:


model=open(r'C:\Users\AkshayKumarJShetty\OneDrive - TheMathCompany Private Limited\Desktop\Udemy_Projects/rf_random.pkl','rb')


# In[122]:


forest=pickle.load(model)


# In[123]:


forest.predict(X_test)


# ## Defining your own evaluation metric :

# In[124]:


def mape(y_true,y_pred):
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


# In[125]:


mape(y_test,forest.predict(X_test))


# ## How to Automate ML Pipeline

# In[126]:


def predict(ml_model):
    
    model=ml_model.fit(X_train,y_train)
    print('Training_score: {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('Predictions are : {}'.format(y_prediction))
    print('\n')
    
    from sklearn import metrics
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2_score: {}'.format(r2_score))
    print('MSE : ', metrics.mean_squared_error(y_test,y_prediction))
    print('MAE : ', metrics.mean_absolute_error(y_test,y_prediction))
    print('RMSE : ', np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    print('MAPE : ', mape(y_test,y_prediction))
    sns.distplot(y_test-y_prediction)
    


# In[127]:


predict(RandomForestRegressor())


# ## how to hypertune ml model
#     Hyperparameter Tuning or Hyperparameter Optimization
#     1.Choose following method for hyperparameter tuning
#         a.RandomizedSearchCV --> Fast way to Hypertune model
#         b.GridSearchCV--> Slow way to hypertune my model
#     2.Choose ML algo that u have to hypertune
#     2.Assign hyperparameters in form of dictionary or create hyper-parameter space
#     3.define searching &  apply searching on Training data or  Fit the CV model 
#     4.Check best parameters and best score

# In[128]:


from sklearn.model_selection import RandomizedSearchCV


# In[129]:


### initialise your estimator
reg_rf=RandomForestRegressor()


# In[130]:


np.linspace(start=1000,stop=1200,num=6)


# In[131]:


# Number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=1000,stop=1200,num=6)]

# Number of features to consider at every split
max_features=["auto", "sqrt"]

# Maximum number of levels in tree
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=4)]

# Minimum number of samples required to split a node
min_samples_split=[5,10,15,100]


# In[132]:


# Create the grid or hyper-parameter space
random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split
    
}


# In[133]:


random_grid


# In[134]:


rf_Random=RandomizedSearchCV(reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)


# In[135]:


rf_Random.fit(X_train,y_train)


# In[136]:


### to get your best model..
rf_Random.best_params_


# In[137]:


pred2=rf_Random.predict(X_test)


# In[138]:


from sklearn import metrics
metrics.r2_score(y_test,pred2)


# In[ ]:




