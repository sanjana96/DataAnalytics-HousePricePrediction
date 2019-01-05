
# coding: utf-8

# In[1]:

import pandas as pd


# In[16]:

#Read train data
train = pd.read_csv("train.csv")


# In[17]:

#Do one hot encoding for all the categorical columns using get_dummies funtion
clean=pd.get_dummies(train['MSSubClass'])
clean=clean.join(pd.get_dummies(train['MSZoning']))

#Add column name to category names if the same category exists in more than one column to avoid name clashing
newc=[]
for i in train['Street']:
    newc.append('Street'+(str)(i))
clean=clean.join(pd.get_dummies(newc))
newc=[]
for i in train['Alley']:
    newc.append('Alley'+(str)(i))
clean=clean.join(pd.get_dummies(newc))#No alley access becomes seperate column

clean=clean.join(pd.get_dummies(train['LotShape']))
clean=clean.join(pd.get_dummies(train['LandContour']))
clean=clean.join(pd.get_dummies(train['Utilities']))
clean=clean.join(pd.get_dummies(train['LotConfig']))
clean=clean.join(pd.get_dummies(train['Neighborhood']))

newc=[]
for i in train['Condition1']:
    newc.append('Condition1'+(str)(i))
clean=clean.join(pd.get_dummies(newc))
newc=[]
for i in train['Condition2']:
    newc.append('Condition2'+(str)(i))
clean=clean.join(pd.get_dummies(newc))

clean=clean.join(pd.get_dummies(train['BldgType']))
clean=clean.join(pd.get_dummies(train['HouseStyle']))#can probably be avoided?
clean=clean.join(pd.get_dummies(train['RoofStyle']))
clean=clean.join(pd.get_dummies(train['RoofMatl']))

newc=[]
for i in train['Exterior1st']:
    newc.append('Exterior1st'+(str)(i))
clean=clean.join(pd.get_dummies(newc))
newc=[]
for i in train['Exterior2nd']:
    newc.append('Exterior2nd'+(str)(i))
clean=clean.join(pd.get_dummies(newc))

clean=clean.join(pd.get_dummies(train['MasVnrType']))
#clean=clean.join(pd.get_dummies(train['Foundation']))
clean=clean.join(pd.get_dummies(train['Heating']))
clean=clean.join(pd.get_dummies(train['Electrical']))
clean=clean.join(pd.get_dummies(train['GarageType']))
clean=clean.join(pd.get_dummies(train['PavedDrive']))
#clean=clean.join(pd.get_dummies(train['MiscFeature'])) #Since MiscVal is there, can be discarded?
clean=clean.join(pd.get_dummies(train['SaleType']))
clean=clean.join(pd.get_dummies(train['SaleCondition']))

#Using some columns as it is 
clean['LotFrontage']=train['LotFrontage']
clean['LotArea']=train['LotArea']
clean['OverallQual']=train['OverallQual']
clean['OverallCond']=train['OverallCond']
clean['YearBuilt']=train['YearBuilt']
clean['BsmtFinSF1']=train['BsmtFinSF1']
clean['BsmtFinSF2']=train['BsmtFinSF2']
clean['BsmtUnfSF']=train['BsmtUnfSF']
clean['TotalBsmtSF']=train['TotalBsmtSF']
clean['1stFlrSF']=train['1stFlrSF']
clean['2ndFlrSF']=train['2ndFlrSF']
clean['LowQualFinSF']=train['LowQualFinSF']
clean['GrLivArea']=train['GrLivArea']
clean['BsmtFullBath']=train['BsmtFullBath']
clean['BsmtHalfBath']=train['BsmtHalfBath']
clean['BedroomAbvGr']=train['BedroomAbvGr']
clean['KitchenAbvGr']=train['KitchenAbvGr']
clean['TotRmsAbvGrd']=train['TotRmsAbvGrd']
clean['Fireplaces']=train['Fireplaces']
clean['GarageYrBlt']=train['GarageYrBlt']
clean['GarageCars']=train['GarageCars']
clean['GarageArea']=train['GarageArea']
clean['WoodDeckSF']=train['WoodDeckSF']
clean['OpenPorchSF']=train['OpenPorchSF']
clean['EnclosedPorch']=train['EnclosedPorch']
clean['3SsnPorch']=train['3SsnPorch']
clean['ScreenPorch']=train['ScreenPorch']
clean['PoolArea']=train['PoolArea']
clean['MiscVal']=train['MiscVal']
clean['MoSold']=train['MoSold']
clean['YrSold']=train['YrSold']

clean['SalePrice']=train['SalePrice']


#Giving ordinal values to some columns where categories represent Excellent, good, average etc
def rate(s):
    qual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, "Po": 1, 0: 0,"Mn": 3, "Av": 3,"No": 2}
    s = s.fillna(0)
    return [qual[e] for e in s]

clean['ExterQual'] = rate(train['ExterQual'])
clean['ExterCond'] = rate(train['ExterCond'])
clean['BsmtQual'] = rate(train['BsmtQual'])
clean['BsmtCond'] = rate(train['BsmtCond'])
clean['BsmtQual'] = rate(train['BsmtQual'])
clean['BsmtExposure'] = rate(train['BsmtExposure'])
clean['HeatingQC'] = rate(train['HeatingQC'])
clean['KitchenQual'] = rate(train['KitchenQual'])
clean['FireplaceQu'] = rate(train['FireplaceQu'])
clean['GarageQual'] = rate(train['GarageQual'])
clean['GarageCond'] = rate(train['GarageCond'])
clean['PoolQC'] = rate(train['PoolQC'])

def fin(s):
    f={'GLQ':6 ,'ALQ':5 ,'BLQ':4 ,'Rec':3 ,'LwQ':2 ,'Unf':1 ,'NA':0,0:0}
    s = s.fillna(0)
    return [f[e] for e in s]
clean['BsmtFinType1'] = fin(train['BsmtFinType1'])
clean['BsmtFinType2'] = fin(train['BsmtFinType2'])

def AC(s):
    acpre={'N':0,'Y':1,0:0}
    s = s.fillna(0)
    return [acpre[e] for e in s]
clean['CentralAir']=AC(train['CentralAir'])

def slo(s):
    sl={'Gtl':0,'Mod':1,'Sev':2,0:0}
    s = s.fillna(0)
    return [sl[e] for e in s]
clean['LandSlope']=slo(train['LandSlope'])

def functional(s):
    fn={'Typ':8,'Min1':7,'Min2':6,'Mod':5,'Maj1':4,'Maj2':3,'Sev':2,'Sal':1,0:0}
    s = s.fillna(0)
    return [fn[e] for e in s]
clean['Functional']=functional(train['Functional'])

def garagefin(s):
    fn={'Fin':3,'RFn':2,'Unf':1,'NA':0,0:0}
    s = s.fillna(0)
    return [fn[e] for e in s]
clean['GarageFinish']=garagefin(train['GarageFinish'])

def fencequal(s):
    fn={'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NA':0,0:0}
    s = s.fillna(0)
    return [fn[e] for e in s]
clean['Fence']=fencequal(train['Fence'])


# In[18]:

#Check correlations of each column woth saleprice, sort in descending order
clean.corr().sort(columns="SalePrice", ascending=False)["SalePrice"] 


# In[19]:

#Drop one of the columns of the two columns having correlation>0.8
correlations = clean.corr()
for x in correlations.columns:
    for y in correlations.columns:
        if (str)(x)<(str)(y) and correlations[x][y] > .8 and correlations[x][y] !=1:
          clean = clean.drop([x], axis=1) #axis=1 - column
          print(x, y, correlations[x][y])
            


# In[20]:

clean.shape


# In[21]:

#Drop columns with correlation of less than 0.1 and greater than -0.1 with SalePrice
correlations = clean.corr()
for x in correlations.columns:
    cor=correlations[x]['SalePrice']
    if cor<.1 and cor>-0.1 and cor!=1:
        clean = clean.drop([x], axis=1) #axis=1 meaning column
        print(x,correlations[x]['SalePrice'])


# In[22]:

clean.shape


# In[23]:

#Fill LotFrontage column with na values with 0(No LotFrontage)
clean['LotFrontage']=clean['LotFrontage'].fillna(0)


# In[24]:

import pandas as pd
import numpy as np
#Apply log transformation for columns with high skew

numericcols=[]
for i in clean:
    for j in clean[i]:
        if(j!=0 and j!=1):
            numericcols.append(i)
            break

#Columns with high skew obtained from skew.r            
skew=['30','50','60','90','X160','C(all)','RL','RM','AlleyGrvl','Alleynan','IR2','Bnk','HLS','CulDSac',
      'BrDale','BrkSide','Edwards','IDOTRR','MeadowV','NAmes','NoRidge','NridgHt','OldTown','Sawyer','Somerst'
    ,'StoneBr','Timber','Condition1Artery','Condition1Feedr','Condition1Norm','X1Fam','Duplex','Gable','Hip'
    ,'CompShg','WdShngl','Exterior2ndCmentBd','Exterior2ndMetalSd','Exterior2ndWd.Sdng','BrkFace','Stone',
    'FuseA','FuseF','SBrkr','BuiltIn','Detchd','N','Y','WD','Abnorml','Normal','Partial','LotArea','BsmtFinSF1',
    'BsmtUnfSF','TotalBsmtSF','X2ndFlrSF','KitchenAbvGr','WoodDeckSF','OpenPorchSF','EnclosedPorch',
    'ScreenPorch','ExterQual','BsmtQual','BsmtCond','GarageQual','PoolQC','CentralAir','Functional'
    ,'Fence','SalePrice']

#Log transform columns with high skew
for i in clean:
    if i in numericcols and i in skew:
        clean[i]=np.log(clean[i]+1)
        
clean.to_csv('finalclean.csv')


# In[30]:

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#Split training data into train and test 0.2% of training data into test
#local_trainX, local_testX = train_test_split(clean, test_size=0.2, random_state=123)
#local_trainY, local_testY = train_test_split(clean['SalePrice'], test_size=0.2, random_state=123)

trainX, testX, trainY, testY = train_test_split(clean.drop('SalePrice',axis=1), clean['SalePrice'])

#n_estimators: no of trees
#n_jobs: number of jobs to run in parallel -1: number of jobs=no of CPU cores
#Create a randomForestRegressor
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)

#Train random forest
randomForestResult = clf.fit(trainX, trainY)

#Predict for testX
predictions = randomForestResult.predict(testX) 

#Finding rmse
error = np.sqrt(((predictions - testY) ** 2).mean())
print('rmse error',error)

#Training for the whole train data
trainX = clean.drop(['SalePrice'], axis=1)
trainY = clean['SalePrice']
randomForestResult = clf.fit(trainX, trainY)


# In[33]:

#Combining different models
import pandas as pd
clean = pd.read_csv("corrandlogtrain.csv")
lassoresult = pd.read_csv("lassoOntest.csv")
xgboostresult = pd.read_csv("xgboostOntest.csv")
testData = pd.read_csv("corrandlogtest.csv")

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
print('Done1')
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)

#Get forest model
X_train = clean.drop(['SalePrice'], axis=1)
Y_train = clean['SalePrice']
randomForestResult = clf.fit(X_train, Y_train)
prediction=randomForestResult.predict(testData)


#Combining result from various models
#0.6-lasso ,0.4-xgboost - Bestresult
new_pred=0.6*lassoresult['SalePrice']+0.2*xgboostresult['SalePrice']+0.2*prediction
new_pred.to_csv('ResultLassoXgForest.csv')


# In[34]:

new_pred


# In[ ]:



