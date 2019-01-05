#xgboost.r

library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
require(hydroGOF)
library(e1071)

#cleaned training dataset
train=read.csv('corrandlogtrain.csv',stringsAsFactors = FALSE)
#modified testing dataset
test=read.csv('corrandlogtest.csv',stringsAsFactors = FALSE)

expminus1= function(x) {
  result=c()
  l=NROW(x)
  for(i in 0:l)
    result[i]=exp(x[i])-1
  result
}
#for predicting test_data
#creating model using xgboost. eta is 0.1 to prevent overfitting. Low value of eta implies high value of nrounds.
model=xgboost(data=data.matrix(subset(train,select=-SalePrice)), label = train$SalePrice,max.depth = 2,
            eta = 0.1, nthread = 2, nrounds = 100, objective = "reg:linear")
#predicting using model. Gives log of saleprice
pred=predict(model,data.matrix(test))
#Converting prediction to saleprice
pred=expminus1(pred)
#writing prediction into csv file
write.csv(pred,"xgboost.csv")

#to reproduce folds created
set.seed(250)
#create 10 folds for k-fold cross validation
#positions will contain indices of rows included in training data
positions=createFolds(train$SalePrice, k = 10, list = TRUE, returnTrain = TRUE)
#for error between log values
error=c()
#for error between actual and predicted
error2=c()
for(i in 1:10)
{
	#use k-1 folds for train data
  train_data=train[unlist(positions[i]),]
	#use in fold for test
  test_data=train[-unlist(positions[i]),]
  #model using xgboost
  mdl=xgboost(data=data.matrix(subset(train_data,select=-SalePrice)), label = train_data$SalePrice,max.depth = 2,
                 eta =0.1, nrounds = 100, objective = "reg:linear")			 
  prd=predict(mdl,as.matrix(subset(test_data,select = -SalePrice)))
  error[i]=rmse(prd[1:NROW(prd)],test_data$SalePrice)
  prd=expminus1(prd)
  error2[i]=rmse(prd[1:NROW(prd)],expminus1(test_data$SalePrice))
}
m=mean(error)
print(m)
m2=mean(error2)
print(m2)