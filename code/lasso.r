#lasso.r
#writes predicted values for test data into lasso.csv

require(glmnet)
require(hydroGOF)
require(caret)
library(e1071)

#cleaned training data
train=read.csv('corrandlogtrain.csv',stringsAsFactors=FALSE)
#cleaned testing data
test=read.csv('corrandlogtest.csv',stringsAsFactors=FALSE)
expminus1= function(x) {
  result=c()
  l=NROW(x)
  for(i in 0:l)
    result[i]=exp(x[i])-1
  result
}

#model for lasso. alpha=1 indicates that lasso is being used and not ridge
model=cv.glmnet(x=as.matrix(subset(train,select=-SalePrice)),y=train$SalePrice,family="gaussian",alpha=1)
pred=predict(model,newx=as.matrix(test),s="lambda.min")
pred=expminus1(pred)
write.csv(pred,"lasso.csv")


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
  #model using lasso
  mdl=	cv.glmnet(x=as.matrix(subset(train_data,select=-SalePrice)),y=train_data$SalePrice,family="gaussian",alpha=1)	 
  prd=predict(mdl,newx=as.matrix(subset(test_data,select = -SalePrice)),s="lambda.min")
  error[i]=rmse(prd[1:NROW(prd)],test_data$SalePrice)
  prd=expminus1(prd)
  error2[i]=rmse(prd[1:NROW(prd)],expminus1(test_data$SalePrice))
}
m=mean(error)
print(m)
m2=mean(error2)
print(m2)