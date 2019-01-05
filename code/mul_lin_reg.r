#after removing outliers and the attributes that are not relevant in building the model, data with the independent attributes used to build the multiple linear regression model is saved in clean_NEW.csv file

#Case 1: 80% train and 20% test
new_data <- read.csv(file="clean_NEW.csv", header = TRUE, sep = ",", stringsAsFactors=FALSE)
#test_data <-read.csv(file="cleanTestdata.csv", header = TRUE, sep = ",", stringsAsFactors=FALSE)

my_train <- new_data[1:1125,] 
my_test <- new_data[1125:1405,]
write.csv(my_test, "my_test_new.csv")
write.csv(my_train, "my_train_new.csv")
my_test1 <- read.csv('my_test_new.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
my_train1 <- read.csv('my_train_new.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
fit1 <- lm(my_train1$SalePrice ~ ., data = my_train1)
#fit1 <- lm(new_data$SalePrice ~ ., data = new_data)


summary(fit1)
pred <- predict.lm(fit1, newdata=my_test1)
write.csv(pred, file="prediction1.csv")

library("Metrics")
rmse(pred,my_test1$SalePrice)

library("hydroGOF")
nrmse(pred,my_test1$SalePrice)




#Case 2: 50% train and 50% test
test_data <-read.csv(file="cleanTestdata.csv", header = TRUE, sep = ",", stringsAsFactors=FALSE)

new_data <- read.csv(file="clean_NEW.csv", header = TRUE, sep = ",", stringsAsFactors=FALSE)
my_train <- new_data[1:710,] 
my_test <- new_data[710:1405,]
write.csv(my_test, "my_test_new.csv")
write.csv(my_train, "my_train_new.csv")
my_test1 <- read.csv('my_test_new.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
my_train1 <- read.csv('my_train_new.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
fit2 <- lm(my_train1$SalePrice ~ ., data = my_train1)
#fit2 <- lm(new_data$SalePrice ~ ., data = new_data)
summary(fit2)
#pred<-predict.lm(fit2, newdata=test_data)

pred <- predict.lm(fit2, newdata=my_test1)
write.csv(pred, file="prediction2.csv")

#write.csv(pred, file="new_prediction2.csv")

library("Metrics")
rmse(pred,my_test1$SalePrice)

library("hydroGOF")
nrmse(pred,my_test1$SalePrice)



#Case 3:building using log transformed data 

log1plusx = function(x) {
  result=c()
  l=NROW(x)
  for(i in 0:l)
    result[i]=log(1+x[i])
  result
}
train3 <- read.csv(file="clean_NEW.csv", header = TRUE, sep = ",", stringsAsFactors=FALSE)

#train3 <- read.csv(file="cleanTestdata.csv", header = TRUE, sep = ",", stringsAsFactors=FALSE)


#train3$SalePrice=log1plusx(train3$SalePrice)

#find numeric features
nums <- sapply(train3, is.numeric)
#find skewness of each feature
library("moments")
skewed_features <- sapply(train3[,nums],function(x)
{
  abs(skewness(x,na.rm=TRUE))  
})
#find features with skew>0.75
skewed_features=skewed_features[skewed_features>0.75]
sf=names(skewed_features)
#log transform selected features
for(i in sf)
{
  train3[,i]=log1plusx(train3[,i])
 # test3[,i]=log1plusx(test3[,i])
}  
write.csv(train3, file="c:/Users/Johny_Mathew/Desktop/5thSEM/DataAnalytics/DAProject/log_clean.csv")
#write.csv(train3, file="c:/Users/Johny_Mathew/Desktop/5thSEM/DataAnalytics/DAProject/log_clean_test.csv")


#new_data <- read.csv(file="log_clean_test.csv", header = TRUE, sep = ",", stringsAsFactors=FALSE)

new_data <- read.csv(file="log_clean.csv", header = TRUE, sep = ",", stringsAsFactors=FALSE)
#my_train<-new_data
my_train <- new_data[1:710,] 
my_test <- new_data[710:1405,]
write.csv(my_test, "my_test_new.csv")
write.csv(my_train, "my_train_new.csv")
my_test1 <- read.csv('my_test_new.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
my_train1 <- read.csv('my_train_new.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
fit3 <- lm(my_train1$SalePrice ~ ., data = my_train1)
summary(fit3)
#pred <- predict.lm(fit3, newdata=new_data)
pred <- predict.lm(fit3, newdata=my_test1)
write.csv(pred, file="prediction3.csv")

library("Metrics")
rmse(pred,my_test1$SalePrice)

library("hydroGOF")
nrmse(pred,my_test1$SalePrice)


expminus1= function(x) {
  result=c()
  l=NROW(x)
  for(i in 0:l)
    result[i]=exp(x[i])-1
  result
}

pred_rmvLog = expminus1(pred)
my_test_rmvLog= expminus1(my_test1$SalePrice)

library("Metrics")
rmse(pred_rmvLog,my_test_rmvLog)

library("hydroGOF")
nrmse(pred_rmvLog,my_test_rmvLog)



#k-cross validation

set.seed(250)
library("caret")
positions=createFolds(train$MSSubClass, k = 10, list = TRUE, returnTrain = TRUE)
error=c()
error2=c()
for(i in 1:10)my
{
  train_data=new_data[unlist(positions[i]),]
  test_data=new_data[-unlist(positions[i]),]

#MODEL
  model=lm(train_data$SalePrice ~ ., data = train_data)
	
#insert model here. pass train_data

#PREDICTION
  pred=predict.lm(model, newdata=test_data)

#vector containing predictions

  error[i]=rmse(pred[1:NROW(pred)],test_data$SalePrice)

  #not req. for mdl2
  pred2=expminus1(pred)
  error2[i]=rmse(pred2[1:NROW(pred2)],expminus1(test_data$SalePrice))
}

m=mean(error)
print(m) #rmse of log transformed data
m2=mean(error2) #rmse of actual values
print(m2)


"""
 m=mean(error)
> print(m) #rmse of log transformed data
[1] 3.956209
> m2=mean(error2) #rmse of actual values(log transformed data mdl)

> print(m2)
[1] 123397.8

for mdl2
 6850430

"""
