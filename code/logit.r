#logistic reg.

df=read.table(file="clean_NEW.csv", header=TRUE, sep=",")
my_train <-df[1:1125,] 
my_test <- df[1125:1405,]
write.csv(my_test, "my_test_new.csv")
write.csv(my_train, "my_train_new.csv")
my_test1 <- read.csv('my_test_new.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
my_train1 <- read.csv('my_train_new.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)

fct=cut(df$SalePrice,2,include.lowest=FALSE,right=TRUE)

mdl1 <- glm(fct ~.,family=binomial(link='logit'),data=df)
summary(mdl1)

#two ranges
#0 is 34200$ to 395000$
#1 is 395000$ to 756000$

max(df$SalePrice)
try1=data.frame(df[651:651,])
predict(mdl1,try1,type="response")
#op = 1

min(df$SalePrice)
try2=data.frame(df[462:462,])
predict(mdl1,try2,type="response")
#op = 2.220446e-16 
 
try3=data.frame(df[825:825,])
predict(mdl1,try3,type="response")
#op is 2.220446e-16 because actual price is 236000$

write.csv(predict(mdl1,newdata=df,type="response"),file="logitOp.csv")





#k-cross validation

set.seed(250)
library("caret")
positions=createFolds(train$MSSubClass, k = 10, list = TRUE, returnTrain = TRUE)
error=c()
error2=c()
for(i in 1:10)
{
  train_data=df[unlist(positions[i]),]
  test_data=df[-unlist(positions[i]),]


fct=cut(df$SalePrice,2,include.lowest=FALSE,right=TRUE)

mdl1 <- glm(fct ~.,family=binomial(link='logit'),data=df)

#PREDICTION
  pred=predict(model,test_data,type="response")

#vector containing predictions

  error[i]=rmse(pred[1:NROW(pred)],test_data$SalePrice)
}

m=mean(error)
print(m) 


#another model 
#but rejected because AIC was greater than model mdl1 
mdl2 <- glm(fct ~ df$NoRidge+ df$NridgHt + df$Exterior2ndVinylSd+ df$None+ df$Stone  + df$Attchd+ 
          df$Detchd + cut(df$OverallQual,2) 
          + cut(df$BsmtFinSF1,2) +  cut(df$TotalBsmtSF,2), family=binomial(link='logit'),data=df)

summary(mdl2)

write.csv(predict(mdl6,newdata=df,type="response"),"logit2.csv")
#predict(mdl6,new2,type="response")
