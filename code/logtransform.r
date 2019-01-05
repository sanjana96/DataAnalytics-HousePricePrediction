library(e1071)
require(dummies)
require(caret)
require(glmnet)
require(hydroGOF)

train <- read.csv("train.csv", stringsAsFactors=FALSE)
test <- read.csv("test.csv", stringsAsFactors=FALSE)

#log transform of saleprice
log1plusx = function(x) {
  result=c()
  l=NROW(x)
  for(i in 0:l)
    result[i]=log(1+x[i])
  result
}
train$SalePrice=log1plusx(train$SalePrice)

#find numeric features
nums <- sapply(train, is.numeric)
#find skewness of each feature
skewed_features <- sapply(train[,nums],function(x)
  {
    abs(skewness(x,na.rm=TRUE))  
})
#find features with skew>0.75
skewed_features=skewed_features[skewed_features>0.75]
sf=names(skewed_features)
#log transform selected features
for(i in sf)
{
  train[,i]=log1plusx(train[,i])
  test[,i]=log1plusx(test[,i])
}  

write.csv(train,'logtransformedtrain.csv')
write.csv(test,'logtransformedtest.csv')
