
new_data <- read.csv('clean.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
my_train <- new_data[1:1124,] 
my_test <- new_data[1124:1405,]
fit <- lm(data$SalePrice ~ data$X60+ data$NoRidge+ data$NridgHt + data$Exterior2ndVinylSd+ data$None+ data$Stone  + data$Attchd+ 
          data$Detchd + data$OverallQual + data$YearBuilt 
          + data$BsmtFinSF1 +  data$TotalBsmtSF + data$X2ndFlrSF + data$TotRmsAbvGrd + data$Fireplaces + data$GarageCars +
            data$WoodDeckSF + data$OpenPorchSF + data$ExterQual + data$BsmtQual + data$BsmtExposure + data$HeatingQC + data$BsmtFinType1 + data$GarageFinish 
          ,data=data) #r-sqaured = 0.807 - all above - 0.30


fit1 <- lm(my_data$SalePrice ~ my_data$NridgHt + my_data$OverallQual + my_data$YearBuilt + my_data$TotalBsmtSF +
             my_data$TotRmsAbvGrd + my_data$Fireplaces + my_data$GarageCars +
             my_data$ExterQual + my_data$BsmtQual + my_data$HeatingQC + my_data$GarageFinish , data = my_data) #r-sqaured = 0.7686 - all above - 0.40
#r-sqaured = 0.7686 - all above - 0.40
pred=predict.lm(fit1,data=my_test)
my_test2 <- my_test
my_test2$PRED <- pred
write.csv(pred, file = "components2.csv")