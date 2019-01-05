library(caret)
library(data.table)
library(Boruta)
library(plyr)
library(dplyr)
library(pROC)

ID.VAR <- "Id"
TARGET.VAR <- "SalePrice"
sample.df <- read.csv('afterRemovingCorrs.csv', header = TRUE, sep = ",", stringsAsFactors=FALSE)
# extract only candidate feture names
candidate.features <- setdiff(names(sample.df),c(ID.VAR,TARGET.VAR))
data.type <- sapply(candidate.features,function(x){class(sample.df[[x]])})
table(data.type)
print(data.type)
# deterimine data types
explanatory.attributes <- setdiff(names(sample.df),c(ID.VAR,TARGET.VAR))
data.classes <- sapply(explanatory.attributes,function(x){class(sample.df[[x]])})

# categorize data types in the data set?
unique.classes <- unique(data.classes)

attr.data.types <- lapply(unique.classes,function(x){names(data.classes[data.classes==x])})
names(attr.data.types) <- unique.classes

# pull out the response variable
response <- sample.df$SalePrice

# remove identifier and response variables
sample.df <- sample.df[candidate.features]

# for numeric set missing values to -1 for purposes of the random forest run
for (x in attr.data.types$integer){
  sample.df[[x]][is.na(sample.df[[x]])] <- -1
}

for (x in attr.data.types$character){
  sample.df[[x]][is.na(sample.df[[x]])] <- "*MISSING*"
}
set.seed(13)
bor.results <- Boruta(sample.df,response,
                      maxRuns=101,
                      doTrace=0)

print(bor.results)
final.boruta <- TentativeRoughFix(bor.results)
print(final.boruta)

boruta.df <- attStats(final.boruta)