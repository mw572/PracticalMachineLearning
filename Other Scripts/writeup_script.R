### PRACTICAL MACHINE LEARNING - ASSIGNMENT - 22/08/15 - Marcus Williamson ###

## clear up environment
ls()
rm(list=ls())


## import libraries
library(caret)
library(rpart)
library(doMC)
library(rpart.plot)

registerDoMC(cores = 8) # set cores for parallel processing

set.seed(8484) # set our seed for reproducibility


## importing our data from the source: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv, https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
rawtraindata = read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
rawtestdata = read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))


## initial look at our data
dim(rawtraindata);dim(rawtestdata)
View(rawtraindata);View(rawtestdata)


## removing obsolete variables from datasets
classe <- rawtraindata$classe # keeping our classe variables aside

names <- grepl("X|timestamp|window|name", colnames(rawtraindata)) # getting all non predictive variables column names

rawtraindata = rawtraindata[,!names] # removing columns
rawtestdata = rawtestdata[,!names] # repeating for test dataset


## futher cleansing of data incompatible with ML problems
rawtraindata = rawtraindata[ ,colSums(is.na(rawtraindata))==0] # if data is "NA" remove columnn
rawtestdata = rawtestdata[ ,colSums(is.na(rawtestdata))==0] # repeat for test dataset

traindata = rawtraindata[ ,sapply(rawtraindata,is.numeric)] # if data is non existent or non numeric remove column
testdata = rawtestdata[ ,sapply(rawtestdata,is.numeric)] # repeat for test dataset

traindata$classe = classe # adding our classe column back into the training dataset


## partitioning training data into training and validation datasets
inTrain <- createDataPartition(y=traindata$classe,p=0.70, list=FALSE) # using a 70:30 split

training <- traindata[inTrain,] # 70% of data for training
testing <- traindata[-inTrain,] # 30% of data for later validation

dim(training);dim(testing) # ensuring we have sufficient volumes for this 70:30 split

## creating our model
#We are using a Random Forest model with K fold Cross Validation with 10 folds , we use 300 trees in the model training

modFit <- train(classe ~ ., data=training,method="rf", trControl=trainControl(method="cv", number=10), verbose=FALSE, ntree=300, allowParallel=TRUE)
modFit # examining the model
modFit$finalModel

## cross looking at the predictive ability of our model
prediction <- predict(modFit, testing)
confusionMatrix(testing$classe, prediction) # printing error matrix

accuracy <- postResample(prediction, testing$classe) # calculating accuracy
ose <- 1 - as.numeric(confusionMatrix(testing$classe, prediction)$overall[1]) # calculating out of sample error

accuracy;ose # printing these values


## using the model to predict on the unseen data
results <- predict(modFit, testdata[, -length(names(testdata))])
results


## plotting random forest model output
plot(modFit$finalModel,main="Log of resampling results across tuning parameters", log="y")
finalmodel.legend <- if (is.null(modFit$finalModel$test$err.rate)) {colnames(modFit$finalModel$err.rate)} else {colnames(modFit$finalModel$test$err.rate)}
legend("top", cex =0.5, legend=finalmodel.legend, lty=c(1,2,3,4,5,6), col=c(1,2,3,4,5,6), horiz=T) # plotting rival model error rates from which the final model was selected

varImpPlot(modFit$finalModel, main="Importance Plot") # plotting the relative importance of the variables in the final model

treeout <- rpart(classe ~ ., data=training) # creating a single tree from the training data
prp(treeout,tweak=2.2) # plotting a single sample tree as Random Forest is black box algorithm, a single tree does not represent the model but is useful for sense checking data

