

library(randomForest)
library(mclust)
library(fastICA)
library(RandPro)
library(liver)
library(dplyr)
library(titanic)
library(neuralnet)

data("USArrests")   
set.seed(123)

k_means <- kmeans(USArrests, 3)
print(k_means)

combined_date <- cbind(USArrests, cluster = k_means$cluster)
head(combined_date)

fit <- Mclust(USArrests)
summary(fit)
plot(fit, what = "BIC")

data.frame(fit$classification)

### for iris dataset

k_means <- kmeans(iris[,1:4], 3)
print(k_means)

combined_date <- cbind(iris, cluster = k_means$cluster)
head(combined_date)


fit <- Mclust(iris[,1:4],G = 3)
summary(fit)
plot(fit, what = "BIC")

data.frame(iris$Species, fit$classification)

plot(fit, what = "classification")

#### pca

pca_fit <- summary(prcomp(USArrests, center = TRUE,scale. = TRUE))
str(pca_fit)

pca_fit2 <- summary(prcomp(iris[,1:4], center = TRUE,scale. = TRUE))
str(pca_fit2)



#### ica
ica_fit <- fastICA(USArrests, n.comp = 4, 
        tol = 1e-04, verbose = FALSE,
        w.init = NULL)


par(mfcol = c(2, 2))
plot(1:50, ica_fit$S[,1], type = "l", xlab = "S'1", ylab = "")
plot(1:50, ica_fit$S[,2], type = "l", xlab = "S'2", ylab = "")
plot(1:50, ica_fit$S[,3], type = "l", xlab = "S'2", ylab = "")
plot(1:50, ica_fit$S[,4], type = "l", xlab = "S'2", ylab = "")

ica_fit2 <- fastICA(iris[,1:4], n.comp = 3
                  )
par(mfcol = c(1, 3))
plot(1:150, ica_fit2$S[,1], type = "l", xlab = "S'1", ylab = "")
plot(1:150, ica_fit2$S[,2], type = "l", xlab = "S'2", ylab = "")
plot(1:150, ica_fit2$S[,3], type = "l", xlab = "S'2", ylab = "")



## rerun 

pca_fit <- summary(prcomp(USArrests[,1:2], center = TRUE,scale. = TRUE))
pca_fit

pca_fit2 <- summary(prcomp(iris[,1:2], center = TRUE,scale. = TRUE))
pca_fit2

### random projection

set.seed(123)
sample <- sample.int(n = nrow(USArrests), size = floor(.7*nrow(USArrests)), replace = FALSE)
trainn <- USArrests[sample, ]
testt <- USArrests[-sample,]
#Extract the train label and test label
trainl <- as.factor(rownames(trainn))
testl <- as.factor(rownames(testt))
typeof(trainl)

names(trainn) <- NULL
names(testt) <- NULL
#classify the Iris data with default K-NN Classifier.
res <- classify(trainn,testt,trainl,testl)

### dataset 2
data("iris")
#Split the data into training set and test set of 75:25 ratio.
set.seed(123)
sample <- sample.int(n = nrow(iris[,1:2]), size = floor(.75*nrow(iris[,1:2])), replace = FALSE)
trainn <- iris[sample, 1:2]
testt <- iris[-sample,1:2]
#Extract the train label and test label
trainl <- trainn$Species
testl <- testt$Species
typeof(trainl)
#Remove the label from training set and test set
trainn <- trainn[,1:4]
testt <- testt[,1:4]
#classify the Iris data with default K-NN Classifier.
res <- classify(trainn,testt,trainl,testl)
res
### last one 

random_forest <- randomForest(
   Species ~ .,
  data=iris
)
summary(random_forest)
imp = importance(random_forest)
mp <- data.frame(predictors=rownames(imp),imp)

importance(random_forest)
varImpPlot(random_forest)

random_forest2 <- randomForest(data = USArrests, Rape ~.)
summary(random_forest2)
imp2 = importance(random_forest2, type=2)
mp2 <- data.frame(predictors=rownames(imp2),imp2)
mp2
importance(random_forest2)
varImpPlot(random_forest2)


### Problem 3

fit <- Mclust(iris,G = 3)
summary(fit)
plot(fit, what = "BIC")

data.frame(iris$Species, fit$classification)

plot(fit, what = "classification")

k_means <- kmeans(iris[,1:2], 3)
print(k_means)

combined_date <- cbind(iris, cluster = k_means$cluster)
data.frame(iris$Species, combined_date$cluster)


### problem 4


### clustering check
titanic <- titanic_train
fit <- Mclust(na.omit(titanic[,c(5,6,7,8,10,12)]),G = 2)
summary(fit)
check_d1 <- data.frame(true = na.omit(titanic)$Survived, fit = fit$classification)

check_d1$comp <- ifelse(check_d1$true == 0, 2, 1)
sum(check_d1$fit == check_d1$comp)/length(check_d1$true)


k_means <- kmeans(na.omit(titanic[,c(6,7,8,10)]), 2)

check_d2 <- data.frame(true = na.omit(titanic)$Survived, fit = k_means$cluster)

check_d2$comp <- ifelse(check_d2$true == 0, 2, 1)
sum(check_d2$fit == check_d2$comp)/length(check_d2$true)


### dimension deduction

pca_fit <- summary(prcomp(na.omit(titanic[,c(10,6,8,7)]), center = TRUE,scale. = TRUE))
pca_fit

f2 <- randomForest(data = na.omit(titanic[,c(2,10,6,8,7)]), Survived ~.)

importance(f2)
varImpPlot(f2)
### neural network
data = na.omit(titanic[,c(2,10,6,8,7)])
max = apply(data , 2 , max)
min = apply(data, 2 , min)
scaled = as.data.frame(scale(data, center = min, scale = max - min))

samplesize = 0.60 * nrow(data)
set.seed(80)
index = sample( seq_len ( nrow ( data ) ), size = samplesize )

trainNN = data[index , ]
testNN = data[-index , ]

# fit neural network
set.seed(2)
NN = neuralnet(Survived ~ ., trainNN, hicombined_dateen = 3 , linear.output = T )

# plot neural network
plot(NN)


predict_testNN = predict(NN, testNN)
predictor <- ifelse(predict_testNN > 0.5, 1, 0)
compare_table <- table(predictor, testNN$Survived)


### NN after selecting the dimension deduction


data = na.omit(titanic[,c(2,10,6)])
max = apply(data , 2 , max)
min = apply(data, 2 , min)
scaled = as.data.frame(scale(data, center = min, scale = max - min))

samplesize = 0.7 * nrow(data)
set.seed(123)
index = sample( seq_len ( nrow ( data ) ), size = samplesize )

trainNN = data[index , ]
testNN = data[-index , ]

# fit neural network
set.seed(123)
NN = neuralnet(Survived ~ ., trainNN, hicombined_dateen = 3 , linear.output = T )

# plot neural network
plot(NN)


predict_testNN = predict(NN, testNN)
predictor <- ifelse(predict_testNN > 0.5, 1, 0)
compare_table <- table(predictor, testNN$Survived)
(compare_table[1,1]+compare_table[2,2])/sum(compare_table)
