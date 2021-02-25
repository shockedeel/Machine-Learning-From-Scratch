df = read.csv("titanic_project.csv")
df$pclass <- as.factor(df$pclass)
df$survived <- as.factor(df$survived)
df$sex <- as.factor(df$sex)
train <- df[1:900,]
test <- df[900:nrow(df),]
library(caret)
library(e1071)
start <- Sys.time()
nbm <- naiveBayes(survived~pclass+sex+age,data=train)
end <- Sys.time()
nbm
pred <- predict(nbm, newdata = test, type="class")
paste("training time: ", end-start)
confusionMatrix(pred, test$survived, positive = "1")
