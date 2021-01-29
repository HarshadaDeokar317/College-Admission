
#Loading dataset

CA <- read.csv("C:/Users/Harshada/Data/College_admission.csv")
View(CA)


#Check if any missing value
result.mean<-mean(CA)
print(result.mean)

#Second way
is.na(CA)

#Finding structure of dataset

str(CA)

#Factoring
f<-factor(c(CA$gpa))
as.numeric(f)
View(CA)


#Ploting
barplot(table(CA$gre))
barplot(table(CA$gpa))

#Outliers detection using histogram

hist(CA$gre,xlab = "gre",main = "Histogram of gre",breaks = sqrt(nrow(CA)))

#or using ggplot
library(ggplot2)
ggplot(CA) + aes(x=gre) + geom_histogram(bins=30L,fill="red")+ theme_minimal()

#Boxplots also useful to detect potential outliers
boxplot(CA$ses,ylab="ses")
boxplot(CA$admit,ylab="admit")

#To extract exact values of outliers

boxplot.stats(CA$gre)$out

#To extract row number corresponding to outliers

out <- boxplot.stats(CA$gre)$out
out_ind <- which(CA$gre %in% c(out))
out_ind

#Variables for this outliers
CA[out_ind,]

library(outliers)
SD<-CA[1:20,]

#For lowest outlier

test<-dixon.test(SD$gre)
test

#for Highest outlier

test<-dixon.test(SD$gre,opposite = TRUE)
test

#Visualization of outliers using boxplot

out <-boxplot.stats(SD$gre)$out
boxplot(SD$gre,ylab="gre")
mtext(paste("Outliers: ",paste(out,collapse = ",")))

#Normality distribution test

shapiro.test(CA$gre)

#here p value<0.05 hence data is not normaly distributed


#normalization using scale function
library(caret)
da<-as.data.frame(scale(CA[,2]))
summary(CA$gre)

#Reducing variables
library(olsrr)

model <-lm(admit~ gre + gpa + ses + Gender_Male + Race + rank,data = CA)
ols_step_all_possible(model)

#plot method shows fit criteria for all possible regression methods


model<-lm(admit~ gre + gpa + ses + Gender_Male + Race + rank,data = CA)
k <-ols_step_all_possible(model)
plot(k)

#Best subset regression

#select best subset of predictors such as having largest R2 value or smallest MSE

model <-lm(admit~ gre + gpa + ses + Gender_Male + Race + rank,data=CA)
ols_step_best_subset(model)


# plot for best subset regression

model<-lm(admit~ gre+ gpa + ses + Gender_Male + Race+ rank,data=CA)
k<-ols_step_best_subset(model)
plot(k)


#Variable selection

#stepwise forward regression

model<- lm(admit~.,data = CA)
ols_step_forward_p(model)

k<-ols_step_forward_p(model)
plot(k)

#Detailed output

ols_step_forward_p(model,details = TRUE)


#Logistic model

head(CA)
summary(CA)
sapply(CA, sd)



xtabs(~admit +rank,data=CA)


CA$rank<-factor(CA$rank)
CA$rank

CA_logit <-glm(admit~gre + gpa+rank ,data=CA,family = "binomial")
summary(CA_logit)


#Obtain confidence interval
#CIS using profiled log-likelihood

confint(CA_logit)

#CIS using standard errors

confint.default(CA_logit)


#Overall effect of rank using WALD test
library(aod)

#wald.test(b=coef(CA_logit),sigma=vcov(CA_logit),Terms=4:6)

#Odds ratio only

exp(coef(CA_logit))

#Odds ratio and 95% CI
exp(cbind(OR = coef(CA_logit), confint(CA_logit)))


#We will start by calculating the predicted probability of admission at each value of rank, holding gre and gpa at their means. First we create and view the data frame.

newdata1 <- with(CA, data.frame(gre = mean(gre), gpa = mean(gpa), rank = factor(1:4)))
newdata1


#In the below output we see that the predicted probability of being accepted into a graduate program is 0.52 for students from the highest prestige undergraduate institutions (rank=1), and 0.18 for students from the lowest ranked institutions (rank=4), holding gre and gpa at their means.

newdata1$rankP <- predict(CA_logit, newdata = newdata1, type = "response")
newdata1


#We can do something very similar to create a table of predicted probabilities varying the value of gre and rank. We are going to plot these, so we will create 100 values of gre between 200 and 800, at each value of rank (i.e., 1, 2, 3, and 4).



newdata2 <- with(CA, data.frame(gre = rep(seq(from = 200, to = 800, length.out = 100),
                                              4), gpa = mean(gpa), rank = factor(rep(1:4, each = 100))))
newdata2



#The code to generate the predicted probabilities (the first line below) is the same as before, except we are also going to ask for standard errors so we can plot a confidence interval. We get the estimates on the link scale and back transform both the predicted values and confidence limits into probabilities.


newdata3 <- cbind(newdata2, predict(CA_logit, newdata = newdata2, type = "link",
                                    se = TRUE))
newdata3 <- within(newdata3, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})

## view first few rows of final dataset
head(newdata3)


#It can also be helpful to use graphs of predicted probabilities to understand and/or present the model. We will use the ggplot2 package for graphing. Below we make a plot with the predicted probabilities, and 95% confidence intervals.
library(ggplot2)
ggplot(newdata3, aes(x = gre, y = PredictedProb)) + geom_ribbon(aes(ymin = LL,
                                                                    ymax = UL, fill = rank), alpha = 0.2) + geom_line(aes(colour = rank),
                                                                                                                      size = 1)
#To find the difference in deviance for the two models (i.e., the test statistic) we can use the command:

with(CA_logit, null.deviance - deviance)


#The degrees of freedom for the difference between the two models is equal to the number of predictor variables in the mode, and can be obtained using:

with(CA_logit, df.null - df.residual)


# P value can be obtained using

with(CA_logit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

#The chi-square of 41.46 with 5 degrees of freedom and an associated p-value of less than 0.001 tells us that our model as a whole fits significantly better than an empty model. This is sometimes called a likelihood ratio test (the deviance residual is -2*log likelihood). To see the model's log likelihood, 

#7.58e-08=7.58*10^-8=0.0000000758

#Models log liklihood

logLik(CA_logit)

# checking accuracy of model
#Plot ROC

library(ROCR)
library(Metrics)

library(caret)
split<-createDataPartition(y=CA$admit,p=0.6,list = FALSE)
new_train <- CA[split]
new_test <- CA[split]

log_predict<-predict(CA_logit,newdata=CA,type="response")
log_predict<-ifelse(log_predict>0.5,1,0)
pr<-prediction(log_predict,CA$admit)
perf<-performance(pr,measure = "tpr",x.measure = "fpr")
plot(perf)
auc(CA$admit,log_predict)

# Our AUC score is 0.5833. In roc plot we always try to move up and top left
# corner .from this plot we can say the model is predicting more negative values incorrectly
# to move up increase our threshold value to 0.6 and check the performance.


#Confusion matrix 
confusionMatrix(table(predict(CA_logit,type="response")>=0.5,CA$admit==1))


#Plot confusion matrix
ctable<-as.table(matrix(c(254,97,19,30),nrow = 2,byrow = TRUE))
fourfoldplot(ctable,color = c("#CC6666","#99CC99"),conf.level = 0,margin = 1,main="Confusion Matrix")



#Decision Tree model

library(party)

# Create the input data frame.
input.dat <- CA[c(2:200),]

# Give the chart file a name.
png(file = "decision_tree.png")

# Create the tree.
output.tree <- ctree(
  admit ~ gre+ ses +Race+rank, 
  data = input.dat)

# Plot the tree.
plot(output.tree)

# Save the file.
dev.off()


#Decision tree plot

library(rpart)
library(rpart.plot)
fit<-rpart(admit~.,data=CA,method='class')
rpart.plot(fit,extra=106)

#Confusion matrix
pu<-predict(fit,CA,type='class')
tm<-table(CA$admit,pu)
tm
#Accuracy of decision tree model

AC<-sum(diag(tm))/sum(tm)
paste('Accuracy for test',AC)

library(party)
library(randomForest)


#RandomForest model

# Create the forest.
output.forest <- randomForest(admit~ gre+ ses + Race + rank, 
                              data = input.dat)

# View the forest results.
print(output.forest) 

# Importance of each predictor.
importance(output.forest,type=2)
print(importance(output.forest,type = 2)) 
plot(output.forest)


#Categorize gre attribute into Low,Medium and High class


library(dplyr)

CA<-CA%>%
  mutate(gre_class=case_when(
    gre<400~"Low",
    gre>440&gre<580~"Medium",
    gre>580~"High"
  ))
CA

CA$gre_class=factor(CA$gre_class,levels = c("Low","Medium","High"))
XT=xtabs(~gre_class+gre,data = CA)
XT

#Count levels in gre
nlevels(CA$gre_class)

#Count no of elements in each levels in gre
count(CA,CA$gre_class)


#Random forest gives better result out of all these models.

