---
editor_options:
  markdown:
    wrap: 72
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Probability and Statistics

# Lab Assignment 1: Naive Bayes Classifier

### *Name1 Surname1, Name2 Surname2, Name3 Surname3*

## Introduction

During the past three weeks, you learned a couple of essential notions
and theorems, and one of the most important among them is the *Bayes
theorem*.

One of its applications is **Naive Bayes classifier**, which is a
probabilistic classifier whose aim is to determine which class some
observation probably belongs to by using the Bayes formula:
$$\mathsf{P}(\mathrm{class}\mid \mathrm{observation})=\frac{\mathsf{P}(\mathrm{observation}\mid\mathrm{class})\mathsf{P}(\mathrm{class})}{\mathsf{P}(\mathrm{observation})}$$

Under the strong independence assumption, one can calculate
$\mathsf{P}(\mathrm{observation} \mid \mathrm{class})$ as
$$\mathsf{P}(\mathrm{observation}) = \prod_{i=1}^{n} \mathsf{P}(\mathrm{feature_i}), \qquad \mathsf{P}(\mathrm{observation} \mid \mathrm{class}) = \prod_{i=1}^{n} \mathsf{P}(\mathrm{feature}_i \mid \mathrm{class}),$$
where $n$ is the total number of features describing a given
observation. Thus, $\mathsf{P}(\mathrm{class}|\mathrm{observation})$ now
can be calculated as

$$\mathsf{P}(\mathrm{class} \mid \mathrm{\mathrm{observation}}) = \mathsf{P}(\mathrm{class})\times \prod_{i=1}^{n}\frac{\mathsf{P}(\mathrm{feature}_i\mid \mathrm{class})}{\mathsf{P}(\mathrm{feature}_i)}\tag{1}$$

```{r}
library(tidytext)
library(readr)
library(dplyr)
library(ggplot2)
library(hash)
```

### Data pre-processing

```{r}
list.files(getwd())
list.files("data/4-spam")
```

```{r}
test_path <- "data/4-spam/test.csv"
train_path <- "data/4-spam/train.csv"

stop_words <- read_file("stop_words")
splitted_stop_words <- strsplit(stop_words, split='\r\n')
splitted_stop_words <- splitted_stop_words[[1]]
```

```{r}
test <-  read.csv(file = test_path, stringsAsFactors = FALSE)
```

```{r}
train <-  read.csv(file = train_path, stringsAsFactors = FALSE)
tidy_text <- unnest_tokens(train, 'splitted', 'Message', token="words") %>%
  filter(!splitted %in% splitted_stop_words)

```

## Classifier implementation and Data visualization

```{r}
naiveBayes <- setRefClass("naiveBayes",

       # here it would be wise to have some vars to store intermediate result
       # frequency dict etc. Though pay attention to bag of wards!
       fields = list(words="data.frame"),
       methods = list(
                    fit = function(X, y)
                    {
                          bag <<- tidy_text %>% count(splitted, sort=TRUE, Category)
                          words <<- tidy_text %>% count(splitted, sort=TRUE)
                          words$prob_spam = 0
                          for (i in 1:nrow(words)) {
                            word <<- words[i , 1]
                            number <<- words[i, 2]
                            spam_words <<- sum(bag$n[bag$Category=="spam" & bag$splitted == word])
                            ham_words <<- sum(bag$n[bag$Category=="ham" & bag$splitted == word])
                            words$prob_spam[words$splitted==word] <- spam_words / number
                          }
                          message_prob <<- train %>% count(Category)
                          message_ham_prob <<- message_prob[1, 2] / nrow(train)
                          message_spam_prob <<- message_prob[2, 2] / nrow(train)
                          return(words)
                    },

                    predict = function(message, words)
                    {
                      spam_prob = 1
                      ham_prob = 1
                      for(i in 1:nrow(message)) {
                        if (!identical(numeric(0), words$prob_spam[words$splitted == message[i, 2]])) {
                          if (words$prob_spam[words$splitted == message[i, 2]] != 0 &&
                                    words$prob_spam[words$splitted == message[i, 2]] != 1) {
                            spam_prob <- spam_prob * words$prob_spam[words$splitted == message[i, 2]]
                            ham_prob <- ham_prob * (1 - words$prob_spam[words$splitted == message[i, 2]])
                          }
                        }
                      }
                      spam_prob <- spam_prob * message_spam_prob
                      ham_prob <- ham_prob * message_ham_prob
                      if (spam_prob >= ham_prob) {
                        return("spam")
                      } else {
                        return("ham")
                      }
                    },

                    score = function(words)
                    {
                         succes_predict_test <- 0
                         unsucces_predict_test <- 0
                         for (i in 1:nrow(test)) {
                            message <- unnest_tokens(test[i, ], 'splitted', 'Message', token="words") %>%
                                filter(!splitted %in% splitted_stop_words)
                            if (nrow(message) != 0) {
                              r <- model$predict(message, words)
                              if (r == train[i, 1]) {
                                succes_predict_test <- succes_predict_test + 1
                              } else {
                                unsucces_predict_test <- unsucces_predict_test + 1
                              }
                            }

                         }

                           succes_predict_train <- 0
                           unsucces_predict_train <- 0
                           for (i in 1:nrow(train)) {
                              message <- unnest_tokens(train[i, ], 'splitted', 'Message', token="words") %>%
                                  filter(!splitted %in% splitted_stop_words)
                              if (nrow(message) != 0) {
                                r <- model$predict(message, words)
                                if (r == train[i, 1]) {
                                  succes_predict_train <- succes_predict_train + 1
                                } else {
                                  unsucces_predict_train <- unsucces_predict_train + 1
                                }
                              }

                           }
                          result <- c(succes_predict_test, unsucces_predict_test, succes_predict_train, unsucces_predict_train)
                          return(result)
                    },
                    visualize = function(succes_test, unsucces_test, succes_train, unsucces_train)
                    {

                      percent_test  <- succes_test/(succes_test + unsucces_test)
                      percent_success_test <- round(percent_test, 2)
                      percent_fail_test <- 1 - percent_success_test
                      percent_success_test <- paste(as.character(percent_success_test * 100), "%", sep="")
                      percent_fail_test <- paste(as.character(percent_fail_test * 100), "%", sep="")

                      percent_train  <- succes_train/(succes_train + unsucces_train)
                      percent_success_train <- round(percent_train, 2)
                      percent_fail_train <- 1 - percent_success_train
                      percent_success_train <- paste(as.character(percent_success_train * 100), "%", sep="")
                      percent_fail_train <- paste(as.character(percent_fail_train * 100), "%", sep="")

                      # Create the input vectors.
                      colors <- c("green","red")
                      percent_result <- c(percent_success_train, percent_fail_train, percent_success_test, percent_fail_test)
                      regions <- c("Successful","Unsuccessful")

                      png(file = "ResultsDistribution.png")
                      barplot(c(succes_train, unsucces_train, succes_test, unsucces_test), main = "Distribution of results by number of messages",
                              names.arg = percent_result, xlab = "Success Rate", font = 1.5, ylab = "Number of messages", col = colors, ylim=c(0, 4000))
                      # Add the legend to the chart
                      legend("topright", regions, cex = 1.3, fill = colors)

                      dev.off()



                      # Create the data for the chart
                      c <- c(succes_train, unsucces_train, succes_test, unsucces_test)

                      # Give the chart file a name
                      png(file = "F1-Measure.png")

                      #Precision = TruePositives / (TruePositives + FalsePositives)
                      #Recall = TruePositives / (TruePositives + FalseNegatives)
                      Precision1 <- c[1]/(c[1] + c[2])
                      Recall1 <- c[1]/(c[1] + 0)
                      Precision2 <- c[3]/(c[3] + c[4])
                      Recall2 <- c[3]/(c[3] + 0)

                      #F-Measure = (2 * Precision * Recall) / (Precision + Recall)
                      F_Measure1 <- (2 * Precision1 * Recall1)/(Precision1  + Recall1)
                      F_Measure2 <- (2 * Precision2 * Recall2)/(Precision2 + Recall2)
                      F_Measure1 <- round(F_Measure1, 3)
                      F_Measure2 <- round(F_Measure2, 3)

                      H <- c(F_Measure1, F_Measure2)
                      M <- c(c[1] + c[2], c[3] + c[4])

                      F1 <- as.character(F_Measure1)
                      F2 <- as.character(F_Measure2)
                      F1 <- paste("Train -", F1)
                      F2 <- paste("Test - ", F2)
                      # Plot the bar chart
                      barplot(H,names.arg=M,xlab="Number of messages",ylab="F1-measure",col=c("blue", "orange"),
                        main="F1-Measure (a combination of recall and precision)",border="brown",ylim=c(0, 1.2))
                      legend("topright", c(F1, F2), cex = 1.3, fill = c("blue", "orange"))
                      # Save the file
                      dev.off()
                    }
))
model = naiveBayes()
words <- model$fit()
res <- model$score(words)
model$visualize(res[1], res[2], res[3], res[4])
```

## Measure effectiveness of your classifier

First Bar Chart represents successful and unsuccessful predictions for train.csv and test.csv, they are shown in the appropriate sequence (train.csv [successful = 3892/(3892 + 443) = 90%, unsuccessful = 443/(3892 + 443) = 10%]; test.csv[successful = 1017/(1017 + 218) = 82%, unsuccessful = 218/(1017 + 218) = 18%). The second Bar Chart represents the F1 score metric, which includes the calculations of precision and recall. The result confirms the previous graph, as She decreases if the precision or recall decreases. In the first graph, we saw a decrease in accuracy as well.

## Conclusions

To predict which class the current message belongs to, we used The Naive Bayes Classifier, which is one of the simplest classifiers, for every word of the message we found the percentage ratio that it belongs to the ham and spam class. Thus, a set of basic elements was created that influenced belonging to one of the classes. The final class was determined by calculating the resulting percentage of belonging to one of the classes. As can be seen, the training data set has the best prediction result because the test data contains words that are not present in the test data. Accordingly, we did not take them into account, it turned out to be the only lever for errors
