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

All the terms on the right-hand side can be estimated from the data as
respective relative frequencies;\
see [this
site](https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/)
for more detailed explanations.

## Data description

There are 5 datasets uploaded on the cms.

To determine your variant, take your team number from the list of teams
on cms and take *mod 5* - this is the number of your data set.

-   **0 - authors** This data set consists of citations of three famous
    writers: Edgar Alan Poe, Mary Wollstonecraft Shelley and HP
    Lovecraft. The task with this data set is to classify a piece of
    text with the author who was more likely to write it.

-   **1 - discrimination** This data set consists of tweets that have
    discriminatory (sexism or racism) messages or of tweets that are of
    neutral mood. The task is to determine whether a given tweet has
    discriminatory mood or does not.

-   **2 - fake news** This data set contains data of American news: a
    headline and an abstract of the article. Each piece of news is
    classified as fake or credible. The task is to classify the news
    from test.csv as credible or fake.

-   **3 - sentiment** All the text messages contained in this data set
    are labeled with three sentiments: positive, neutral or negative.
    The task is to classify some text message as the one of positive
    mood, negative or neutral.

-   **4 - spam** This last data set contains SMS messages classified as
    spam or non-spam (ham in the data set). The task is to determine
    whether a given message is spam or non-spam.

Each data set consists of two files: *train.csv* and *test.csv*. The
first one you will need find the probabilities distributions for each of
the features, while the second one is needed for checking how well your
classifier works.

```{r}
library(tidytext)
library(readr)
library(dplyr)
library(ggplot2)
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
train <-  read.csv(file = train_path, stringsAsFactors = FALSE)
test <-  read.csv(file = test_path, stringsAsFactors = FALSE)
```

```{r}
# note the power functional features of R bring us! 
train <-  read.csv(file = train_path, stringsAsFactors = FALSE)
tidy_text <- unnest_tokens(train, 'splitted', 'Message', token="words") %>%
  filter(!splitted %in% splitted_stop_words)

```

### Data visualization

Each time you work with some data, you need to understand it before you
start processing it. R has very powerful tools to make nice plots and
visualization. Show what are the most common words for negative and
positive examples as a histogram, word cloud etc. Be creative!

## Classifier implementation

```{r}
naiveBayes <- setRefClass("naiveBayes",
                          
       # here it would be wise to have some vars to store intermediate result
       # frequency dict etc. Though pay attention to bag of wards! 
       fields = list(),
       methods = list(
                    # prepare your training data as X - bag of words for each of your
                    # messages and corresponding label for the message encoded as 0 or 1 
                    # (binary classification task)
                    fit = function(X, y)
                    {
                         bag <- tidy_text %>% count(splitted,sort=TRUE, Category)
                    },
                    
                    # return prediction for a single message 
                    predict = function(message)
                    {
                         # TODO
                    },
                    
                    # score you test set so to get the understanding how well you model
                    # works.
                    # look at f1 score or precision and recall
                    # visualize them 
                    # try how well your model generalizes to real world data! 
                    score = function(X_test, y_test)
                    {
                         # TODO
                    }
))

model = naiveBayes()
model$fit()
```

## Measure effectiveness of your classifier
-   Note that accuracy is not always a good metric for your classifier.
    Look at precision and recall curves, F1 score metric.
-   Visualize them.
-   Show failure cases.

## Conclusions

Summarize your work by explaining in a few sentences the points listed
below.

-   Describe the method implemented in general. Show what are
    mathematical foundations you are basing your solution on.
-   List pros and cons of the method. This should include the
    limitations of your method, all the assumption you make about the
    nature of your data etc.
