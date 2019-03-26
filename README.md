
# Rate Prediction using Amazon Review Dataset

Predicting star ratings using Amazon Review Dataset and LSTM Recurrent Neural Network.

## Table of contents:
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Solution Architecture](#solution-architecture)
- [Training Performence](#traiing-performence)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Run it Locally](#run-it-locally)
- [Acknowledgments](#acknowledgments)

## Introduction

Millions of people use Amazon to buy products. On Amazon, in each and every product people can rate the product and write a review about the product. If a product is good, it gets a positive review and gets a higher star rating, similarly, if a product is bad, it gets a negative review and lower star rating. My aim in this project is to predict star rating automatically based on the product review.

In Amazon, the range of star rating is 1 to 5. That means if the project review is negative, then it will get low star rating (possibly 1 or 2), if the product is average then it will get medium star rating (possibly 3), and if the product is good, then it will get higher star rating (possibly 4 or 5).

This task is similar to Sentiment Analysis, but instead of predicting the positive and negative sentiment(sometimes neutral also), here I need to predict the star rating. 


## Dataset

For this project, I'm using the [Amazon Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html). Amazon Review Dataset is a gigantic collection of product reviews and their star rating. It contains more than 40 millions of reviews(I don't know the original number). 

> Downloading instructions and other information about the dataset can be found on the dataset website.  

In this case, I'm just using a tiny fraction of the dataset, more specifically, I'm using the following files.
- `amazon_reviews_us_Musical_Instruments_v1_00.tsv`
- `amazon_reviews_us_Office_Products_v1_00.tsv`

The entire dataset is in `.tsv` format.

## Solution Architecture

### 1. Step 1:
First of all, the dataset is unbalanced. In other words, there is more sample for one class than others. The unbalanced dataset can cause several problems. To solve this problem, we need to balance the dataset. 

![Unbalanced Dataset](https://user-images.githubusercontent.com/34741145/54993741-9312ca80-4fe8-11e9-87bb-576886d442c1.png)

Clearly, there are so many 5 star rated products. To solve this I'll use `UnderSampler`. To learn about `UnderSampler`, click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)

![UnderSampler](https://user-images.githubusercontent.com/34741145/54993890-f6046180-4fe8-11e9-872a-f1a716725e05.png)
After balancing the dataset, the class distribution becomes balanced, or in other words, there are equal number of sample for each class

![Balanced Dataset](https://user-images.githubusercontent.com/34741145/54994010-454a9200-4fe9-11e9-83f8-39cea9eff603.png)


## Training Performence
## Dependencies
## File Structure
## Run it Locally
## Acknowledgments
- [Amazon Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)

