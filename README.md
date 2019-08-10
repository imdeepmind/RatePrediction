# Rate Prediction using Amazon Review Dataset

Predicting star ratings using Amazon Review Dataset and LSTM Recurrent Neural Network.

## Table of contents:
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Future Targets](#future-targets)
- [Acknowledgments](#acknowledgments)

## Introduction

Millions of people use Amazon to buy products. On Amazon, for every product, people can rate and write a review. If a product is good, it gets a positive review and gets a higher star rating, similarly, if a product is bad, it gets a negative review and lower star rating. My aim in this project is to predict star rating automatically based on the product review.

In Amazon, the range of star rating is 1 to 5. That means if the product review is negative, then it will get low star rating (possibly 1 or 2), if the product is average then it will get medium star rating (possibly 3), and if the product is good, then it will get higher star rating (possibly 4 or 5).

** This project aims to make a system that automatically detects the star rating based on the review.**

This task is similar to Sentiment Analysis, but instead of predicting the positive and negative sentiment(sometimes neutral also), here I need to predict the star rating. 


## Dataset

For this project, I'm using the [Amazon Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html). Amazon Review Dataset is a gigantic collection of product reviews and their star rating. It contains more than 40 millions of reviews(I don't know the original number). 

> Downloading instructions and other information about the dataset can be found on the dataset website.  

In this case, I'm just using a tiny fraction of the dataset, more specifically, I'm using the following files.
- `amazon_reviews_us_Musical_Instruments_v1_00.tsv`
- `amazon_reviews_us_Office_Products_v1_00.tsv`
- `amazon_reviews_us_Music_v1_00`

The entire dataset is in `.tsv` format.

## Model

### 1. Step 1: Balancing the dataset - 
First of all, the dataset is unbalanced. In other words, there is more sample for one class than the other classes. The unbalanced dataset can cause several problems. To solve this problem, we need to balance the dataset. 

To solve this problem and balance the data, I'll use `UnderSampler`. To learn about `UnderSampler`, click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)

![Under Sampler](https://user-images.githubusercontent.com/34741145/62822814-2f0f2e00-bba6-11e9-8f04-f4ffde718066.png)

After using `Under Sampler` on the dataset, the class distribution becomes balanced, or in other words, there are an equal number of sample for each class

### Step 2: Word Tokenizing - 
We all know that Machine Learning algorithms are just some math equations that perform some math operations to do all the amazing things. As these algorithms are just some math equations, they can only deal with numbers. Here in this project, we are dealing with product reviews. 

Work Tokenizing is a process of converting these word reviews into numbers.

Here I'm using Keras Word Tokenizer. To learn more about Word Tokenizer, click [here](https://keras.io/preprocessing/text/).

### Step 3: DL Model - 
Finally, let's talk about the Deep Learning model. The model that I'm using here is a combination of CNN and LSTM recurrent neural network.

![Model](https://user-images.githubusercontent.com/34741145/62822813-2f0f2e00-bba6-11e9-8c03-36f2cabe3b45.png)

## Dependencies
Following are the dependencies of the project
- Keras
- Pandas
- Numpy
- ImBalance

## File Structure
There are a total of two folders, `demo` and `machine-learning`. 

The `demo` folder contains a demo app for the demo of this project. We can ignore it.

The `machine-learning` folder is the main part of the application. This folder contains two subfolders `model` and `preprocessing`.
The `model` folder contains the main model for this project. The `preprocessing` folder contains all the code for preprocessing the dataset.

## Future Targets
Currently the model in only 50% accurate. So I have a target to increase the accuracy to 75%.

## Acknowledgments
- [Amazon Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)
