# This file remove all unnecessary columns and merge the datasets into one dataset

import pandas as pd

data1 = pd.read_csv('dataset/amazon_reviews_us_Musical_Instruments_v1_00.tsv', sep='\t', error_bad_lines=False)
data2 = pd.read_csv('dataset/amazon_reviews_us_Office_Products_v1_00.tsv', sep='\t', error_bad_lines=False)

data1 = data1[data1['verified_purchase'] == 'Y']
data2 = data2[data2['verified_purchase'] == 'Y']

data1 = data1[['review_body', 'star_rating']]
data2 = data2[['review_body', 'star_rating']]

data = pd.concat((data1, data2))

del data1, data2

data.columns = ['review', 'star']

data.to_csv('dataset/reviews.csv', index=False)
