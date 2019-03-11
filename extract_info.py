# This script will remove all unnecessary columns and merge the datasets into one dataset

import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler

COUNTER = 0
stars = []
dataFiles = ['dataset/amazon_reviews_us_Musical_Instruments_v1_00.tsv', 'dataset/amazon_reviews_us_Office_Products_v1_00.tsv']

rus = RandomUnderSampler(random_state=1969)

path = 'dataset/reviewstxt/'
if not os.path.exists(path):
    os.makedirs(path)

for f in dataFiles:
    data = pd.read_csv(f, sep='\t', error_bad_lines=False)

    data = data[data['verified_purchase'] == 'Y']

    X = data['review_body'].values.reshape(-1,1)
    y = pd.to_numeric(data['star_rating']).values.reshape(-1,1)

    X, y = rus.fit_resample(X,y)

    for i in range(len(X)):
        with open(path + str(COUNTER+1) + '.txt', 'w') as file:
            # May need to add some filters
            if isinstance(X[i][0], str):
                file.write(X[i][0])
                stars.append(y[i][0])
                COUNTER += 1

    del data, X, y

star_df = pd.DataFrame(columns=['id', 'star'])

star_df['id'] = range(1, len(stars)+1)
star_df['star'] = pd.Series(stars)

staf_df = star_df.sample(frac=1)

star_df.to_csv('dataset/meta.csv', index=False)
