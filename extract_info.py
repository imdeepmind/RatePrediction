# This script will remove all unnecessary columns and merge the datasets into one dataset

import pandas as pd
import os

COUNTER = 0
stars = []
dataFiles = ['dataset/amazon_reviews_us_Musical_Instruments_v1_00.tsv', 'dataset/amazon_reviews_us_Office_Products_v1_00.tsv']

path = 'dataset/reviews/'
if not os.path.exists(path):
    os.makedirs(path)

for f in dataFiles:
    data = pd.read_csv(f, sep='\t', error_bad_lines=False)

    data = data[data['verified_purchase'] == 'Y']

    data = data[['review_body', 'star_rating']]

    data = data.values

    for dt in data:
        with open('dataset/reviews/' + str(COUNTER+1) + '.txt', 'w') as file:
            # May need to add some filters
            if isinstance(dt[0], str):
                file.write(dt[0])
                stars.append(dt[1])
                COUNTER += 1
    
    del data


star_df = pd.DataFrame(columns=['id', 'star'])

star_df['id'] = range(1, len(stars)+1)
star_df['star'] = pd.Series(stars)

star_df.to_csv('dataset/meta.csv', index=False)
