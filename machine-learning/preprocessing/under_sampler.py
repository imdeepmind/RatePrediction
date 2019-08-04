import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Dataset paths
paths = ['dataset/amazon_reviews_us_Musical_Instruments_v1_00.tsv', 
         'dataset/amazon_reviews_us_Office_Products_v1_00.tsv']

# Initializing RandomUnderSampler
rus = RandomUnderSampler(random_state=1969)

# Initialing arrays
reviews = []
ratings = []

# Looping through the paths
for path in paths:
    print('--Processing first dataset {}th--'.format(path))
    
    # Reading the dataset
    data = pd.read_csv(path, sep='\t', error_bad_lines=False)
    
    # Filtering the dataset
    data = data[data['verified_purchase'] == 'Y']
    
    # Selecting the two columns
    data = data[['review_body', 'star_rating']]
    
    # Dropping rows with nan values
    data = data.dropna()
    
    # Converting the dataframe into numpy arrays
    X = data['review_body'].values.reshape(-1,1)
    y = pd.to_numeric(data['star_rating']).values.reshape(-1,1)
    
    # Deleting data to save space
    del data
    
    print('--Sampling dataset--')
    
    # Sampiling
    X, y = rus.fit_resample(X,y)
    
    # Appending the data into arrays
    reviews = reviews + X.tolist()
    ratings = ratings + y.tolist()
    
    # Saving space
    del X, y
    
# Initializng a DataFrame
dataset = pd.DataFrame(columns=['reviews', 'ratings'])

# Looping through the ratings to convert into int
newRatings = []
for rate in ratings:
    newRatings.append(rate[0])
   
# Deleting old ratings array
del ratings

# DLooping through the reviews to convert into string
newReviews = []
for review in reviews:
    newReviews.append(review[0])
   
# Deleting old reviews array
del reviews
    
# Putting data into the DataFrame
dataset['reviews'] = newReviews
dataset['ratings'] = newRatings

# Deleting reviews and newRatings array
del newReviews, newRatings

# Shuffling the datset
dataset = dataset.sample(frac=1)

# Saving the data as a csv file
dataset.to_csv('dataset/dataset.csv', index=False)