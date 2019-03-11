import pandas as pd

data = pd.read_csv('dataset/meta.csv')
star = data['star']

ratings = pd.unique(star)
allRates = []

for rate in ratings:
    allRates.append(len(star[star == rate]))

minReviewIndex = allRates.index(min(allRates))
minReviewValue = allRates[minReviewIndex]

onestar = data[data['star'] == 1.0]
twostar = data[data['star'] == 2.0]
threestar = data[data['star'] == 3.0]
fourstar = data[data['star'] == 4.0]
fivestar = data[data['star'] == 5.0]

onestar = onestar[0:minReviewValue]
twostar = twostar[0:minReviewValue]
threestar = threestar[0:minReviewValue]
fourstar = fourstar[0:minReviewValue]
fivestar = fivestar[0:minReviewValue]

newData = pd.concat((onestar, twostar, threestar, fourstar, fivestar))

newData = newData.sample(frac=1)

newData.to_csv('dataset/processedMeta.csv', index=False)
