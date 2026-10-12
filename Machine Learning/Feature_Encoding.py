from pandas.conftest import observed
from sklearn.datasets import fetch_openml

data = fetch_openml(name='titanic', version=1, as_frame=True)

X, y = data['data'], data['target']

X.embarked.value_counts()

X.embarked.info()

X = X(X.embarked.notna())

#Label Encoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X['embarked_label_encoded'] = label_encoder.fit_transform(X['embarked'])
X.sample(5)

#Ordinal Encoder
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[['S', 'C', 'Q']])
X['embarked_ordinal_encoded'] = ordinal_encoder.fit_transform(X[['embarked']])
X.sample(5)


#One-Hot Encoding
import pandas as pd
X = X.join(pd.get_dummies(X['embarked'], dtype='int32', prefix='embarked'))
X.sample(5)


#Binary Encoding
import category_encoders as ce

binary_encoder = ce.BinaryEncoder()
binary_encoded = binary_encoder.fit_transform(X[['embarked']])

X = pd.concat([X, binary_encoded], axis=1)


#Frequency Encoding
freq_encoding = X['embarked'].value_counts(normalize=True)
X['embarked_freq_encoded'] = X['embarked'].map(freq_encoding)
X.sample(5)


#Target Encoding (Mean)
df = X.copy()
df['survived'] = y
df['survived'] = df['survived'].astype('int')

target_means = df.groupby('embarked', observed=True)['survived'].mean()

X['embarked_target_encoded'] = X['embarked'].map(target_means)
print(X)


#Embeddings
import torch

unique_values = 150 #with 150 unique value
embedding_dim = 8 # dimension of 8 --> our vector have 8 values

embedding = torch.nn.Embedding(unique_values, embedding_dim)
values = torch.LongTensor([1, 2, 3, 4, 6, 66, 140]) #take identify, but not 150 or higher
output = embedding(values) #embedding into vector space

print(output)
#result get 8 dimensional vectors and these are just positions in the 8 dimensional vector space
# --> of course, they are random but be trained so that these positions are actually optimal or useful at least

embarked_col = X['embarked']

embarked_mapping = {val: idx for idx, val in enumerate(embarked_col.dropna().unique())}
embarked_indices = embarked_col.map(embarked_mapping).fillna(0).astype(int)

embedding = torch.nn.Embedding(len(embarked_mapping), 8) #3 different value, 8 dimension ??? wtf

#take embarked column and put it into Vector space again, randomly initialized but can be trained to more useful
embarked_tensor = torch.LongTensor(embarked_indices)
embedded_output = embedding(embarked_tensor)

print(embedded_output)