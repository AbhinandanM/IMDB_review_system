import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,Dropout,Dense,Embedding


data = pd.read_csv("/content/drive/MyDrive/imdb/IMDB Dataset.csv")
data.head()

"""# Data Visualization"""

data['review'][1]

data['sentiment'][1]

data['review'][3]

data['sentiment'][3]

data['sentiment'].replace('positive',1,inplace=True)
data['sentiment'].replace('negative',0,inplace=True)

data.head()


sns.countplot(x='sentiment',data=data)

sns.violinplot(x='sentiment',data=data)

"""# Data Preprocessing"""

columns = []
for i in data.columns:
  columns.append(i)
print(columns)

data.isnull().sum()

data.shape

data.describe()

data.info()

big = data['review'][0]
for i in data['review']:
  if len(i) > len(big):
    big = i
print(big)

data['review'][0]

small = data['review'][0]
for i in data['review']:
  if len(i) < len(small):
    small = i
print(small)

for i in range(len(data['review'])):
  if data["review"][i] == "Read the book, forget the movie!":
    print("The smallest review is at index ",i)

data.head(5)

"""# Text Preprocessing"""



x = data["review"]
y = data["sentiment"]

y

tokenizer = Tokenizer(10000,lower=True)
tokenizer.fit_on_texts(x)
x[1]

sequence = tokenizer.texts_to_sequences(x)
sequence[1]


encoded_sequence = pad_sequences(sequence,maxlen=200,padding='pre')
encoded_sequence[1]

final_input = encoded_sequence
final_output = np.array(y)

final_output


x_train, x_test, y_train, y_test = train_test_split(final_input,final_output, test_size=0.33, random_state=42)

print(x_train.shape,y_train.shape)


model = Sequential()
model.add(
    Embedding(
        input_dim=10000,
        output_dim=120,
        input_length=200,
    ),)
model.add(
    Bidirectional(LSTM(64, return_sequences=True))
)
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

prediction = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=2,batch_size=32)

plt.plot(prediction.history['loss'],label='train')
plt.plot(prediction.history['val_loss'],label='validation')
plt.legend(loc='best')
plt.show()

plt.plot(prediction.history['accuracy'],label='train')
plt.plot(prediction.history['val_accuracy'],label='validation')
plt.legend(loc='best')
plt.show()

model.save("imdb review.h5")

"""# Creating Custom review predictor bot"""

def imdb_bot(review):
  sentence_list = []
  sentence_list.append(review)
  sequence = tokenizer.texts_to_sequences(sentence_list)
  input = pad_sequences(sequence,maxlen=200,padding='pre')
  output = model.predict_classes(input)
  if output.all() == 1:
    print("The reviews are good you should watch this")
  else:
    print("The reviews are not quite good you should try another one")

imdb_bot("tees maar khan")

imdb_bot("All salman khan movies")

imdb_bot("justice leauge dark appokalips war")