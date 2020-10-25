# STOCK SENTIMENT ANALYSIS

We live in a world where we are constantly bombarded with social media feeds, tweets and news articles. This huge amount of data can be leveraged to predict people's sentiment towards a particular company or a stock.


The goal of this program is to use Natural Language Processing to convert words into numbers and train a LSTM network model to make predictions.

![](Understanding.jpg)

# HOW IT WORKS

It reads information about a stock from a csv file. A tokenizer then vectorizes a text corpus. This text is then used to train the model to predict whether a stock has positive or negative sentiment. (positive is represented by 1 and negative by 0).

A confusion matrix is then used to graph predictions and errors.

![](CM.jpg)

The output is then presented in the form of a plot.

![](W.jpeg)

# REQUIREMENTS

You will require the following libraries:

Pandas

numpy

matplotlib

seaborn

wordcloud

nltk

gensim

tensorflow

# THE MODEL 

```python
# Sequential Model
model = Sequential()

# embedding layer
model.add(Embedding(total_words, output_dim = 512))

# Bi-Directional RNN and LSTM
model.add(LSTM(256))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()
# train the model
model.fit(padded_train, y_train_cat, batch_size = 32, validation_split = 0.2, epochs = 2)
# make prediction
pred = model.predict(padded_test)
# make prediction
prediction = []
```
# CONTRIBUTION

Anyone is welcome to use this repository as they please and contribute however they like.

# LICENSE 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
