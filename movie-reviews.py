#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.datasets import imdb


# In[3]:


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# In[4]:


word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)


# In[5]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# In[6]:


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[46]:


from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(6, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[47]:


from tensorflow.keras import optimizers

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[48]:


history = model.fit(x_train,
                   y_train,
                   epochs=4,
                   batch_size=512)


# In[49]:


results = model.evaluate(x_test, y_test)


# In[50]:


results

