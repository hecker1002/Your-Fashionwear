#!/usr/bin/env python
# coding: utf-8

# In[64]:


import tensorflow as tf
from tensorflow import keras

tf.__version__
keras.__version__


# In[67]:


fashion_mnist = keras.datasets.fashion_mnist

(X_train_full , Y_train_full),(X_test , Y_test) = fashion_mnist.load_data()


# In[68]:


X_train_full.shape
# the trianing datset contains 60,000 correctly labeled images of clothes in a image of 28x28 pixels (2D array)
Y_train_full.shape # 60,000 correct labels 


# In[69]:


# building a validation set from training dataset to choose best hyperparsaetr during gradient descent to finfd optimum weight
# and biases of each neuron wrt previous layer

# and also scaling the intensity into range 0-1 by dividing each intensity by 255.0

X_valid , X_train = X_train_full[:5000]/255.0 , X_train_full[5000:]/255.0 #features
Y_valid , Y_train = Y_train_full[:5000] , Y_train_full[5000:] #label(should be converted tonon-numeric)



# In[72]:


# all the categorical data (classes) possible as output .
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# In[74]:


# Building the ANN model using keras (not training , just building the MLP layer-by-layer with arguments = no of neurons in each layer 
# & activation function used for that layer to convert the iutput into meaningful output for next layer of neurons .)

# Here , model = Neural Network 

model = keras.models.Sequential() # keras already know we can only  build a sequential MLP with it .

# adding first layer
model.add(keras.layers.Flatten(input_shape=[28,28])) # this layer is bound to take a 2D array as input so flatten it into 1D array.

# adding 2nd layer (first hidden layer) 
model.add(keras.layers.Dense(300,activation="relu")) # Dense means - the weights of each neuron with its input neuron strongly connected 

#adding 3rd layer( second hidden layer)
model.add(keras.layers.Dense(100 , activation="relu"))

# addinfg last (output layer) => no of neurons = no of classes .
model.add(keras.layers.Dense(10 , activation="softmax")) # since last output layer works like logistic reg. so softmax reg.



# In[44]:


# recieve all info. about each layer of model (ANN)

model.summary() # dense have alrgest no of parameter and flattened layer(Inp) has 0 .


# In[46]:


# get a list of layers of model

model.layers # returns a list


# In[48]:


# get name of a specific layer
model.layers[1].name


# In[56]:


# specifically , capture a specific layer of model using its name .
model.get_layer('dense_2').name
hidden1 = model.layers[1]


# In[63]:


# weights and biases are initialized randomly before training of MLP(ANN)

# weights and biases for first hidden layer
weights , biases = hidden1.get_weights()

weights # 2D array of 784 rows(weight of 1 neuron corrsponding to all input 784 pixels) and 300 columns (300 neurons of 2nd layer) showing

biases.shape # 300 (for each neuron's input , randomly initialized at 0)


# In[79]:


# compiling the model - specifying the loss function and the method to minimize its loss(deviation from actual result) for each neuron in whole 


model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd" ,metrics=["accuracy"]) # loss function=spare_crossentr. and guses sgd to find min value of itsactiavtion and returns the best weights and biases for that neuron .


# In[83]:


# training the model on training data set(by giving X_feat , Y_label and epochs too) and finding accuracy  by giving validation set .

model.fit(X_train , Y_train , epochs=30 ,validation_data=(X_valid,Y_valid))


# In[84]:


# finding final accuracy of modell using test data

model.evaluate(X_test , Y_test)


# In[88]:


# predict probabily for new data instances

Y_class = model.predict(X_test[:3])


# predict proper classes for new data intances (classes with highes tproability corresponding to  these new data instances)
Y_class = model.predict_classes(X_test[:3])
Y_class


# In[ ]:




