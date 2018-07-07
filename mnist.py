#test set accuracy of 97.7% and train set accuracy of 99.3%
from keras.layers import Dense,Activation,Dropout
from keras.datasets import mnist
from keras.models import Sequential
import keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train/255
x_test=x_test/255
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

y_train=keras.utils.np_utils.to_categorical(y_train,10)
y_test=keras.utils.np_utils.to_categorical(y_test,10)

model=Sequential()

model.add(Dense(250,input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=1000)
score=model.evaluate(x_test,y_test)
print(score)