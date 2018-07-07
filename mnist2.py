#test set accuracy of 98.7% and train set accuracy of 99.1%
from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.datasets import mnist
from keras.models import Sequential
import keras
from keras import backend as K
K.set_image_dim_ordering('th')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train/255
x_test=x_test/255
x_train = x_train.reshape(60000, 1, 28, 28).astype('float64')
x_test = x_test.reshape(10000, 1, 28, 28).astype('float64')

y_train=keras.utils.np_utils.to_categorical(y_train,10)
y_test=keras.utils.np_utils.to_categorical(y_test,10)

model=Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(1,28,28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(250))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=1000)
score=model.evaluate(x_test,y_test)
print(score)