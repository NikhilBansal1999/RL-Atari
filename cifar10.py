from keras.datasets import cifar10
import numpy
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Activation,Flatten
from keras import optimizers
import keras
from keras import regularizers

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train/255
x_test=x_test/255
#x_train=x_train.reshape(50000,3072)
#x_test=x_test.reshape(10000,3072)
#(50000, 32, 32, 3)
#(10000, 32, 32, 3)
#(50000, 1)
#(10000, 1)

y_train=keras.utils.np_utils.to_categorical(y_train,10)
y_test=keras.utils.np_utils.to_categorical(y_test,10)

Classify = Sequential()

Classify.add(Conv2D(64,3,input_shape=(32,32,3)))
Classify.add(Activation('relu'))

Classify.add(Conv2D(32,3))
Classify.add(Activation('relu'))

Classify.add(Conv2D(16,3))
Classify.add(Activation('relu'))

Classify.add(MaxPooling2D(pool_size=(2,2)))
Classify.add(Dropout(0.1))

Classify.add(Flatten())

Classify.add(Dense(512))
Classify.add(Activation('relu'))
Classify.add(Dropout(0.2))

Classify.add(Dense(512))
Classify.add(Activation('relu'))
Classify.add(Dropout(0.2))

Classify.add(Dense(512))
Classify.add(Activation('relu'))
Classify.add(Dropout(0.2))

Classify.add(Dense(10))
Classify.add(Activation('softmax'))
Classify.add(Dropout(0.2))

optimize=optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
Classify.compile(optimizer=optimize, loss='categorical_crossentropy', metrics=['accuracy'])

Classify.load_weights('Cifar10.h5')
Classify.fit(x_train,y_train,epochs=1,batch_size=100,validation_data=(x_test,y_test),shuffle=True)

Classify.save_weights('Cifar10.h5')
score=Classify.evaluate(x_test,y_test,batch_size=100)
print(score)