from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train=x_train.reshape((60000,784))
x_test=x_test.reshape((10000,784))
Y_train=np.zeros((60000,10))
for i in range(60000):
	Y_train[i][int(y_train[i])]=1
	
y_train=Y_train
weights1=np.random.rand(784,110)
weights2=np.random.rand(110,10)

for i in range(20):
	XW1=x_train.dot(weights1)
	act2=1/(1+np.exp(-1*XW1))
	act2W2=act2.dot(weights2)
	pred_prob=1/(1+np.exp(-1*act2W2))
	loss=np.sum((pred_prob-y_train)**2)
	#pred_prob=1/(1+np.exp(-1*act2W2))
	grad2=(pred_prob-y_train).T.dot(act2).T
	#grad1=(pred_prob-y_train).dot(weights2.T)*act2*(1-act2).T.dot(x_train).T
	grad1=(pred_prob-y_train).dot(weights2.T)
	grad1=(grad1*act2*(1-act2)).T
	grad1=grad1.dot(x_train).T
	weights1=weights1-0.000002*grad1
	weights2=weights2-0.000002*grad2
	print(i,loss)
	
test_act2=1/(1+np.exp(-1*x_test.dot(weights1)))
test_pred=1/(1+np.exp(-1*test_act2.dot(weights2)))
prediction=np.zeros(10000)
for i in range(10000):
	maxind=0
	for j in range(1,10):
		if test_pred[i][j]>test_pred[i][maxind]:
			maxind=j
	prediction[i]=maxind
	
correct=np.sum(prediction==y_test)
print("CORRECT PREDICTIONS:",correct)
	
	
	