import numpy as np

def sigmoid(x, deriv=False):
	if deriv == True:
		return x*(1-x)
	return 1/(1+np.exp(-x))

X = np.array([[0],
              [np.pi/6],
              [np.pi/4],
              [np.pi/3],
              [np.pi/2]])
    
# output dataset            
Y = np.array([[0],
			  [np.sin(np.pi/6)],
			  [np.sin(np.pi/4)],
			  [np.sin(np.pi/3)],
			  [np.sin(np.pi/2)]])

W1 = 2*np.random.random((1,50)) - 1
W2 = 2*np.random.random((50,1)) - 1

for i in range (0,200000):
	#forward propagation
	L1 = sigmoid(np.dot(X,W1))
	L2 = sigmoid(np.dot(L1,W2))

	#backward prop
	EL2 = Y - L2
	delta_EL2 = EL2*sigmoid(L2,True)
	W2 += np.dot(L1.T,delta_EL2)

	EL1 = np.dot(delta_EL2,W1)
	delta_EL1 = EL1*sigmoid(L1,True)
	W1 += np.dot(X.T,delta_EL1)



X = np.array([0])

L1 = sigmoid(np.dot(X,W1))
L2 = sigmoid(np.dot(L1,W2))
print L2

