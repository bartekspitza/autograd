from src.tensor import Tensor
from src.nn import MLP
import numpy as np
from mnist import MNIST

mndata = MNIST('./mnist_dataset/')
mndata.gz = True
mnist_x_train, mnist_y_train = mndata.load_training()
mnist_x_test, mnist_y_test = mndata.load_testing()

x_train = Tensor(list(mnist_x_train))
x_test = Tensor(list(mnist_x_test))
y_train = Tensor(list(mnist_y_train))
y_test = Tensor(list(mnist_y_test))

rng = np.random.default_rng(seed=1)

# Scale down pixels from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# OneHot label vectors
def onehot(vector):
    tmp = []
    for scalar in vector.data:
        tmp2 = [1 if i==scalar else 0 for i in range(10)]
        tmp.append(tmp2)
    return Tensor(tmp)
y_train = onehot(y_train)
y_test = onehot(y_test)

# Multinomial sampling with replacement
def draw_batch(batch_size): 
    draw = lambda: int(rng.random() * x_train.shape[0])
    batch = [draw() for _ in range(batch_size)]
    return x_train[batch], y_train[batch]

# Loss function
def mle(x, y):
    y_pred = x*y
    maximum_likelihood = (y*y_pred).sum(axis=1)
    neg_log_loss = -1 * maximum_likelihood.log()
    return neg_log_loss.sum() 

nn = MLP(inputs=784, hidden=[200, 100, 50, 40, 30], outs=10)
epochs = 100
batch_size = 500
lr = 0.1

for e in range(epochs):
    x, y = draw_batch(batch_size)
    out = nn(x)
    loss = mle(out, y) / batch_size
    loss.backward()
    nn.train(lr=lr)
    print(f'Epoch {e}: {loss.data.item()}')

correct = 0
for i in range(len(x_test.data)):
    y = y_test[i]
    x = nn(x_test[i])
    corr = np.argmax(y.data)
    predicted = np.argmax(x.data)
    if predicted == corr: correct += 1

print(f'Test accuracy: {correct/len(x_test.data)}')


exit(0)