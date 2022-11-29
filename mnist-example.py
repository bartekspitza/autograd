from src import nn
import numpy as np
from mnist import MNIST

mndata = MNIST('./mnist_dataset/')
mndata.gz = True
mnist_x_train, mnist_y_train = mndata.load_training()
mnist_x_test, mnist_y_test = mndata.load_testing()

x_train = nn.Tensor(list(mnist_x_train))
x_test = nn.Tensor(list(mnist_x_test))
y_train = nn.Tensor(list(mnist_y_train))
y_test = nn.Tensor(list(mnist_y_test))

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
    return nn.Tensor(tmp)
y_train = onehot(y_train)
y_test = onehot(y_test)

W = nn.Tensor(rng.normal(size=(784, 200)))
W2 = nn.Tensor(rng.normal(size=(200, 100)))
W3 = nn.Tensor(rng.normal(size=(100, 50)))
b = nn.Tensor(rng.normal(size=(200,)))
b2 = nn.Tensor(rng.normal(size=(100,)))
b3 = nn.Tensor(rng.normal(size=(50,)))
O = nn.Tensor(rng.normal(size=(50, 10)))
parameters = [W,W2,W3,b,b2,b3,O]

# Multinomial sampling with replacement
def draw_batch(batch_size): 
    draw = lambda: int(rng.random() * x_train.shape[0])
    batch = [draw() for _ in range(batch_size)]
    return x_train[batch], y_train[batch]

def forward(x):
    # L1
    x = b + (x@W) # l1
    x = x.tanh()
    x = b2 + (x@W2)
    x = x.tanh()
    x = b3 + (x@W3)
    x = x.tanh()
    # Output with softmax
    x = x@O
    x = x.exp()

    return x / x.sum(axis=x.dim-1).reshape((-1, 1))

epochs = 1000
batch_size = 20
lr = 0.01
for e in range(epochs):
    x, y = draw_batch(batch_size)

    # Forward
    out = forward(x)

    # Predictions
    y_pred = out*y
    
    # Loss
    maximum_likelihood = (y*y_pred).sum(axis=1)
    neg_log_loss = -1 * maximum_likelihood.log()
    nll = neg_log_loss.sum() 
    loss = nll / batch_size
    print(f'Epoch {e}: {loss.data.item()}')

    # Compute gradients
    loss.grad = np.array([1])
    loss.backward()
    
    # Learn
    for p in parameters:
        p.data -= lr * p.grad
        p.grad = np.zeros(p.shape)

correct = 0
for i in range(len(x_test.data)):
    y = y_test[i]
    x = forward(x_test[i])
    corr = np.argmax(y.data)
    predicted = np.argmax(x.data)
    if predicted == corr: correct += 1

print(f'Test accuracy: {correct/len(x_test.data)}')


exit(0)