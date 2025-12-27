import numpy as np
import matplotlib.pyplot as plt
import os

TRAIN_PATH = "../data/mnist_train.csv"
TEST_PATH = "../data/mnist_test.csv"
OUTPUT_DIR = "../outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


train_data = np.loadtxt(TRAIN_PATH, delimiter=",", skiprows=1)
test_data = np.loadtxt(TEST_PATH, delimiter=",", skiprows=1)

X_train = train_data[:, 1:] / 255.0
y_train = train_data[:, 0]

X_test = test_data[:, 1:] / 255.0
y_test = test_data[:, 0]


def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels.astype(int)] = 1
    return encoded


y_train_oh = one_hot_encode(y_train)
y_test_oh = one_hot_encode(y_test)

INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10

LEARNING_RATE = 0.05
EPOCHS = 30
BATCH_SIZE = 128

np.random.seed(42)

W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.01
b1 = np.zeros((1, HIDDEN_SIZE))
W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.01
b2 = np.zeros((1, OUTPUT_SIZE))


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A2, (X, Z1, A1)


def compute_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


def compute_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


def backward_propagation(cache, y_true, y_pred):
    X, Z1, A1 = cache
    m = X.shape[0]

    dZ2 = y_pred - y_true
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


losses = []
accuracies = []

n = X_train.shape[0]

for epoch in range(EPOCHS):
    perm = np.random.permutation(n)
    X_shuffled = X_train[perm]
    y_shuffled = y_train_oh[perm]

    for i in range(0, n, BATCH_SIZE):
        X_batch = X_shuffled[i:i+BATCH_SIZE]
        y_batch = y_shuffled[i:i+BATCH_SIZE]

        y_pred, cache = forward_propagation(X_batch)
        dW1, db1, dW2, db2 = backward_propagation(cache, y_batch, y_pred)

        W1 -= LEARNING_RATE * dW1
        b1 -= LEARNING_RATE * db1
        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2

    train_pred, _ = forward_propagation(X_train)
    loss = compute_loss(y_train_oh, train_pred)
    acc = compute_accuracy(y_train_oh, train_pred)

    losses.append(loss)
    accuracies.append(acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")


test_pred, _ = forward_propagation(X_test)
test_acc = compute_accuracy(y_test_oh, test_pred)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")


plt.plot(losses)
plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
plt.close()

plt.plot(accuracies)
plt.title("Training Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig(f"{OUTPUT_DIR}/accuracy_curve.png")
plt.close()
