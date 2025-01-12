from cnn import Conv, MaxPool, Softmax
import numpy as np
import matplotlib.pyplot as plt
import struct
import time


def loadImages(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
        return images
    
def loadLabels(filename):
    with open(filename, 'rb') as file:
        magic, num = struct.unpack('>II', file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)
        return labels
    
trainImages = loadImages('train-images.idx3-ubyte')
trainLabels = loadLabels('train-labels.idx1-ubyte')
testImages = loadImages('t10k-images.idx3-ubyte')
testLabels = loadLabels('t10k-labels.idx1-ubyte')

conv = Conv(10)
pool = MaxPool()
sfmx = Softmax(13 * 13 * 10, 10)

epoch = 5
learnRate = 0.001

def forward(image, label=None, predict=False):
    output = conv.forward((image / 255) - 0.5)
    output = pool.forward(output)
    output = sfmx.forward(output)

    if predict and not label:
        return np.argmax(output)
    
    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0
    
    return output, loss, acc

def training(image, label, learnRate):
    output, loss, acc = forward(image, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / output[label]

    gradient = sfmx.backprop(gradient, learnRate)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, learnRate)

    return loss, acc

trainImages = loadImages('train-images.idx3-ubyte')
trainLabels = loadLabels('train-labels.idx1-ubyte')
testImages = loadImages('t10k-images.idx3-ubyte')
testLabels = loadLabels('t10k-labels.idx1-ubyte')

start = time.perf_counter()
for i in range(epoch):
    print('--- Epoch %d ---' % (i + 1))

    permutation = np.random.permutation(len(trainImages))
    trainImages = trainImages[permutation]
    trainLabels = trainLabels[permutation]

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(trainImages, trainLabels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = training(im, label, learnRate)
        loss += l
        num_correct += acc

end = time.perf_counter()
print('End Training')
print(f'Training Runtime: {(end - start):.4f} seconds')

time.sleep(1)

print('Start Testing')
loss = 0
num_correct = 0
for im, label in zip(testImages, testLabels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(testImages)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)