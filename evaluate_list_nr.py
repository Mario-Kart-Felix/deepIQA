import argparse
import os

from chainer import cuda
from chainer import serializers
import cv2
from nr_model import Model
import numpy as np
import six
from sklearn.feature_extraction.image import extract_patches


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', required=True)
parser.add_argument('--img_list', required=True)
parser.add_argument('--save_path', default='scores.txt')
parser.add_argument('--model', required=True, help='path to the trained model')
parser.add_argument('--top', choices=('patchwise', 'weighted'), default='weighted', help='top layer and loss definition')
parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
args = parser.parse_args()

model = Model(top=args.top)
cuda.cudnn_enabled = True
cuda.check_cuda_available()
xp = cuda.cupy
serializers.load_hdf5(args.model, model)
model.to_gpu()

img_paths = np.loadtxt(args.img_list, dtype=str)
lines = []
scores = []
for img_path in img_paths:
    img = cv2.imread(os.path.join(args.img_dir, img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    patches = extract_patches(img, (32, 32, 3), 32)
    X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))

    y = []
    weights = []
    batchsize = min(2000, X.shape[0])
    t = xp.zeros((1, 1), np.float32)
    for i in six.moves.range(0, X.shape[0], batchsize):
        X_batch = X[i:i + batchsize]
        X_batch = xp.array(X_batch.astype(np.float32))

        model.forward(X_batch, t, False, X_batch.shape[0])

        y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
        weights.append(xp.asnumpy(model.a[0].data).reshape((-1,)))

    y = np.concatenate(y)
    weights = np.concatenate(weights)

    line = '%s %f' % (img_path, np.sum(y * weights) / np.sum(weights))
    lines.append(line)
    scores.append(np.sum(y * weights) / np.sum(weights))
    print(line)

np.savetxt(args.save_path, np.array(lines)[np.argsort(scores)], delimiter='\n', fmt='%s')
