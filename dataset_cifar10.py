from keras.layers import *
from keras.optimizers import *
from keras.datasets import mnist, cifar10
from keras.models import Model
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# X_train = np.expand_dims(X_train, axis = 3) / 255
# X_test = np.expand_dims(X_test, axis = 3) / 255
X_train = X_train / 255
X_test = X_test / 255

IMG_SHAPE = X_train.shape[1:]


def SiameseNetworkModel():
  input_ = Input(shape = IMG_SHAPE)
  
  #Block conv 1
  conv_1_32 = Conv2D(32, kernel_size = (3, 3), padding = 'same')(input_)
  drop_1_32 = Dropout(0.25)(conv_1_32)
  maxpool_1_32 = MaxPooling2D(pool_size = (2, 2))(drop_1_32)
  
  conv_1_64 = Conv2D(64, kernel_size = (3, 3), padding = 'same')(maxpool_1_32)
  drop_1_64 = Dropout(0.25)(conv_1_64)
  maxpool_1_64 = MaxPooling2D(pool_size = (2, 2))(drop_1_64)
  
  conv_1_128 = Conv2D(128, kernel_size = (3, 3), padding = 'same')(maxpool_1_64)
  drop_1_128 = Dropout(0.25)(conv_1_128)
  maxpool_1_128 = MaxPooling2D(pool_size = (2, 2))(drop_1_128)
  
  #Block conv 2
  conv_2_32 = Conv2D(32, kernel_size = (3, 3), padding = 'same')(input_)
  drop_2_32 = Dropout(0.25)(conv_2_32)
  maxpool_2_32 = MaxPooling2D(pool_size = (2, 2))(drop_2_32)
  
  conv_2_64 = Conv2D(64, kernel_size = (3, 3), padding = 'same')(maxpool_2_32)
  drop_2_64 = Dropout(0.25)(conv_2_64)
  maxpool_2_64 = MaxPooling2D(pool_size = (2, 2))(drop_2_64)
  
  conv_2_128 = Conv2D(128, kernel_size = (3, 3), padding = 'same')(maxpool_2_64)
  drop_2_128 = Dropout(0.25)(conv_2_128)
  maxpool_2_128 = MaxPooling2D(pool_size = (2, 2))(drop_2_128)

  #Block classification
  concat_ = Concatenate()([maxpool_2_128, maxpool_1_128])
  flatten_ = Flatten()(concat_)
  dropout_ = Dropout(0.25)(flatten_)
  softmax = Dense(10, activation = 'softmax')(dropout_)
  
  return Model(inputs = [input_], outputs = [softmax])

model = SiameseNetworkModel()
model.summary()

checkpoint = ModelCheckpoint('weight.hdf5', monitor = 'val_acc', save_best_only = True, mode = 'max', verbose = 1)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 25, validation_split = 0.1, callbacks = [checkpoint], batch_size = 32)


loss, acc = model.evaluate(X_test, y_test)

print('loss: {}, acc: {}'.format(loss, acc))


#mnist dataset: loss: 0.08611345658177452, acc: 0.9904
#cifar10 dataset: loss: 1.3713039035797119, acc: 0.6387

