from keras.layers import *
from keras.optimizers import *
from keras.datasets import mnist, cifar10
from keras.models import Model
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
# from scipy.utils import Shuffle


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# X_train = np.expand_dims(X_train, axis = 3) / 255
# X_test = np.expand_dims(X_test, axis = 3) / 255
X_train = X_train / 255
X_test = X_test / 255

endp = int(X_train.shape[0] * 0.9)
X_val = X_train[endp:]
y_val = y_train[endp:]
X_train = X_train[:endp]
y_train = y_train[:endp]

IMG_SHAPE = X_train.shape[1:]

datagen = ImageDataGenerator(
          rotation_range = 10,
          width_shift_range = 0.1,
          height_shift_range = 0.1, 
          horizontal_flip = True
)


def SiameseNetworkModel():
  input_ = Input(shape = IMG_SHAPE)
  
  #Block conv 1
  conv_1_32 = Conv2D(32, kernel_size = (3, 3), padding = 'same', dilation_rate = (1, 1))(input_)
  drop_1_32 = Dropout(0.1)(conv_1_32)
  norm_1_32 = BatchNormalization()(drop_1_32)
  maxpool_1_32 = MaxPooling2D(pool_size = (2, 2))(norm_1_32)
  
  
  conv_1_64 = Conv2D(64, kernel_size = (3, 3), padding = 'same', dilation_rate = (1, 1))(maxpool_1_32)
  drop_1_64 = Dropout(0.1)(conv_1_64)
  norm_1_64 = BatchNormalization()(drop_1_64)
  maxpool_1_64 = MaxPooling2D(pool_size = (2, 2))(norm_1_64)
  
  conv_1_128 = Conv2D(128, kernel_size = (3, 3), padding = 'same', dilation_rate = (1, 1))(maxpool_1_64)
  drop_1_128 = Dropout(0.1)(conv_1_128)
  norm_1_128 = BatchNormalization()(drop_1_128)
  maxpool_1_128 = MaxPooling2D(pool_size = (2, 2))(norm_1_128)
  
  #Block conv 2
  conv_2_32 = Conv2D(32, kernel_size = (3, 3), padding = 'same', dilation_rate = (2, 2))(input_)
  drop_2_32 = Dropout(0.1)(conv_2_32)
  norm_2_32 = BatchNormalization()(drop_2_32)
  maxpool_2_32 = MaxPooling2D(pool_size = (2, 2))(norm_2_32)
  
  conv_2_64 = Conv2D(64, kernel_size = (3, 3), padding = 'same', dilation_rate = (2, 2))(maxpool_2_32)
  drop_2_64 = Dropout(0.1)(conv_2_64)
  norm_2_64 = BatchNormalization()(drop_2_64)
  maxpool_2_64 = MaxPooling2D(pool_size = (2, 2))(norm_2_64)
  
  conv_2_128 = Conv2D(128, kernel_size = (3, 3), padding = 'same', dilation_rate = (2, 2))(maxpool_2_64)
  drop_2_128 = Dropout(0.1)(conv_2_128)
  norm_2_128 = BatchNormalization()(drop_2_128)
  maxpool_2_128 = MaxPooling2D(pool_size = (2, 2))(norm_2_128)

  #Block classification
  concat_ = Concatenate()([maxpool_2_128, maxpool_1_128])
  flatten_ = Flatten()(concat_)
  dense_ = Dense(512, activation = 'relu')(flatten_)
  dropout_ = Dropout(0.1)(dense_)
  
  softmax = Dense(10, activation = 'softmax')(dropout_)
  
  return Model(inputs = [input_], outputs = [softmax])

model = SiameseNetworkModel()
model.summary()

checkpoint = ModelCheckpoint('weight.hdf5', monitor = 'val_acc', save_best_only = True, mode = 'max', verbose = 1)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(X_train, y_train, epochs = 25, validation_split = 0.1, callbacks = [checkpoint], batch_size = 32)
model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 100, steps_per_epoch = X_train.shape[0] / 32, validation_data = (X_val, y_val), callbacks = [checkpoint])

loss, acc = model.evaluate(X_test, y_test)

print('loss: {}, acc: {}'.format(loss, acc))


#loss: 1.3713039035797119, acc: 0.6387

#loss: 1.1491667903900147, acc: 0.6625

#dropout 0.25 -> 0.1 with 100 epochs => loss: 0.6515964792728424, acc: 0.7875

#add dense(512) layers with 100 epochs => loss: 0.7207691070795059, acc: 0.818

#add conv(256) under each conv block => 0.5914149845719338, acc: 0.8409


