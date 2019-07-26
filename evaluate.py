from keras.models import *
# from dataset_cifar10 import SiameseNetworkModel
from keras.datasets import cifar10
from keras.layers import * 

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_test = X_test / 255
IMG_SHAPE = X_test.shape[1:]

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
  
  conv_1_256 = Conv2D(256, kernel_size = (3, 3), padding = 'same', dilation_rate = (1, 1))(maxpool_1_128)
  drop_1_256 = Dropout(0.1)(conv_1_256)
  norm_1_256 = BatchNormalization()(drop_1_256)
  maxpool_1_256 = MaxPooling2D(pool_size = (2, 2))(norm_1_256)
  
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
  
  conv_2_256 = Conv2D(256, kernel_size = (3, 3), padding = 'same', dilation_rate = (2, 2))(maxpool_2_128)
  drop_2_256 = Dropout(0.1)(conv_2_256)
  norm_2_256 = BatchNormalization()(drop_2_256)
  maxpool_2_256 = MaxPooling2D(pool_size = (2, 2))(norm_2_256)

  #Block classification
  concat_ = Concatenate()([maxpool_2_256, maxpool_1_256])
  flatten_ = Flatten()(concat_)
  dense_ = Dense(512, activation = 'relu')(flatten_)
  dropout_ = Dropout(0.1)(dense_)
  
  softmax = Dense(10, activation = 'softmax')(dropout_)
  
  return Model(inputs = [input_], outputs = [softmax])

model = SiameseNetworkModel()


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.load_weights('weight_848.hdf5')

loss, acc = model.evaluate(X_test, y_test)

print('loss: {}, acc: {}'.format(loss, acc))