from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.decomposition import PCA
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import regularizers
K.set_image_dim_ordering('th')
from keras.utils.np_utils import to_categorical



train_image = 'datait.out'
train_label ='datalt.out'
test_image = 'datai.out'
test_label = 'datal.out'


# dimensions of our images.
img_width, img_height = 28, 28

epochs = 25
batch_size = 16



image_data = np.loadtxt(train_image, delimiter=',' , dtype='float32')
label_data = np.loadtxt(train_label , delimiter =',' , dtype='float32')
label_data = to_categorical(label_data.astype('int32'), num_classes=5)

print( 'Preprocessing.... ' ,len(image_data), len(label_data))


pca = PCA(n_components=100, whiten=True)
pca.fit(image_data)
image_data = pca.transform(image_data)
image_data =  pca.inverse_transform(image_data)
print(pca.explained_variance_ratio_.sum())

X_train, y_train = image_data , label_data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)


image_data = np.loadtxt(test_image, delimiter=',', dtype='float32')
label_data = np.loadtxt(test_label, delimiter=',', dtype='float32')

label_data = to_categorical(label_data.astype('int32'), num_classes=5)

X_test, Y_test = image_data, label_data

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# define data preparation
datagen = ImageDataGenerator(zca_whitening=True )
# fit parameters from data
datagen.fit(X_train)


input_shape = (1, img_width, img_height)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64 , activity_regularizer=regularizers.l2(0.03)))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(5))
model.add(Activation('sigmoid'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_generator  = datagen.flow(X_train, y_train, batch_size=10 , shuffle = True)
validation_generator = datagen.flow(X_test, Y_test, batch_size=10 , shuffle = True)

model.fit_generator(
    train_generator,
    steps_per_epoch=17169 // 50,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=3858 // 50)

model.save_weights('data1.h5')