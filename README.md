# video predict by Honglak Lee et. al 2016, NIPS. 
``
from keras.layers import Dense, Flatten, Conv2D, ZeroPadding2D, Reshape, Deconv2D
from keras.models import Sequential

input_shape = (210, 160, 12)

model = Sequential()
model.add(ZeroPadding2D(padding = (0, 1), input_shape = input_shape))
model.add(Conv2D(64, kernel_size=(8, 8), activation = 'relu', strides=2))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(128, (6,6), activation='relu', strides=2))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(128, (6,6), activation='relu', strides=2))

model.add(ZeroPadding2D(padding=(0,0)))
model.add(Conv2D(128, (4,4), activation='relu', strides=2))

model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))

#todo add action

model.add(Dense(128*11*8, activation='relu'))
model.add(Reshape((11, 8, 128)))

model.add(ZeroPadding2D(padding=(0,0)))
model.add(Deconv2D(128, (4,4), activation='relu', strides=2))

# model.add(Reshape((18, 24, 128)))
# model.add(Reshape((18, 128, 24)))
model.add(Reshape((24, 18, 128)))
# model.add(Reshape((24, 128, 18)))
# model.add(Reshape((128, 24, 18)))
# model.add(Reshape((128, 18, 24)))

model.add(Deconv2D(128, (6,6), activation='relu', padding='same', strides=2))
model.add(ZeroPadding2D(padding=(1,1)))

model.add(Deconv2D(128, (6,6), activation='relu', padding='same', strides=2))
model.add(ZeroPadding2D(padding=(1,0)))

model.add(Deconv2D(3, kernel_size=(8, 8), activation = 'relu', padding='valid', strides=2))
model.add(ZeroPadding2D(padding = (0, 1)))

model.summary()
``
