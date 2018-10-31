import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from IPython.display import clear_output
# Generate dummy data
import numpy as np
###############################
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.pause(.3)
        plt.cla()
        # plt.show();


plot_losses = PlotLosses()
##################################

x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
print(y_train)
input=Input(shape=(20,))
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
x1=Dense(64, activation='relu')(input)
# model.add(Dropout(0.5))
x2=Dense(64, activation='relu')(x1)
# model.add(Dropout(0.5))
x3=Dense(10, activation='softmax')(x2)
model = Model(inputs=input, outputs=x3)
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

result=model.fit(x_train, y_train,epochs=100, batch_size=128,validation_data=(x_test, y_test),verbose=1,callbacks=[plot_losses])
print ('val_loss:',result.history['val_loss'])
plt.figure(2)
plt.plot(np.array(range(100)),result.history['val_loss'])
plt.show()
# loss,acc= model.evaluate(x_test, y_test)
# plt.show()
# print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))