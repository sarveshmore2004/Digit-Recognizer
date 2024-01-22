import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

mnist = tf.keras.datasets.mnist

(X_train , Y_train) , (X_test , Y_test) = mnist.load_data()
X_train , X_test = X_train/255.0 , X_test/255.0
X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)

inputs = KL.Input(shape=(28, 28, 1))
x = KL.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
x = KL.Conv2D(64, (3, 3), activation='relu')(x)
x = KL.MaxPooling2D(pool_size=(2, 2))(x)
x = KL.Dropout(0.25)(x)
x = KL.Flatten()(x)
x = KL.Dense(128, activation='relu')(x)
x = KL.Dropout(0.5)(x)
outputs = KL.Dense(10, activation='softmax')(x)

model = KM.Model(inputs, outputs)

model.compile(optimizer="adam" , metrics=["accuracy" ] , loss= "sparse_categorical_crossentropy")

model.fit(X_train , Y_train , epochs = 5)

model.save('model2')

# new_model = tf.keras.models.load_model('my_model')