#pso_nn

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import cv2
from keras.optimizers import Adam

def PSO_nn(lr,drp,train,val):
        emotion_model = Sequential()

        emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(drp))

        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(drp))

        emotion_model.add(Flatten())
        emotion_model.add(Dense(1024, activation='relu'))
        emotion_model.add(Dropout(drp))
        emotion_model.add(Dense(7, activation='softmax'))

        cv2.ocl.setUseOpenCL(False)
        emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr, decay=1e-6), metrics=['accuracy'])
        
        emotion_model_info = emotion_model.fit(
        train,
        steps_per_epoch=28709 // 64,
        epochs=1,
        validation_data=val,
        validation_steps=7178 // 64)
        return emotion_model_info.history['val_loss'][0]

   
