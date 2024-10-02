import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input
from keras.src.models import Sequential
from keras.src.datasets import mnist
from keras.src.utils import to_categorical

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование входных данных
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Преобразование меток в категории (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Архитектура модели
model = Sequential()

# Входной слой
model.add(Input(shape=(28, 28, 1)))

# Первый сверточный слой
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Второй сверточный слой
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Полносвязные слои
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Выходной слой
model.add(Dense(10, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=11, batch_size=128, validation_split=0.2)

# Оценка модели
score = model.evaluate(X_test, y_test)
print(f"Точность на тестовых данных: {score[1] * 100:.2f}%")

# Сохранение модели
model.save('mnist_model.h5')
