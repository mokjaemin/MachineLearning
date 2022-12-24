

# CNN 합성곱 신경망을 사용한 이미지 분류

# 데이터 전처리
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
# 3차원을 4차원으로 만들어줌
# 가로:28 세로:28 사진 50000개를 가로:28 세로:28 각사진마다 rgb인 사진이 50000개인 4차원 배열로


# 모델링
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
# 필터(뉴런의 개수)는 32개, 커널(도장)의 크기는 (3,3)

# 폴링층 추가
model.add(keras.layers.MaxPooling2D(2))
# 특성맵의 크기(14,14,32)

# 두번쨰 합성곱-폴링 층 추가
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2))
# 특성맵의 크기(7,7,64)


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

# 데이터 점검
# print(model.summary())
# keras.utils.plot_model(model, show_shapes='True', to_file='cnn-architecture.png', dpi=300)


# 컴파일과 훈련
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
# history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])


import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()

model.evaluate(val_scaled, val_target)

