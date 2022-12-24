# 신경망 모델 훈련

# 데이터 준비
import imp
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()
train_scaled = train_input/255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled , train_target, test_size=0.2, random_state=42
)


# 모델 함수를 정의해서 사용
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# model = model_fn()
# model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
# verbose = 0 -> 훈련과정을 출력하지 않음.
# histoty라는 변수에 훈련을 넣음으로써 훈련 측정값을 변수에 담음.


# 담겨있는 내용 확인
# print(history.history.keys())
# loss(손실) 와 accuracy(정확도)가 담겨있음.


# 그래프를 통해 에포크에 따른 손실 확인
import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
# 그래프를 통해 에포크 값이 커질 수록 손실이 낮아짐을 알 수 있음.


# 그래프를 통해 에포크에 따른 정확도 확인
# plt.plot(history.history['accuracy'])
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()
# 에포크가 커짐에 따라 정확도가 커짐을 알 수 있음.


# 에포크의 횟수를 20으로 늘려서 손실 다시 측정.
# model = model_fn()
# model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
# plt.plot(history.history['loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
# 에포크가 커질수록 손실은 계속 감소함.


# 인공신경망에서 모델이 잘 훈련되었는지 판단하려면 정확도보다는 손실함수의 값을 확인.
# 이를 통해 과대/과소 적합 판단.
# model = model_fn()
# model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# history = model.fit(
#     train_scaled, train_target, epochs=20, verbose=0,
#     validation_data = (val_scaled, val_target)
# )
# print(history.history.keys())
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()
# 결과를 보면 훈련모델에 대한 손실은 계속 감소하지만 검증세트에 대한 손실은 2.5이후 다시 증가함


# adam 옵티마이저를 통해 학습률 조정
# model = model_fn()
# model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy', metrics='accuracy')
# history = model.fit(
#     train_scaled, train_target, epochs=20, verbose=0,
#     validation_data = (val_scaled, val_target)
# )
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()
# adam을 사용하닌 확실히 과대적합(에포크가 커질수록 간격이 벌어짐)이 줄어들음.
# 즉 adam 옵티마이저가 데이터셋에 더 적합.


# 과대적합을 막는 다른방법 -> 드롭아웃
# 원리 : 훈련과정중 층에있는 일부 뉴런을 랜덤하게 꺼서 과대적합을 막는다.
# 드롭아웃이라는 층을 새로만들어 30퍼센트 정도 드롭아웃을 해보자
# 물론 이층에는 파라미터는 없다.
# model = model_fn(keras.layers.Dropout(0.3))
# # print(model.summary()) # 드롭아웃 층이 잘 생성되었음을 알 수 있음.
# model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy', metrics='accuracy')
# history = model.fit(
#     train_scaled, train_target, epochs=20, verbose=0,
#     validation_data = (val_scaled, val_target)
# )
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()
# 그래프를 통해 과대적합이 전보다 줄었음을 확인할 수 있고 에포크 10정도가 과대적합이 가장 작음을 알 수 있음.


# 에포크를 10으로 해보자
# model = model_fn(keras.layers.Dropout(0.3))
# model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy', metrics='accuracy')
# history = model.fit(
#     train_scaled, train_target, epochs=10, verbose=0,
#     validation_data = (val_scaled, val_target)
# )


# 모델 저장 해보기

# 파라미터를 확장자 h5 즉, HDF5 포멧으로 저장해보기
# model.save_weights('model-weights.h5')
# 모델 구조와 모델 파라미터를 함께 저장해보기
# model.save('model-whole.h5')


# 훈련되지 않은 모델을 만들고 저장된 모델로 훈련시켜보기
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.h5')

# 방법1 - predict를 통한 정확도 출력
# 검증세트에 대한 각각 클래스일 확률을 측정해 가장 높은 확률을 가진 클래스와 해당 클래스를 비교하여
# 정확도를 출력
import numpy as np
# val_labels = np.argmax(model.predict(val_scaled), axis=-1)
# print(np.mean(val_labels == val_target))


# 방법2 - evaluate를 통한 정혹도 출력
# 위와 차이점 - 파일 안에 모델의 구조와 파라미터가 함께 저장되어있다
# 따라서 evaluate 사용 가능
# model = keras.models.load_model('model-whole.h5')
# model.evaluate(val_scaled, val_target)


# 콜백
# 위 전체과정을 아주 간단히 한번에 끝내보자
# 최적의 에포크를 찾고 저장하고 불러와 훈련 및 평가까지!
# model = model_fn(keras.layers.Dropout(0.3))
# model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy', metrics='accuracy')
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
# model.fit(
#     train_scaled, train_target, epochs=20, verbose=0,
#     validation_data = (val_scaled, val_target),
#     callbacks = [checkpoint_cb]
# )
# model = keras.models.load_model('best-model.h5')
# model.evaluate(val_scaled, val_target)


# 조기종료
# 위의 과정으로 지금까지의 과정을 간단히 해보았다
# 하지만 20번의 에포크 말고 과대적합이 시작되면 자동 종료하게끔 만들어보자.
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
# patience -> 2번이상 증가하지 않으면 훈련중지 ,
# restore_best_weights -> 가장 낮은 검증 손실을 파라미터로 가져옴
history = model.fit(
    train_scaled, train_target, epochs=20, verbose=0,
    validation_data = (val_scaled, val_target),
    callbacks = [checkpoint_cb, early_stopping_cb]
)
# print(early_stopping_cb.stopped_epoch) # 중지될때의 에포크 횟수


# 그래프를 통해 확인
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()


# 평가
model.evaluate(val_scaled, val_target)