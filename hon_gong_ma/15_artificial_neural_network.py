# 인공 신경망


# 데이터 준비
import imp
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
# print(train_input.shape, train_target.shape)


# 데이터 이미지 파악
import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 10, figsize=(10,10))
# for i in range(10):
#     axs[i].imshow(train_input[i], cmap='gray_r')
#     axs[i].axis('off')
# plt.show()


# 타깃 데이터 확인.
# print([train_target[i] for i in range(10)])
# 0 - 티셔츠, 1 - 바지, 2 - 스웨터...


# 클래스별 개수 확인
import numpy as np
# print(np.unique(train_target, return_counts=True))


# input을 정규화 및 1차원 배열로 만들기
train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1, 28*28)
# print(train_scaled.shape)


# SGD Classifier + 교차검증
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
# sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
# scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
# print(np.mean(scores['test_score']))



# 인공 신경망에서는 교차검증을 하면 너무 오래걸림.
# 그래서 세트만 나눠서 실행
import tensorflow as tf
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
# 트레인세트의 20퍼센트를 검증세트로 바꿈


# 잘 나누어졌는지 확인.
# print(train_scaled.shape, train_target.shape)
# print(val_scaled.shape, val_target.shape)


# 밀집층(딱히 의미는 없고 밀집된 모양이라 밀집층) 만들기
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
# 순서대로 뉴런의 수(분류될 클래스의 수), 뉴런에 적용할 함수, 입력의크기(픽셀 수)

# 모델 객체 만들기
model = keras.Sequential(dense)

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# loss 는 다중분류 크로스 엔트로피 손실함수임.
# 이중분류일때는 binary_crossentropy


# sparse 정리
# 원-핫 인코딩 -> 0 또는 1 의 정수로 타깃이 표현되었을때 사용
# 지금 데이터의 타깃이 1~9의 정수로 이루어짐
# 그래서 원-핫 인코딩 사용할 필요없음.
# 그럴떄 sparse 붙임.


# 훈련
model.fit(train_scaled, train_target, epochs=5)


# 성능 평가
model.evaluate(val_scaled, val_target)