# 심층 신경망
# 인공 신경망 강화.
# 인공 신경망에 층을 추가.

from tensorflow import keras
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()


# 픽셀값 0~255 를 0~1 사이로 변환
# 28*28 크기의 2차원 배열을 784크기의 1차원 배열로 변경
# 훈련세트와 검증세트로 나눔
from sklearn.model_selection import train_test_split
train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)


# 입력층(784개의 뉴런)과 출력층(10개의 뉴런)사이에 은닉층(100개의 뉴런)을 추가.
# 은닉층은 시그모이드 함수 사용, 출력층은 소프트맥스 함수를 사용.
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')


# 심층 신경망 만들기
model = keras.Sequential([dense1, dense2])

# 심층 신경망 층에 대한 정보
# print(model.summary())


# 층을 추가하는 다른 방법
model = keras.Sequential(
    [
        keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
        keras.layers.Dense(10, activation='softmax', name='output')
    ], name='패션 MNIST 모델'
)
# print(model.summary())


# 한번에 층을 다 만들지 않고 중간중간에 층을 추가하고 싶을때
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))



# 모델 훈련
# model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# model.fit(train_scaled, train_target, epochs=5)


# 렐루함수
# 점점 층이 많아질때 은닉층의 활성화 함수로 사용.
# 원리는 입력이 양수일 경우 그냥 입력을 출력시키고 음수일때는 0으로 만들어버림.

# Flatten
# 위의 과정 중 2차원 배열을 1차원 배열로 만드는 과정을 인공신경망의 층처럼 활용.
# 파라미터는 없음.

# 모델 준비
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# 데이터 준비
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()
train_scaled = train_input/255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 훈련
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

# 검증세트에의 성능도 확인.
model.evaluate(val_scaled, val_target)


# 각종 하이퍼 파라미터
# 하이퍼 파라미터 - 컴퓨터가 지정해주는 값이 아닌 사용자가 지정하는 값.


# 1. optimizer를 sgd(확률적 경사 하강법)으로 지정해보기
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')

# or

# 객체를 통한 학습으로 학습률도 지정할 수 있음.
sgd = keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')


# 그냥 적어만두는 모멘텀 최적화와 네스테로프 모멘텀 최적화
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)


# 2. optimizer를 adagrad로 지정해보기
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')


# 3. optimizer를 rmsprop으롤 지정해보기
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics='accuracy')
