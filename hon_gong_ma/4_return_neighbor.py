
# 결정 계수 회귀
# 클래스 분류가 아닌 특정 숫자 예측
# 길이에 따른 무게를 통해 무게 예측하는 회귀

from operator import index
from scipy.sparse.construct import random
from sklearn.neighbors import KNeighborsClassifier # 이웃 분류 알고리즘
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor # 이웃 회귀 알고리즘
from sklearn.metrics import mean_absolute_error # 타깃과 예측의 절대값 오차를 평균하여 반환



perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])



# 산점도 출력
# 농어의 길이가 커짐에 따라 무게도 커짐
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()


# 훈련세트와 테스트세트로 나눔
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)


# 사이킷 런을 사용할떄는 훈련, 테스트 인풋이 2차원 배열이여야 함 타깃은 노상관.
# reshape(-1, 1) - 전체 배열을 2차원으로 만들어줌 즉, 가로를 세로로 만들어줌.
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
# print(test_input.shape, train_input.shape)



# 결정 계수
# 훈련 및 테스트
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
# score - 분류에서는 정답을 맞춘 개수에 비율
# 회귀에서는 결정계수를 측정함. 
# print(knr.score(test_input, test_target))



# 테스트 세트에 대한 예측
test_prediction = knr.predict(test_input)
print(test_prediction)
print(test_target)


# 테스트 세트의 실제 결과와 예측 결과에 대한 평균 절대값 오차를 계산
mae = mean_absolute_error(test_target, test_prediction)
# 결과는 19정도 나오는데 실제 값이랑 대략 19정도 다르다는 것을 알 수 있음.



# 훈련 모델로 결정계수 측정
# 아래의 결과가 test score 보다 낮게 나옴 - 과소적합
# 과도하게 훈련모델의 결정계수가 높으면 과대 적합 - 너무 훈련모델에만 맞는다. 잘 예측 못한다.
# 반대 경우 - 과소적합 - 적절히 훈련되지 않았다.
# print(knr.score(train_input, train_target))



# 과소적합의 해결법
# 평가 이웃의 개수를 줄인다 - 테스트 스코어를 낮춤.
knr.n_neighbors = 3
knr.fit(train_input, train_target)
# print(knr.score(train_input, train_target))
# print(knr.score(test_input, test_target))



