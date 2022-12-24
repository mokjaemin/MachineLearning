
# 선형 회귀

import numpy as np
from operator import index
from scipy.sparse.construct import random
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier # 이웃 분류 알고리즘
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor # 이웃 회귀 알고리즘
from sklearn.metrics import mean_absolute_error # 타깃과 예측의 절대값 오차를 평균하여 반환
from sklearn.linear_model import LinearRegression





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


# 훈련세트와 테스트세트로 나눠줌.
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)



# 인풋을 2차원으로 만들어줌.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)


# 훈련
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)


# 길이가 50인 생선 무게 예측
# print(knr.predict([[50]]))


# 최근접 이웃 모델 확인
# distances, indexes = knr.kneighbors([[50]])
# plt.scatter(train_input, train_target)
# plt.scatter(50, 1010, marker='^')
# plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# print(np.mean(np.mean(train_target[indexes])))



# 100 cm 생선 다시 그려봄
# print(knr.predict([[100]]))
# distances, indexes = knr.kneighbors([[100]])
# plt.scatter(train_input, train_target)
# plt.scatter(100, 1010, marker='^')
# plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# 최근접 이웃 모델은 데이터 부족시 정확히 예측 불가
# 선형 회귀를 통해 예측
# y = ax + b 형태
# x - 길이, y - 무게
lr = LinearRegression()
lr.fit(train_input, train_target)
# print(lr.predict([[50]]))
# print(lr.coef_, lr.intercept_) # 각각 a,b


# 그래프로 직선 확인
plt.scatter(train_input, train_target)
plt.scatter(50, 1241.8, marker='^')
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_]) # 두점을 직선으로 이어줌.
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()


# 과대, 과소 적합 측정
# print(lr.score(train_input, train_target))
# print(lr.score(test_input, test_target))



# 다항 회귀(곡선)
# y = axx + bx + c
# 두배열을 나란히 연결
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))


# 다시 훈련
# input을 제곱값과 기본값으로 주었기에 2차방정식으로 자동 생성
# lr = LinearRegression()
# lr.fit(train_poly, train_target)


# 무게 다시 예측해보기
# print(lr.predict([[50**2, 50]]))


# 절편 확인
# 출력값 순서대로 abc임
# print(lr.coef_, lr.intercept_)


# 방정식 그려보기
# point = np.arange(15, 50)
# plt.plot(point, lr.coef_[0]*point**2 + lr.coef_[1]*point + lr.intercept_, 'r')
# plt.show()


# 결정계수 측정을 통해 과대, 과소 적합 측정
# print(lr.score(train_poly, train_target))
# print(lr.score(test_poly, test_target))
