# 선형 회귀 - 일차원 선으로 예측
# 다항 회귀 - n차 그래프로 예측
# 다중 회귀 - 더 많은 특성을 부여하여 정확성을 높여줌
# 특성 공학 - 두가지 이상의 기존 특성을 통해 새로운 특성을 추출하여 정확성을 높임.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures # 특성을 새로 만들거나 데이터 전처리를 위한 도구 제공
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler # 표준점수를 위한 도구
from sklearn.linear_model import Ridge # 릿지
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso



# setting data
df = pd.read_csv('http://bit.ly/perch_csv_data')

# length, height, width
# input
perch_full = df.to_numpy()
print(perch_full)

# weight
# target
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])



# split data to train and test
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42
)


# practice polynomalFeatures
# 아래와 같은 경우에 기존 데이터 2,3에 2의 제곱, 3의 제곱, 2와 3의 곱과 1을 데이터에 추가함.
# poly = PolynomialFeatures()
# poly.fit([[2,3]])
# print(poly.transform([[2,3]]))

# 1 제거
# poly = PolynomialFeatures(include_bias=False)
# poly.fit([[2,3]])
# print(poly.transform([[2,3]]))


# 훈련모델에 적용하여 새로운 특성 만들기
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)

# train_poly 가 만들어진 방식 알기
# print(poly.get_feature_names())

# test도 적용
test_poly = poly.transform(test_input)




# 훈련시작
lr = LinearRegression()
lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target))
# print(lr.score(test_poly, test_target))


# 특성을 더 추가해보자
# 위에서는 길이, 높이 두께로 표현
# 세제곱, 네제곱항을 추가하여 특성 추가
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
# print(train_poly.shape) # 특성의 개수는 55개
# print(poly.get_feature_names())



# 훈련 다시 시작
lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target))
# print(lr.score(test_poly, test_target))
# 결과가 과대 적합 - 훈련을 너무 시킴.


# 과대적합 줄이기
# preview - 과소적합을 줄이기 위해선 인접 알고리즘에서는 인접 개수를 줄여주면 됐음.
# regularization
# 규제 - 훈련세트를 과도하게 학습하지 못하도록 훼방 놓는것.


# 표준점수를 통한 데이터 전처리
ss = StandardScaler()
ss.fit(train_poly)
# 표준점수로 변환됨
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)


# 규제법 - 릿지, 라쏘
# 릿지 - 계수를 제곱한 값을 기준으로 규제를 적용
# 라쏘 - 계수의 절대값을 기준으로 규제를 적용

# 릿지
ridge = Ridge()
ridge.fit(train_scaled, train_target)
# print(ridge.score(train_scaled, train_target))
# print(ridge.score(test_scaled, test_target))

# alpha 변수를 통해 계수를 줄이는 양 조절
# 적절한 alpha 값 찾아보기
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))


# 그래프를 통해 확인
# 로그 함수를 통해 x축의 간격을 일정하게 맞춰줌
# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('socre')
# plt.show()

# 가장 가깝고 테스트세트의 점수가 가장 높은 0.1을 알파값으로 설정.
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
# print(ridge.score(train_scaled, train_target))
# print(ridge.score(test_scaled, test_target))


# 라쏘 회귀
lasso = Lasso()
lasso.fit(train_scaled, train_target)
# print(lasso.score(train_scaled, train_target))


# 알파 조절
# train_score = []
# test_score = []
# alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
# for alpha in alpha_list:
#     lasso = Lasso(alpha=alpha, max_iter=10000)
#     lasso.fit(train_scaled, train_target)
#     train_score.append(lasso.score(train_scaled, train_target))
#     test_score.append(lasso.score(test_scaled, test_target))


# 그래프를 통해
# 최적의 알파는 10임
plt.plot(np.log10(alpha_list), train_score, label = 'train')
plt.plot(np.log10(alpha_list), test_score, label = 'test')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('score')
# plt.show()

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
# print(lasso.score(train_scaled, train_target))
# print(lasso.score(test_scaled, test_target))

# 라쏘가 0을 만든 개수
# print(np.sum(lasso.coef_ == 0))

# 결론 : 55개의 특징중 lasso가 사용한 특성은 0개임.