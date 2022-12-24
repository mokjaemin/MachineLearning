

# 결정트리
# 트리형식의 원리로 결과를 예측해보자.


# 데이터 불러오기
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')


# 누락된 데이터 없는지 확인
# column 에 대해 non-null count 가 동일하면 누락이 없다.
# 누락 있을시에는 평균으로 누락 채움.
# print(wine.info())


# 기본적인 수학적 통계
# 사분위수 - 25%, 50% - 데이터를 일렬로 했을때 중간값이나 4분의1 값을 의미.
# print(wine.describe())


# 데이터 전처리
data_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
data_target = wine['class'].to_numpy()
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data_input, data_target, random_state=42
)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# 로지스틱 회귀 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
# print(lr.score(train_scaled, train_target))
# print(lr.score(test_scaled, test_target))
# print(lr.coef_, lr.intercept_)


# 로지스틱 회귀의 score가 낮아서 결정 트리 알고리즘을 사용
# 결정 트리 알고리즘.
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
# print(dt.score(train_scaled, train_target))
# print(dt.score(test_scaled, test_target))


# 결정트리의 원리를 알기 위해 그려봄
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# plt.figure(figsize=(10,7))
# plot_tree(dt)
# plt.show()


# 결정트리가 너무 복잡해서 자세히 확인
# plt.figure(figsize=(10,7))
# plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
# filled 내려갈수록 특성 클래스의 비율이 높아지면 진하게
# plt.show()


# ex)
# sugar <= -0.273
# gini = 0.364
# samples = 4872
# value = [1165, 3707]
# sugar 조건을 만족하면 왼쪽 반대면 오른쪽으로 분류됨
# value의 왼쪽이 음성클래스, 오른쪽이 양성클래스
# gini는 불순도 - (1 - 음성비율 제곱 + 양성비율 제곱)


# 훈련세트의 점수가 테스트세트의 점수보다 높은 현상 해결
# 가지치기 - max_depth를 조절
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
# print(dt.score(train_scaled, train_target))
# print(dt.score(test_scaled, test_target))

# plt.figure(figsize=(15, 5))
# plot_tree(dt, filled=True, feature_names=['alcohol','sugar','pH'])
# plt.show()


# 결정 트리는 표준화를 할 필요가 없음.
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
# print(dt.score(train_input, train_target))
# print(dt.score(test_input, test_target))


# 결정 트리 그려보기
plt.figure(figsize=(15, 5))
plot_tree(dt, filled=True, feature_names=['alcohol','sugar','pH'])
# plt.show()


# 특성별 중요도 확인.
# 값이 높은 특성일수록 결정트리에서의 중요도가 높음.
# print(dt.feature_importances_)