import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 표준점수를 위한 도구
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀
from scipy.special import expit # 로지스틱 함수
from scipy.special import softmax # 소프트맥스 함수
import matplotlib.pyplot as plt

# 데이터 불러오기
fish = pd.read_csv('http://bit.ly/fish_csv_data')

# 생선 종류 확인
# print(pd.unique(fish['Species']))

# 입력 데이터 만들기
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish[['Species']].to_numpy()

# 입력데이터, 훈련데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 표준화 처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)



# 훈련시작 후 검사
kn = KNeighborsClassifier()
kn.fit(train_scaled, train_target)
# print(kn.score(train_scaled, train_target))
# print(kn.score(test_scaled, test_target))
# print(kn.predict(test_scaled[:5]))
# print(test_target[:5])



# 클래스별 확률값 반환
proba = kn.predict_proba(test_scaled[:5])
# 기본으로 round 함수는 소수점 첫째자리에서 반올림하는데 decimals= 4 는 소수점 5번째자리에서 반올림.
# print(np.round(proba, decimals=4))
# 첫번째 행의 첫번째 열은 생선1에대한 확률 ...


# 인접 확인
# 4번째 생선의 인접 확인
distances, indexes = kn.kneighbors(test_scaled[3:4])
# print(train_target[indexes])


# 로지스틱 회귀
# z = ax + by + cj .... 등의 형태에서
# z를 확률로 나타내기 위해서 z의 범위를 0과1 사이로 만들기 위해 사용


# 그래프 그려보기
z = np.arange(-5, 5, 0.1) # 0.1 은 곡선도
phi = 1 / (1+np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
# plt.show()


# 로지스틱 회귀로 이진 분류 수행하기
# 도미와 빙어 골라내기
# true or false 반환
indexes = []
bream_smelt_TOR = (train_target == 'Bream') | (train_target == 'Smelt')


# true or false 반환을 통해 도미와 빙어 인덱스 찾기
for i in range(0 ,len(bream_smelt_TOR)):
    if bream_smelt_TOR[i] == True:
        indexes.append(i)
# print(indexes)


# 도미와 빙어 훈련 및 타깃 데이터 만들기
train_bream_smelt = train_scaled[indexes]
target_bream_smelt = train_target[indexes]


# # 훈련 시작
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
# print(lr.predict(train_bream_smelt[:5]))
# print(target_bream_smelt[:5])


# 예측 확률
# print(lr.predict_proba(train_bream_smelt[:5]))


# 계수 측정
# print(lr.coef_, lr.intercept_)


# z 값 계산 해보기
decisions = lr.decision_function(train_bream_smelt[:])
# print(decisions)


# z 값을 로지스틱 함수에 통과시켜 확률 구하기
# print(expit(decisions))


# 다중 분류
lr = LogisticRegression(C=20, max_iter=1000) # C는 규제, max_iter은 반복횟수
lr.fit(train_scaled, train_target)
# print(lr.score(train_scaled, train_target))
# print(lr.score(test_scaled, test_target))

# 예측
# print(lr.predict(train_scaled[:5]))
# print(train_target[:5])

# 확률 측정
# 1.bream 2.parcki 3.perch 4.pike 5.roach 6.smelt 7.whitefish
# print(lr.classes_)
proba = lr.predict_proba(test_scaled[:5])
# print(np.round(proba, decimals=3))
# print(test_target[10:20])



# 다중 분류는 소프트맥스 함수를 사용
# 이는 각각의 클래스의 z를 0-1사이의 값으로 구하고 이들의 합이 0-1 사이로 압축
# z1 ~ z7까지 구해보고 소프트맥스함수를 활용하여 확률로 바꿔보기

# z 값 계산.
decision = lr.decision_function(test_scaled[:5])
# print(np.round(decision, decimals=2))


# 소프트맥스 함수 통과 시키기.
# 즉, 확률 구해보기.
proba = softmax(decision, axis=1)
# print(np.around(proba, decimals=3))