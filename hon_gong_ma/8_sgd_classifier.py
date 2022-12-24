# # # 점진적 학습 - 데이터를 학습시켜 데이터가 없어져도 예측가능, 새로운 데이터 추가에 용이
# # # 확률적 경사 하강법을 통해 가능
# # # 손실 함수 - 머신러닝 알고리즘이 얼마나 엉터리인지 측정, 작을수록 좋음.


# # SGDClassifier
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
# print(fish)

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
# print(fish_input)
# print(fish_target)

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
    )




from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)



# # 확률적 경사 하강법
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
# loss 는 손실함수의 종류, max_iter 는 훈련세트의 반복횟수(에포크 횟수)
sc.fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))


# partial_fit
# fit 한 후 사용가능하면 한번씩 에포크 더 해줌.
sc.partial_fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))


# 과대과소 적합 조절하는 방법
# 에포크 횟수 조절
import numpy as np

sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)



for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))



import matplotlib.pyplot as plt
plt.plot(train_score, 'r')
plt.plot(test_score, 'b')
plt.xlabel('epoch')
plt.ylabel('accuracy')
# plt.show()

# 에포크 횟수 100이 적당할듯
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
# tol = none -> 자동으로 멈추지 않고 무조건 반복
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))
print(sc.predict(test_scaled[:5]))


# 플러스 - 힌지
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
# print(sc.score(train_scaled,train_target))
# print(sc.score(test_scaled,test_target))






