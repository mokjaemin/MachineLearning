
# 검증세트
# 훈련세트, 검증세트, 테스트세트를 두어 교차 검증을 한다.
# 훈련세트를 한번 더 나눠주는 방식

from numpy.random import shuffle
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, random_state=42
)

# 훈련세트를 다시 훈련세트와 검증세트로 나눔
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, random_state=42
)
# print(sub_input.shape, val_input.shape, test_input.shape)


# 훈련 및 평가
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
# print(dt.score(sub_input, sub_target))
# print(dt.score(val_input, val_target))

# 교차 검증
# 5-폴트 교차검증 
# 최초 훈련세트를 오등분하여 처음부분, 5분의 2부분 등으로 검증세트로 두어
# 세번의 훈련을 함.
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
# print(scores)
# fit-time - 각 세트에 대해 훈련시간과 표시
# score-time - 각 세트에 대해 검증시간을 표시
# test-score - 검증 폴드의 점수


# 각 검증폴드의 점수의 평균 구하기
import numpy as np
# print(np.mean(scores['test_score']))


# 교차검증 함수 cross_validate 함수의 default
# 5차 폴트, 나눌때마다 데이터를 섞지 않음.
from sklearn.model_selection import StratifiedKFold
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 이를 활용해 폴드수와 섞을지 여부 판단 가능
scores = cross_validate(dt, train_input, train_target, cv=splitter)
# print(np.mean(scores['test_score']))


# 그리드 서치
# 최적의 매개변수(하이퍼파라미터)와 최적의 max_depth을 찾아서 교차검증까지 해줌
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

# 훈련 후 최적의 평가속성
dt = gs.best_estimator_
# print(dt.score(train_input, train_target))

# 최적의 param
# print(gs.best_params_)

# 교차검증 점수
# print(gs.cv_results_['mean_test_score'])

# 위와 동일한 검사지만 간략
best_index = np.argmax(gs.cv_results_['mean_test_score'])
# print(gs.cv_results_['params'][best_index])


# 그리드 서치 최종본
params = {
    'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
    'max_depth' : range(5, 20, 1),
    'min_samples_split' : range(2, 100, 10)
}
# print(params['min_samples_split'])

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
# 최상의 param
# print(gs.best_params_)
# 최상의 교차 검증 점수
# print(np.max(gs.cv_results_['mean_test_score']))


# 랜덤 서치
from scipy.stats import uniform, randint
params = {
    'min_impurity_decrease' : uniform(0.0001, 0.001),
    'max_depth' : randint(20, 50),
    'min_samples_split' : randint(2, 25),
    'min_samples_leaf' : randint(1, 25)
}


from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(
    random_state=42), params,
    n_iter=100, n_jobs=-1, random_state=42
)
gs.fit(train_input, train_target)

# 결과
# print(gs.best_params_)
# print(np.max(gs.cv_results_['mean_test_score']))

# 마지막 평가
dt = gs.best_estimator_
# print(dt.score(test_input, test_target))