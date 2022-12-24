# 트리 앙상블
# 결정트리의 진화라고 생각.
# 결정트리를 랜덤하게 진행하는 느낌.


# 데이터 준비
from numpy.core.numeric import cross
from numpy.random import rand, random_sample, shuffle
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, random_state=42
)



from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier


# 랜덤포레스트 훈련및 교차검증.
# rf = RandomForestClassifier(n_jobs=-1, random_state=42)
# scores = cross_validate(rf, train_input, train_target, return_train_score=True,
# n_jobs=-1)
# return_train_score을 true로 두어 훈련세트에 대한 점수도 함꼐 출력.


# 검사
import numpy as np
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 결과는 과대적합 되어 있음.


# 특성중요도 확인
# alcohol, suger, ph 중에 뭐가 중요한지 파악
# rf.fit(train_input, train_target)
# print(rf.feature_importances_)
# sugar의 중요도가 가장 높구낭.
# 이는 원래 결정트리의 중요도보다 낮음
# 이유는 랜덤하게 특성을 선택해서 결정트리를 훈련하기 때문이다.


# 랜덤포레스트는 랜덤하게 훈련데이터를 결정한다고했음.
# 선택되지 않은 데이터가 존재함. 이를 oob샘플이라고 부름.
# rf = RandomForestClassifier(oob_score=True, n_jobs= -1, random_state=42)
# rf.fit(train_input, train_target)
# print(rf.oob_score_)
# oob 샘플은 교차검증과 비슷한 역할을 한다.



# 엑스트라 트리
# 랜덤 포래스트와의 차이점 - 데이터를 부트스트랩(랜덤하게 데이터 선택)하지 않고 전체 데이터 사용.
# 노드 분할을 가장 좋은 분할로 하는게 아니라 랜덤하게 분할.
# from sklearn.ensemble import ExtraTreesClassifier
# et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
# scores = cross_validate(et, train_input, train_target,
# return_train_score=True, n_jobs=-1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 랜덤포레스트와 비슷한 결과군요.


# 특성 중요도 출력
# et.fit(train_input, train_target)
# print(et.feature_importances_)
# 랜덤포레스트와 같게도 당도에 대한 중요도가 결정트리에서의 중요도보다 낮아짐.


# 그레이디언트 부스팅
# 깊이가 얕은 결정트리를 이용하여 이전트리의 오차를 보완하는 방식이다.
# 기본적으로 깊이 3의 트리 100개를 사용
# 경사 하강법과 방법이 유사.

from sklearn.ensemble import GradientBoostingClassifier
# gb = GradientBoostingClassifier(random_state=42)
# scores = cross_validate(gb, train_input, train_target,
# return_train_score=True, n_jobs=-1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))


# 트리의 개수를 늘리고 학습률을 증가시켜 성능 향상
# 트리의 개수를 500개로 늘리고 학습률을 기본값 0.1에서 0.2로 늘림.
# gb = GradientBoostingClassifier(random_state=42, n_estimators=500, learning_rate=0.2)
# scores = cross_validate(gb, train_input, train_target,
# return_train_score=True, n_jobs=-1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))


# 중요도 확인.
# gb.fit(train_input, train_target)
# print(gb.feature_importances_)
# 다른 트리보다 당도를 더 중요시 여김.


# 히스토그램 기반 그레이디언트 부스팅
# 위의 그레이디언트 부스팅의 속도와 성능을 향상시킴.
# 트리의 개수 대신 max_iter를 통해 성능 조절.

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores= cross_validate(hgb, train_input, train_target, return_train_score=True)
# print(np.mean(scores['train_score']), np.mean(scores['test_score'])) 


# 훈련세트 특성 중요도 조사.
# 특성을 하나씩 랜덤하게 섞어서 모델의 성능이 변화하는지 관찰하여 어떤 특성이 중요한지 계산.
from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target,
n_repeats=10, random_state=42, n_jobs=-1) # n_repeats 는 랜덤하게 섞을 횟수
# print(result.importances_mean)
# 순서대로 특성중요도, 평균, 표준편차

# 테스트세트 특성 중요도 조사.
result = permutation_importance(hgb, test_input, test_target,
n_repeats=10, random_state=42, n_jobs=-1) 
# print(result.importances_mean)

# 성능 최종 확인.
# print(hgb.score(test_input, test_target))


# xgboost
# 다양한 부스팅 알고리즘을 지원하는 라이브러리.
# from xgboost import XGBClassifier
# xgb = XGBClassifier(tree_method='hist', random_state =42)
# scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))


#lightgbm
# 다양한 부스팅 알고리즘을 지원하는 라이브러리.
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input ,train_target, return_train_score=True, n_jobs=-1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))

