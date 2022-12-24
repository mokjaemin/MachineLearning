
# 1장. 개념정리


# 1.1 머신러닝 이란?

# 어떤 작업 T에 대한 컴퓨터 프로그램의 성능을 P로 측정했을 때 경험 E로 인해 성능이 향상됐다면,
# 이 컴퓨터 프로그램은 작업 T와 성능 측정 P에 대해 경험 E로 학습한 것이다.
# 훈련세트(경험 E) : 시스템이 학습하는데 사용하는 샘플
# 정확도(성능 측정 P) : 성능 측정, 상황에 따른 지표가 성능 측정의 지표가 됨.


# 1.2 왜 머신러닝을 사용하는가?

# "자동으로 감지한다는 장점"
# (+) 데이터 마이닝 : 머신러닝 기술을 적용하여 대용량의 데이터를 분석하면 겉으로는 보이지 않는 패턴을 발견
# 유용한 분야
# 1. 기존 솔루션으로는 많은 수동 조정과 규칙이 필요한 문제
# 2. 전통적인 방식으로는 해결 방법이 없는 복잡한 문제
# 3. 유동적인 환경
# 4. 복잡한 문제와 대량의 데이터에서 통찰 얻기


# 1.4 머신러닝의 종류

# 요약
# 1. 사람의 감독하에 훈련하는 것인지 아닌지 -> 지도, 비지도, 준지도, 강화학습
# 2. 실시간으로 점진적인 학습 -> 온라인, 배치학습
# 3. 단순하게 알고 있는 데이터 포인트와 새 데이터 포인트를 비교하는 지
# 아니면 과학자들이 하는 것처럼 훈련 데이터셋에서 패턴을 발견하여 예측 모델 만드는지
# -> 사례 기반, 모델 기반 학습

# 1. 지도 학습
# 알고리즘에 주입하는 훈련 데이터에 '레이블'이라는 원하는 답이 포함.
# 종류 : 분류, 회귀 (특성(예측변수)을 사용해 타깃수치를 예측.)
# (+) 일부 회귀는 분류로도 사용가능.
# 예시 : K-최근접 이웃, 선형 회귀, 로지스틱 회귀, 서포트 벡터 머신(SVM), 결정트리와 랜덤 포레스트, 신경망

# 2. 비지도 학습
# 훈련 데이터에 '레이불'이 없다.
# 종류 및 예시
# 1. 군집
# - K-평균
# - DBSCAN
# - 계층 군집 분석(HCA)
# - 이상치 탐지와 특이치 감지
# - 원-클래스
# - 아이솔레이션 포레스트
# 2. 시각화와 차원축소
# - 주성분 분석(PCA)
# - 지역적 선형 임베딩(LLE)
# - t-SNE
# 3. 연관 규칙 학습
# - 어프라디어리
# - 이클렛

# 간단한 예시
# 계층 군집, 시각화, 차원축소, 특성추출, 이상치 탐지(이상한 값 감지), 특이치 탐지(샘플간 이상 값 감지),
# 연관 규칙 학습(특성간 흥미로운 관계 찾음, 바베큐 소스를 사는 사람들은 대부분 바베큐 고기를 산다.)

# 이상치 감지 vs 특이치 감지
# 강아지 사진들 속 치와와
# 이상치 감지 -> 치와와를 이상치로 분류
# 특이치 감지 -> 그냥 강아지로 생각하고 분류 안함.

# 3. 준지도 학습
# 일부만 '레이블'이 있는 경우
# ex) 가족 사진 데이터가 있다고 가정햿을 때, 구성원 A가 어떤 사진에 있는지 알아냄
# 하지만 이름은 모름.
# 지도학습과 비지도 학습의 조합

# 4. 강화학습
