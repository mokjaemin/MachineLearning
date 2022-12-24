

# 분류
# 클래스를 분류

# 전처리
# 데이터의 올바른 처리를 위해 데이터 판단시 일정한 기준으로 맞춰주는 작업.

from operator import index
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


# 두 데이터 연결
fish_data = np.column_stack((fish_length, fish_weight))
# print(fish_data[:5])


# 타깃 데이터 설정
fish1 = np.ones(35)
fish2 = np.zeros(14)

# 타깃데이터 연결
fish_target = np.concatenate((fish1, fish2))
# print(fish_target)


# 랜덤으로 훈련데이터와 테스트 데이터 선정
# 원리 : 앞에 두 변수는 함수안 첫번째 변수안에서 뒤에 두 변수는 함수안 두번째 변수안에서 나눠짐.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 42)


# 제대로 나눠졌는지 크기 측정
# print(train_input.shape, test_input.shape)
# print(train_target.shape, test_target.shape)


# 하지만 무작위로 설정하면 test와 train의 비율이 샘플링 편향되어 있음.
# stratify 추가해서 해결
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify = fish_target, random_state = 42
    )


# 학습 시작
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
# print(kn.score(test_input, test_target))


# 검사
# print(kn.predict([[25, 150]]))
# 잘못된 검사 결과 출력


# 이유를 알기위해 plot 확인
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(20, 150, marker='^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# 기본값 5의 주변 이웃을 반환해보기
distances, indexes = kn.kneighbors([[25, 150]])
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(20, 150, marker='^')
# plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()
# print(train_input[indexes])
# print(train_target[indexes])
# print(distances)



# 이유
# kneighborsClassifier는 원리가 기본값 5개를 기준으로 주변에 많은 생선을 해당 생선의 종류로 인식
# 주변에 빙어가 많아서 빙어로 인식했음.
# 또한 그래프를 통해 확인했을때 x축의 범위가 y축의 범위가 넓어 y축과 조금만 멀어져도 엄청 먼거리로 생각
# x,y 축의 범위를 지정해줄 필요가 있음.


# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(20, 150, marker='^')
# plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
# plt.xlim(0, 1000) # x,y축의 범위를 같게 해줌.
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# 표준 점수를 통해 전처리
# 표준점수 = (점수-평균)/표준편차
mean = np.mean(train_input, axis=0) # axis=0 이면 세로의 평균
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean)/std # 자연스럽게 브로드 캐스팅(하나하나 전체 array에 적용)


# 전처리 완료한 데이터로 검사
new = ([25, 150] - mean)/std
# plt.scatter(train_scaled[:,0], train_scaled[:,1])
# plt.scatter(new[0], new[1], marker='^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# 학습시작
test_scaled = (test_input-mean)/std
kn.fit(train_scaled, train_target)
print(kn.score(test_scaled, test_target))

# 다시 검사
print(kn.predict([new]))

# 다시 최근접 점 5개 검사
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
