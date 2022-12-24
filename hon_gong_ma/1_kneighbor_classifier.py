# 생선 분류 머신러닝
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# 도미의 길이와 무게 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 분류
# 클래스를 분류

# 빙어의 길이와 무게
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]



# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# 데이터 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight


# 합친 데이터 2차원 리스트로 만들기
# zip - 나열된 리스트에서 원소를 하나씩 꺼내줌
fish_data = [[l,w] for l, w in zip(length, weight)]

# print(fish_data)


# 도미는 35마리, 빙어는 14마리
fish_target = [1]*35 + [0]*14
# print(fish_target)


# k-최근접 이웃 - 근처에 있는 데이터끼리 묶어줌
# 데이터 분류를 도와줄 객체
kn = KNeighborsClassifier()


# 데이터 전달 및 훈련
kn.fit(fish_data, fish_target)


# 모델 평가 값이 1이면 가장 정확
# print(kn.score(fish_data, fish_target))


# 길이 30, 무게 600 이 도미인지 빙어인지 예측
# 도미는 1, 빙어는 0
# print(kn.predict([[30,600]]))


# 데이터 잘 들어갔는지 확인
# print(kn._fit_X)
# print(kn._y)

# kneighborsclassifier는 기본적으로 5개의 데이터를 참고하여 예측
# 변경 가능
# kn49 = KNeighborsClassifier(n_neighbors=49)


# 참고 데이터 49개중 35개가 도미라서 도미로 예측하게 될 가능성이 큼
# kn49.fit(fish_data, fish_target)
# print(kn49.score(fish_data, fish_target)) # 정확도 0.7로 줄어들음
# print(35/49)


# 정학도가 몇개의 데이터참고부터 1보다 낮아질까?
for n in range(5,50):
    kn.n_neighbors = n
    score= kn.score(fish_data, fish_target)
    if score < 1:
        print(n, score)
        break


# 18개의 데이터 참고부터 정확도 1보다 낮아짐.