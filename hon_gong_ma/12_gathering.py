
# 군집 알고리즘
# 입력은 있고 타깃은 없을때!
# 입력 사진들을 이용하여 클래스 분류



# 데이터 준비
import numpy as np
import matplotlib.pyplot as plt
fruits = np.load('./fruits_300.npy')
# print(fruits.shape)


# 첫번째 사진의 첫번째 행 불러보기
# print(fruits[0, 0, :])


# plt로 보기
# plt.imshow(fruits[0], cmap='gray')
# plt.show()


# 반전 효과 주고 다시 보기
# plt.imshow(fruits[0], cmap='gray_r')
# plt.show()

# 나머지 과일도 그려보기
# 여러개의 그래프를 한번에 그리기에 용이한 subplots 이용.
# 0~99 - 사과, 100~199 - 파인애플, 200~299 - 바나나
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(fruits[100], cmap='gray_r')
# axs[1].imshow(fruits[200], cmap='gray_r')
# plt.show()


# 과일 배열 일차원으로 만들기
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
# print(apple.shape)


# 픽셀의 평균 구하기
# print(apple.mean(axis=1))
# axis = 1 -> 가로가로가로가로 평균


# 히스토그램으로 픽셀값 평균의 빈도 확인.
# plt.hist(apple.mean(axis=1), alpha=0.8)
# plt.hist(pineapple.mean(axis=1), alpha=0.8)
# plt.hist(banana.mean(axis=1), alpha=0.8)
# plt.legend(['apple', 'pineapple', 'banana'])
# plt.show()
# 결과를 보면 바나나는 픽셀의 평균값으로도 나머지 클래스와 구별 가능.


# 다른방법 고안
# 각사진마다 안에있는 픽셀의 평균값을 구해보자
# fig, axs = plt.subplots(1, 3, figsize=(20,5))
# axs[0].bar(range(10000), np.mean(apple, axis=0))
# axs[1].bar(range(10000), np.mean(pineapple, axis=0))
# axs[2].bar(range(10000), np.mean(banana, axis=0))
# plt.show()
# 결과분석
# 사과는 사진의 픽셀순서 마지막으로 갈수록 값이 높아지고
# 파인애플은 비교적 골고루 분포
# 바나나는 비교적 중앙쪽이 높다.


# 위의 픽셀 평균을 이미지처럼 표현
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
# fig, axs = plt.subplots(1, 3, figsize=(20,5))
# axs[0].imshow(apple_mean, cmap='gray_r')
# axs[1].imshow(pineapple_mean, cmap='gray_r')
# axs[2].imshow(banana_mean, cmap='gray_r')
# plt.show()



# 절대값 오차를 활용하여 평균값과 가까운 사진 고르기
abs_diff = np.abs(fruits - apple_mean) # abs - 절대값을 구하는 함수
abs_mean = np.mean(abs_diff, axis=(1,2)) 
# 원리는 이해가 안되나 1차원배열에 크기는 300인 오차평균을 구하게 됨.


# apple_mean 과 오차가 가장 작은 샘플 100개 골라보기
apple_index = np.argsort(abs_mean)[:100]
# argsort 는 작은 것에서 큰 순서대로 나열한 abs_mean 배열의 인덱스를 반환함.
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off') # 좌표축을 그리지 않음.
# plt.show()