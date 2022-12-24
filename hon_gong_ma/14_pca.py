
# 주성분 분석 알고리즘 pca
# 주성분은 데이터 분류에 있어서 필요한 특징으로 생각.
# 차원(특성)을 축소함으로써 데이터의 크기를 줄임.
# 너무 많은 데이터로 인한 용량문제에서 용이하겠다.



# 데이터 준비
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# 주성분 분석
from sklearn.decomposition import PCA
pca = PCA(n_components=50) # n_components - 주성분의 개수
pca.fit(fruits_2d)
# print(pca.components_.shape)
# 결과(50, 10000) - 주성분의 개수, 특성의 개수


import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n=len(arr) # 샘플의 개수를 n에 저장
    rows = int(np.ceil(n/10)) # 한행당 10개씩
    cols = n if rows < 2 else 10 # 열의 수는 row가 2 보다 작지 않으면 10
    fig, axs = plt.subplots(rows, cols, 
        figsize = (cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10+j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()


# 주성분을 그림으로 그려봄
# draw_fruits(pca.components_.reshape(-1, 100, 100))


# 원래 10000개의 특성을 가진 300개의 이미지였지만
# 이를 50개의 특성(주성분)을 가진 300개의 이미지로 변경
# print(fruits_2d.shape)
fruits_pca = pca.transform(fruits_2d)
# print(fruits_pca.shape)


# 데이터의 차원(특성)을 줄였으니 복구도 가능할까?
# 가능 하네용
fruits_inverse = pca.inverse_transform(fruits_pca)
# print(fruits_inverse.shape)


# 다시 2d를 3d로 바꾸어 사진 출력해보자
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
# for start in [0, 100, 200]:
#     draw_fruits(fruits_reconstruct[start:start+100])
#     print("\n")


# 설명된 분산
# 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지를 기록한 것.
# print(np.sum(pca.explained_variance_ratio_))
# 결과는 92가 나오는데 92퍼센트 정도의 원본데이터를 보존하고 있다.



# 적절한 주성분의 개수 찾기
# plt.plot(pca.explained_variance_ratio_)
# plt.show()
# 결과를 보면 10이상의 주성분 부터는 비슷하다.


# pca로 축소된 데이터를 다른 알고리즘에 적용해보기
# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
target = np.array([0]*100 + [1]*100 + [2]*100)
# 사과, 파인애플, 바나나

# 원본데이터를 활용한 교차검증
from sklearn.model_selection import cross_validate
# scores = cross_validate(lr, fruits_2d, target)
# print(np.mean(scores['test_score']))
# print(np.mean(scores['fit_time'])) # 검증 시간 출력


# pca로 축소된 데이터를 활용한 교차검증
scores = cross_validate(lr, fruits_pca, target)
# print(np.mean(scores['test_score']))
# print(np.mean(scores['fit_time'])) # 검증 시간 출력
# 결과 분석 : 정확도도 올라가고 시간도 단축됨.


# pca객체 생성시 주성분의 개수 대신에 원본데이터를 얼마나 잘 나타낼지 비율을 정할 수도 있음.
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
# print(pca.n_components_)
# 원본 데이터의 50퍼센트만 유지하기 위해서 주성분의 개수는 2개면 충분


# 2개의 특성만으로도 교차검증 점수가 좋을까?
fruits_pca = pca.transform(fruits_2d)
# print(fruits_pca.shape)
scores = cross_validate(lr, fruits_pca, target)
# print(np.mean(scores['test_score']))
# print(np.mean(scores['fit_time'])) # 검증 시간 출력
# 결과는 좋군


# k-평균 알고리즘으로 클러스터 찾아보기(분류해보기)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
# print(np.unique(km.labels_, return_counts=True))


# 그림으로 보기
# for label in range(0,3):
#     draw_fruits(fruits[km.labels_ == label])
#     print("\n")


# 산점도도 그려보기
# for label in range(0,3):
#     data = fruits_pca[km.labels_==label]
#     plt.scatter(data[:,0], data[:,1])
# plt.legend(['apple', 'pineapple', 'banana'])
# plt.show()
