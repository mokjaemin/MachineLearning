
# k-평균 알고리즘
# 원리 : 빨간점을 옮기면서 클러스팅.

# 데이터 준비
import numpy as np
fruits = np.load('./fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
# 3차원(샘플수, 너비, 높이) -> 2차원(샘플수, 너비*높이) 로 변경


# 훈련
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
# n_clusters -> 클래스의 수로 생각.
km.fit(fruits_2d)


# 결과
# print(km.labels_)
# print(np.unique(km.labels_, return_counts=True))
# 91, 98, 111개로 나뉘었음.


# plt로 확인을 하기 위한 함수.
# 3차원 배열을 입력으로 받음.
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


# 0 그려보기
# draw_fruits(fruits[km.labels_==0])

# 1 그려보기
# draw_fruits(fruits[km.labels_==1])

# 2 그려보기
# draw_fruits(fruits[km.labels_==2])


# 클러스터 중심을 이미지로 표현하기
# draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)


# 훈련데이터 샘플과 클러스터까지(판단중심)의 거리
# print(km.transform(fruits_2d[100:101])) # 인덱스가 100인 샘플의 각 클러스터까지의 거리
# 결과는 0번째 클러스터까지의 거리와 제일 가깝다.

# predict로 확인
# print(km.predict(fruits_2d[100:101]))
# 마찬가지로 0번째와 가깝다고 생각.

# 0번째니까 파인에플이겠구나
# draw_fruits(fruits[100:101])
# 예상대로 였음.


# 최적의 클러스터 위치를 찾기위해 클러스터를 옮긴 횟수
# print(km.n_iter_)
# 4번 옮겼구나!


# 지금까지 위에서는 클래스의 개수를 지정함으로써 타깃에 대한 힌트를 얻었음.
# 하지만 타깃의 개수도 모른다면?
# 엘보우 방법 활용 - 클러스터의 개수를 늘려가면서 이너셔의 변화를 관찰하여 최적의 클러스터 개수를 찾는 방법.
# 이너셔(클러스터에 속한 샘플들이 얼마나 가깝게 모여있는지)
# 클러스터와 샘플간의 거리의 제곱, 클러스터 증가시 이너셔는 감소.
# 결론 : 엘보우 방법은 그래프에서 꺾여있는 부분이 최적의 클러스터 개수.
# inertia = []
# for k in range(2, 7):
#     km = KMeans(n_clusters=k, random_state=42)
#     km.fit(fruits_2d)
#     inertia.append(km.inertia_)
# plt.plot(range(2,7), inertia)
# plt.xlabel('K')
# plt.ylabel('Inertia')
# plt.show()
# 최적의 k는 3