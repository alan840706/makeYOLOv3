# coding=utf-8
# k-means ++ for YOLOv2 anchors
# 通過k-means ++ 演算法獲取YOLOv2需要的anchors的尺寸
import numpy as np

# 定義Box類，描述bounding box的座標
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# 計算兩個box在某個軸上的重疊部分
# x1是box1的中心在該軸上的座標
# len1是box1在該軸上的長度
# x2是box2的中心在該軸上的座標
# len2是box2在該軸上的長度
# 返回值是該軸上重疊的長度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 計算box a 和box b 的交集面積
# a和b都是Box型別例項
# 返回值area是box a 和box b 的交集面積
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 計算 box a 和 box b 的並集面積
# a和b都是Box型別例項
# 返回值u是box a 和box b 的並集面積
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 計算 box a 和 box b 的 iou
# a和b都是Box型別例項
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# 使用k-means ++ 初始化 centroids，減少隨機初始化的centroids對最終結果的影響
# boxes是所有bounding boxes的Box物件列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors個centroid
def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num, 1)
    centroids.append(boxes[centroid_index])

    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0,n_anchors-1):

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break

    return centroids


# 進行 k-means 計算新的centroids
# boxes是所有bounding boxes的Box物件列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是計算出的新簇中心
# 返回值groups是n_anchors個簇包含的boxes的列表
# 返回值loss是所有box距離所屬的最近的centroid的距離的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i]) #這裡涉及到了距離的跟新，作者直接使用平均w作為新的寬
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


# 計算給定bounding boxes的n_anchors數量的centroids
# label_path是訓練集列表檔案地址
# n_anchors 是anchors的數量
# loss_convergence是允許的loss的最小變化值
# grid_size * grid_size 是柵格數量
# iterations_num是最大迭代次數
# plus = 1時啟用k means ++ 初始化centroids
def compute_centroids(label_path,n_anchors,loss_convergence,grid_size,iterations_num,plus):

    boxes = []
    label_files = []
    f = open(label_path)
    for line in f:
        label_path = line.rstrip().replace('images', 'labels')
        label_path = label_path.replace('JPEGImages', 'labels')
        label_path = label_path.replace('.jpg', '.txt')
        label_path = label_path.replace('.JPEG', '.txt')
        label_path = label_path.replace('.png', '.txt')
        label_files.append(label_path)
    f.close()

    for label_file in label_files:
        with open(label_file, 'rb') as f:
          for line in f:
              temp = line.decode().strip().split(" ")
              if len(temp) > 1:
                  boxes.append(Box(0, 0, float(temp[3]), float(temp[4])))

    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        print("loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break
        old_loss = loss

        for centroid in centroids:
            print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    print("k-means result：\n")
    count=0
    for centroid in centroids:
        if(count==0):
            buff=str(centroid.w * grid_size)+","+str( centroid.h * grid_size)
        else:
            buff=buff+","+str(centroid.w * grid_size)+","+str( centroid.h * grid_size)
        count+=1
    print(buff)
label_path = "/cfg.person/train.txt"
n_anchors = 5
loss_convergence = 1e-6
grid_size = 13
iterations_num = 100
plus = 0
compute_centroids(label_path,n_anchors,loss_convergence,grid_size,iterations_num,plus)
