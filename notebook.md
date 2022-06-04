# Study Note

## deepsort
超参数：
DEEPSORT:
  MODEL_TYPE: "osnet_x_25"
  MAX_DIST: 0.2 # The matching threshold. Samples with larger distance are considered an invalid match
  MAX_IOU_DISTANCE: 0.7 # Gating threshold. Associations with cost larger than this value are disregarded.
  MAX_AGE: 30 # Maximum number of missed misses before a track is deleted
  N_INIT: 3 # Number of frames that a track remains in initialization phase
  NN_BUDGET: 100 # Maximum size of the appearance descriptors gallery
`
class DeepSort(object):
    def __init__(self, model, device, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
        ## 1. 超参数初始化
        ## 2. 加载reid模型
        ## 3. 初始化tracker

        return
    ## 输入这一帧所有的检测结果。
    def update(self, bbox_xywh, confidences, classes, ori_img):
        ## 1. 调整检测框的格式，提取人体特征
        ## 2. 对每一个轨迹进行预测
        ## 3. 对每一个轨迹进行匹配更新， 输入 所有检测框，所有框的置信度和id， 所有框的人体特征
        ## 4. 输出跟踪框
        return

class Tracker:
    ## 所有轨迹的类
    def __init__(self, metric, max_iou_distance=0.9, max_age=30, n_init=3, _lambda=0):
        ## 1. 超参数初始化
        ## 2. 加载卡尔曼滤波
        return 0
    def predict(self):
        ## 运行卡尔曼滤波
        ## 卡尔曼滤波在做什么？
        ## 卡尔曼滤波主要是通过之前的检测框，预测一个本帧的检测框，
        ## 然后 这个预测框与本帧检测器检测到的检测框加权得到一个置信度更高的检测框。
        ## 当然 这个本帧检测器检测到的检测框是需要先计算距离配对得到的。
        ##  cx，cy: 边界框的中心点
        ##  w, h: 边界框的宽和高
        ##  vx, vy, vw, vh: 变化速度
        ## 

        return 0
    def update(self, detections, classes, confidences):
        ## 1. 轨迹与检测框进行匹配 self._match(detections)。 返回 成功的匹配，未匹配到的轨迹，未匹配到的检测框
        ## 2. 更新每一条轨迹的信息，状态。
              a. 对于成功匹配的轨迹 更新信息
              b. 对于未成功匹配的轨迹 若它处于启动状态，则直接删除
                                   若它处于生长状态，且miss了足够多次，则直接删除
              c. 对于未成功匹配的检测 初始化为一条轨迹
        ## 3. 更新active_track的特征。
        return 0
    def _match(self, detections):
        ## 1.  将轨迹分为 正在生长 和 不是正在生长（刚启动） 两种
        ## 2.  将正在生长的轨迹使用 appearance feature 进行轨迹生长
            相应的会得到 匹配成功的轨迹， 未匹配成功的轨迹， 未匹配成功的检测
        ## 3.  如果轨迹time_since_update == 1, 则再给一次IOU匹配的机会
               如果轨迹time_since_update != 1, 则扔进未匹配到的轨迹
        ## 4.  将上一步筛选出来的轨迹and刚启动的轨迹与剩下的未匹配的检测框进行IOU匹配。
        ## 5.  返回 成功的匹配，未匹配到的轨迹，未匹配到的检测框。
        ##  具体如何配对？？
        ##   第一步匹配：首先将正在生长的轨迹和所有的检测框进行匹配
        ##   计算两种距离 a、根据卡尔曼滤波算马氏距离
                        b、根据轨迹的特征（最多100个）和检测框的特征算最近距离。
        ##   上述两种距离可以得到一个距离矩阵  [len(tracks), len(detection)]
        ##   通过匈牙利匹配，得到轨迹与检测框的最佳的匹配 
        ##   第二步匹配：计算轨迹与检测框的iou，得到距离矩阵（其他步骤和上面相同）
        return 0

class Track:
    ## 单条轨迹的类
    ## 轨迹的状态：  待启动， 正生长， 删除
    def __init__(self, mean, covariance, track_id, class_id, conf, n_init, max_age,
                 feature=None):  
        ## 单条轨迹需要那些信息？
        ## 轨迹id， 轨迹状态， 轨迹起始帧号，轨迹最后一帧的帧号，轨迹长度，轨迹点
        ## 类别id， 轨迹置信度，time_since_update 轨迹预测所需的参数（KalmanFilter: self.mean self.covarivance）。 
        return
    def predict(self, kf):
        ## 做卡尔曼滤波预测。
        return
    def update(self, kf, detection, class_id, conf):
        ## 1. 更新轨迹相关信息
        ## 2. 用匹配成功的检测框 更新 self.mean 与 self.covariance（相当于 用预测信息与检测框修正出更准确的跟踪框）
        ## 3. 更新这条轨迹的人体特征
        return 
`




## ST-GCN




## Pipeline

1. 参数初始化