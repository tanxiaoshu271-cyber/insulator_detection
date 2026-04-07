import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)

    return a * mask



def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)
    #print("jloc shape:", jloc.shape)
    #print("jloc values:", jloc.tolist())

    #print("joff shape:", joff.shape)
    #print("joff values:", joff.tolist())

    scores, index = torch.topk(jloc, k=topk)
    # y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
    y = torch.div(index,width,rounding_mode='trunc').float()+ torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores>th], scores[scores>th]


# import numpy as np
# from scipy.ndimage import gaussian_filter
#
# def get_junctions(batch_heatmaps, topk=300,threshold=0.03, sigma=3):
#     batch_size = batch_heatmaps.shape[0]  # 获取 batch_size
#     num_joints = batch_heatmaps.shape[3]  # 获取关节点数量（channels）
#     height = batch_heatmaps.shape[2]  # 获取热力图的高度
#     width = batch_heatmaps.shape[1]  # 获取热力图的宽度
#
#     preds = []
#     maxvals = []
#     all_peaks_with_score = []
#
#     for i in range(batch_size):
#         all_peaks = []
#         max_val = []
#         peaks_score = []
#
#         for part in range(num_joints):
#             page_peaks = []
#             page_val = []
#             peaks_score_for_part = []
#
#             # 获取该关节点的热力图
#             map_ori = batch_heatmaps[i, :, :, part].detach().cpu().numpy()
#             # 对热力图进行高斯滤波
#             map = gaussian_filter(map_ori, sigma=sigma)
#
#             # 创建邻域滤波器（上下左右）
#             map_left = np.zeros(map.shape)
#             map_left[:, 1:] = map[:, :-1]
#             map_right = np.zeros(map.shape)
#             map_right[:, :-1] = map[:, 1:]
#             map_up = np.zeros(map.shape)
#             map_up[1:, :] = map[:-1, :]
#             map_down = np.zeros(map.shape)
#             map_down[:-1, :] = map[1:, :]
#
#             # 寻找符合条件的峰值
#             peaks_binary = np.logical_and.reduce(
#                 (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > threshold))
#             peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # (w, h)
#
#             # 获取每个峰值的坐标和分数
#             peaks_with_score = [(x[0], x[1], map_ori[x[1], x[0]]) for x in peaks if map_ori[x[1], x[0]] > 0]
#
#             # 排序，按分数从高到低
#             peaks_with_score = sorted(peaks_with_score, key=lambda x: x[2], reverse=True)
#
#             # 只保留前 topk 个峰值，且分数大于阈值
#             peaks_with_score = [peak for peak in peaks_with_score if peak[2] > threshold][:topk]
#
#             # 存储峰值坐标和分数
#             for peaks_ in peaks_with_score:
#                 if len(peaks_) != 0:
#                     page_peaks.append(peaks_[:2])  # 只保存坐标 (w, h)
#                     page_val.append(peaks_)
#                     peaks_score_for_part.append(peaks_)
#
#             all_peaks.append(page_peaks)
#             max_val.append(np.array(page_val))
#             peaks_score.append(peaks_score_for_part)
#
#         preds.append(all_peaks)
#         maxvals.append(max_val)
#         all_peaks_with_score.append(peaks_score)
#
#     return preds, maxvals, all_peaks_with_score





def plot_lines(lines, scale=1.0, color = 'red', **kwargs):
    if isinstance(lines, np.ndarray):
        plt.plot([lines[:,0]*scale,lines[:,2]*scale],[lines[:,1]*scale,lines[:,3]*scale],color=color,linestyle='-')
    else:
        lines_np = lines.detach().cpu().numpy()
        plt.plot([lines_np[:,0]*scale,lines_np[:,2]*scale],[lines_np[:,1]*scale,lines_np[:,3]*scale],color=color,linestyle='-')

    