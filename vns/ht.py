import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 节点位置和电梯坐标
node_positions = [
    (619, 547, 0), (799, 477, 0), (890, 470, 0),
    (1005, 570, 0), (1005, 470, 0), (1010, 260, 0),
    (1100, 375, 0), (1265, 435, 0), (1320, 257, 0),
    (1070, 490, 500), (930, 200, 500), (670, 500, 500),
    (540, 190, 500), (250, 370, 500), (1450, 340, 0), (1450, 340, 500)
]

# 转换为 Numpy 数组
node_coordinates = np.array(node_positions)

# 路线计划，0是起点，1-8是一层需求点，9-13是二层需求点，14是一层电梯，15是二层电梯
distributionPlan = [
    [0, 8, 14, 15, 9, 10, 15, 14, 0],
    [0, 3, 4, 0],
    [0, 5, 6, 7, 0],
    [0, 14, 15, 11, 12, 13, 15, 14, 0],
    [0, 1, 2, 0]
]

# 路径颜色
colors = ['b', 'g', 'c', 'm', 'y', 'k']

# 绘图函数
def plot_routes():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 标注节点
    node_labels = [str(i) for i in range(16)]
    for idx, coord in enumerate(node_coordinates):
        if idx == 0:
            ax.scatter(*coord, color='green', marker='o', s=100)
            ax.text(coord[0], coord[1] + 100, coord[2], 'Start', color='black', fontsize=16, ha='right')
        elif idx == 14:
            ax.scatter(*coord, color='blue', marker='^', s=100)
            ax.text(coord[0], coord[1] + 80, coord[2], 'lift', color='black', fontsize=16, ha='right')
        elif idx == 15:
            ax.scatter(*coord, color='blue', marker='^', s=100)
            ax.text(coord[0], coord[1] + 80, coord[2], 'lift', color='black', fontsize=16, ha='right')
        else:
            ax.scatter(*coord, color='red', marker='*', s=100)
            ax.text(coord[0], coord[1], coord[2], node_labels[idx], color='black', fontsize=16, ha='right')

    # 绘制路径并添加箭头
    for path_idx, path in enumerate(distributionPlan):
        path_color = colors[path_idx % len(colors)]
        for i in range(len(path) - 1):
            start = node_coordinates[path[i]]
            end = node_coordinates[path[i+1]]
            # 绘制线段
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=path_color, lw=2)

    # 添加楼层网格
    for z in [0, 500]:
        X, Y = np.meshgrid(np.linspace(200, 1500, 5), np.linspace(0, 600, 5))
        ax.plot_surface(X, Y, z * np.ones_like(X), color='gray', alpha=0.3)

    # 适当调整视图角度
    ax.view_init(elev=55, azim=45)

    plt.show()

# 绘制路径图
plot_routes()