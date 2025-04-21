import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
import matplotlib.pyplot as plt  

# 定义路径绘制函数
def draw_path(car_routes, CityCoordinates):
    '''
    画路径图
    输入：car_routes-车辆路径列表，CityCoordinates-城市坐标；
    输出：路径图
    '''
    for route in car_routes:
        x, y = [], []
        for i in route:
            Coordinate = CityCoordinates[i]
            x.append(Coordinate[0])
            y.append(Coordinate[1])
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Vehicle Routes')
    plt.show()

# 用字典存储所有参数
# 节点坐标，单位是0.1m，转换为km
dataDict = {
    'NodeCoor': [
        (619 / 10000, 547 / 10000), (799 / 10000, 477 / 10000), (890 / 10000, 470 / 10000),
        (1005 / 10000, 570 / 10000), (1005 / 10000, 470 / 10000), (1010 / 10000, 260 / 10000),
        (1100 / 10000, 375 / 10000), (1265 / 10000, 435 / 10000), (1320 / 10000, 257 / 10000),
        (1070 / 10000, 490 / 10000), (930 / 10000, 200 / 10000), (670 / 10000, 500 / 10000),
        (540 / 10000, 190 / 10000), (250 / 10000, 370 / 10000)
    ],
    'Demand': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],# 将配送中心的需求设置为0
    'Timewindow': [
        (0, 24),  # 配送中心，不需要时间窗约束
    (9.00, 9.25), (9.25, 9.50), (9.50, 9.75), (9.75, 10.00),
    (10.00, 10.25), (10.25, 10.50), (10.50, 10.75), (10.75, 11.00), (11.00, 11.25),
    (11.25, 11.50), (11.50, 11.75), (11.75, 12.00), (12.00, 12.25)
    ],
    'MaxLoad': 3,# 车辆最大容量
    'ServiceTime': 0.0083,
    'Velocity': 5,
    'unit_distance_cost': 18,
    'unit_vehicle_fixed_cost': 150000,
    'Vehicles': [1, 2, 3, 4, 5],
    'BatteryCapacity': 32,  # 电池容量
    'EnergyConsumptionPerUnitDistance': 1  # 每单位距离的耗电量
}

# 节点总数
n = len(dataDict['NodeCoor'])
# 距离矩阵
distance_matrix = [[0] * n for _ in range(n)]

# 曼哈顿距离计算
distance_matrix = np.array([
[0.0, 0.025, 0.0348, 0.0409, 0.0463, 0.0678, 0.0653, 0.0758, 0.0991, 0.1958, 0.2118, 0.2398, 0.2518, 0.2688],
    [0.025, 0.0, 0.0098, 0.0299, 0.0213, 0.0428, 0.0403, 0.0508, 0.0741, 0.1708, 0.1868, 0.2148, 0.2268, 0.2438],
    [0.0348, 0.0098, 0.0, 0.0215, 0.0115, 0.033, 0.0305, 0.041, 0.0643, 0.161, 0.177, 0.205, 0.217, 0.234],
    [0.0409, 0.0299, 0.0215, 0.0, 0.01, 0.0315, 0.029, 0.0395, 0.0628, 0.1595, 0.1755, 0.2035, 0.2155, 0.2325],
    [0.0463, 0.0213, 0.0115, 0.01, 0.0, 0.0215, 0.019, 0.0295, 0.0528, 0.1495, 0.1655, 0.1935, 0.2055, 0.2225],
    [0.0678, 0.0428, 0.033, 0.0315, 0.0215, 0.0, 0.0205, 0.043, 0.0313, 0.144, 0.16, 0.188, 0.2, 0.217],
    [0.0653, 0.0403, 0.0305, 0.029, 0.019, 0.0205, 0.0, 0.0225, 0.0338, 0.1305, 0.1465, 0.1745, 0.1865, 0.2035],
    [0.0758, 0.0508, 0.041, 0.0395, 0.0295, 0.043, 0.0225, 0.0, 0.0233, 0.12, 0.136, 0.164, 0.176, 0.193],
    [0.0991, 0.0741, 0.0643, 0.0628, 0.0528, 0.0313, 0.0338, 0.0233, 0.0, 0.1133, 0.1293, 0.1573, 0.1693, 0.1863],
    [0.1958, 0.1708, 0.161, 0.1595, 0.1495, 0.144, 0.1305, 0.12, 0.1133, 0.0, 0.04, 0.044, 0.08, 0.091],
    [0.2118, 0.1868, 0.177, 0.1755, 0.1655, 0.16, 0.1465, 0.136, 0.1293, 0.04, 0.0, 0.056, 0.04, 0.085],
    [0.2398, 0.2148, 0.205, 0.2035, 0.1935, 0.188, 0.1745, 0.164, 0.1573, 0.044, 0.056, 0.0, 0.044, 0.055],
    [0.2518, 0.2268, 0.217, 0.2155, 0.2055, 0.2, 0.1865, 0.176, 0.1693, 0.08, 0.04, 0.044, 0.0, 0.047],
    [0.2688, 0.2438, 0.234, 0.2325, 0.2225, 0.217, 0.2035, 0.193, 0.1863, 0.091, 0.085, 0.055, 0.047, 0.0]
])
dataDict['DistanceMatrix'] = distance_matrix

# 创建模型
model = gp.Model("VRPTW")

# 添加变量
x = {}
for k in dataDict['Vehicles']:
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

# 添加时间变量
arrival_time = {}
for i in range(n):
    for k in dataDict['Vehicles']:
        arrival_time[i, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"arrival_time_{i}_{k}")

# 添加电量变量
energy_used = {}
for k in dataDict['Vehicles']:
    for i in range(n):
        energy_used[i, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"energy_used_{i}_{k}")

# 设置目标函数
model.setObjective(
    dataDict['unit_vehicle_fixed_cost'] * quicksum(quicksum(x[0, j, k] for j in range(1, n)) for k in dataDict['Vehicles']) +
    dataDict['unit_distance_cost'] * quicksum(dataDict['DistanceMatrix'][i][j] * x[i, j, k] for i in range(n) for j in range(n) for k in dataDict['Vehicles'] if i != j),
    sense=GRB.MINIMIZE
)

# 添加约束
# 每个顾客节点必须被一辆车访问一次
for i in range(1, n):  # 节点 1 到 n-1 是顾客节点，0 是配送中心
    model.addConstr(quicksum(x[i, j, k] for j in range(n) if i != j for k in dataDict['Vehicles']) == 1, name=f"visit_{i}")

# 每辆车从配送中心出发并返回配送中心
for k in dataDict['Vehicles']:
    model.addConstr(quicksum(x[0, j, k] for j in range(1, n)) == 1, name=f"start_from_depot_{k}")
    model.addConstr(quicksum(x[j, 0, k] for j in range(1, n)) == 1, name=f"return_to_depot_{k}")

# 添加车辆负载约束
for k in dataDict['Vehicles']:
    model.addConstr(quicksum(dataDict['Demand'][i] * quicksum(x[i, j, k] for j in range(n) if i != j) for i in range(1, n)) <= dataDict['MaxLoad'], name=f"load_{k}")

# 添加时间窗口约束
for i in range(1, n):  # 节点 1 到 n-1 是顾客节点，0 是配送中心
    for k in dataDict['Vehicles']:
        model.addConstr(arrival_time[i, k] >= dataDict['Timewindow'][i][0], name=f"earliest_arrival_{i}_{k}")
        model.addConstr(arrival_time[i, k] <= dataDict['Timewindow'][i][1], name=f"latest_arrival_{i}_{k}")

# 连续性约束：如果车辆k从节点i到达节点j，则它在节点i的到达时间加上服务时间和行驶时间等于它在节点j的到达时间
for i in range(n):
    for j in range(1, n):
        if i != j:
            for k in dataDict['Vehicles']:
                model.addConstr(arrival_time[j, k] >= arrival_time[i, k] + dataDict['ServiceTime'] + dataDict['DistanceMatrix'][i][j] / dataDict['Velocity'] - (1 - x[i, j, k]) * 1e6, name=f"time_continuity_{i}_{j}_{k}")

# 添加车流量平衡约束
for k in dataDict['Vehicles']:
    for i in range(n):
        model.addConstr(quicksum(x[i, j, k] for j in range(n) if i != j) == quicksum(x[j, i, k] for j in range(n) if i != j), name=f"path_continuity_{i}_{k}")

# 添加电量约束
for k in dataDict['Vehicles']:
    model.addConstr(quicksum(dataDict['DistanceMatrix'][i][j] * x[i, j, k] * dataDict['EnergyConsumptionPerUnitDistance'] for i in range(n) for j in range(n) if i != j) <= dataDict['BatteryCapacity'], name=f"battery_capacity_{k}")

# 求解模型
model.optimize()

# 如果模型无解，使用冲突求解器
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible. Identifying conflicts...")
    model.computeIIS()
    model.write("model.ilp")
    if model.IISMinimal:
        print("\nThe following constraints are causing the model to be infeasible:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"{c.constrName}")
    else:
        print("\nThe model is infeasible, but Gurobi could not identify minimal infeasible constraints.")
else:
    # 输出结果
    if model.status == GRB.OPTIMAL:
        print("Optimal objective:", model.objVal)
        car_routes = []
        for k in dataDict['Vehicles']:
            route = []
            for i in range(n):
                for j in range(n):
                    if i != j and x[i, j, k].x > 0.5:
                        route.append(i)
            car_routes.append(route)
        
        # 输出每辆车的路径
        for k, route in enumerate(car_routes):
            print(f"\nVehicle {k + 1} route:")
            print(route)
        
        # 绘制路径图
        draw_path(car_routes, dataDict['NodeCoor'])
        
        # 计算和输出每辆车的总能量消耗
        for k, route in enumerate(car_routes):
            energy_used_total = sum(dataDict['DistanceMatrix'][route[i]][route[i + 1]] * dataDict['EnergyConsumptionPerUnitDistance'] for i in range(len(route) - 1))
            print(f"\nVehicle {k + 1} total energy used:", energy_used_total)
    else:
        print("No optimal solution found")