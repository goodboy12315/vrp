import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from datetime import datetime, timedelta

# 全局变量
num_vehicles = 10
vehicle_capacity = 3
vehicle_battery = 32000  # 满电可以走多少米
service_time = 0.0083  # 每个站点的服务时间
speed = 5  # 速度为5 km/h,0.72元/km
fixed_cost_per_vehicle = 150000  # 每辆车的固定成本为1000单位金钱
vehicle_used = [0] * num_vehicles  # 添加一个变量来记录每辆车是否被使用

# 假设的节点坐标（包括配送中心在内）单位m
coordinates = np.array([
    (619 / 10000, 547 / 10000), (799 / 10000, 477 / 10000), (890 / 10000, 470 / 10000),
    (1005 / 10000, 570 / 10000), (1005 / 10000, 470 / 10000), (1010 / 10000, 260 / 10000),
    (1100 / 10000, 375 / 10000), (1265 / 10000, 435 / 10000), (1320 / 10000, 257 / 10000),
    (1070 / 10000, 490 / 10000), (930 / 10000, 200 / 10000), (670 / 10000, 500 / 10000),
    (540 / 10000, 190 / 10000), (250 / 10000, 370 / 10000)
])


# 假设的距离矩阵/m
distances = np.array([
[0,25,34.8,40.9,46.3,67.8,65.3,75.8,99.1,195.8,211.8,239.8,251.8,268.8],
[25,0,9.8,29.9,21.3,42.8,40.3,50.8,74.1,170.8,186.8,214.8,226.8,243.8],
[34.8,9.8,0,21.5,11.5,33,30.5,41,64.3,161,177,205,217,234],
[40.9,29.9,21.5,0,10,31.5,29,39.5,62.8,159.5,175.5,203.5,215.5,232.5],
[46.3,21.3,11.5,10,0,21.5,19,29.5,52.8,149.5,165.5,193.5,205.5,222.5],
[67.8,42.8,33,31.5,21.5,0,20.5,43,31.3,144,160,188,200,217],
[65.3,40.3,30.5,29,19,20.5,0,22.5,33.8,130.5,146.5,174.5,186.5,203.5],
[75.8,50.8,41,39.5,29.5,43,22.5,0,23.3,120,136,164,176,193],
[99.1,74.1,64.3,62.8,52.8,31.3,33.8,23.3,0,113.3,129.3,157.3,169.3,186.3],
[195.8,170.8,161,159.5,149.5,144,130.5,120,113.3,0,40,44,80,91],
[211.8,186.8,177,175.5,165.5,160,146.5,136,129.3,40,0,56,40,85],
[239.8,214.8,205,203.5,193.5,188,174.5,164,157.3,44,56,0,44,55],
[251.8,226.8,217,215.5,205.5,200,186.5,176,169.3,80,40,44,0,47],
[268.8,243.8,234,232.5,222.5,217,203.5,193,186.3,91,85,55,47,0]
])
distances=distances*10

# 工位需求量、时间窗/h
demands = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
time_windows = [(9.00, 9.25), (9.25, 9.50), (9.50, 9.75), (9.75, 10.00),
(10.00, 10.25), (10.25, 10.50), (10.50, 10.75), (10.75, 11.00), (11.00, 11.25),
(11.25, 11.50), (11.50, 11.75), (11.75, 12.00), (12.00, 12.25)]

# 计算行驶时间矩阵
travel_times = distances*0.001 / speed   # 转换为小时

# 在算法开始前设置固定的随机种子
random.seed(42)
np.random.seed(42)

# 初始化解：贪心初始化
def greedy_initialize_solution():
    solution = [[] for _ in range(num_vehicles)]
    nodes = list(range(1, len(demands) + 1))
    random.shuffle(nodes)
    visited_nodes = set()  # Track visited nodes
    for node in nodes:
        min_cost = float('inf')
        selected_vehicle = None
        for vehicle_id in range(num_vehicles):
            current_route = solution[vehicle_id]
            if node not in visited_nodes and (not current_route or node != current_route[-1]):
                current_solution = [route[:] for route in solution]
                current_solution[vehicle_id].append(node)
                current_cost_details = calculate_cost(current_solution)
                current_cost = current_cost_details['total_cost']
                if current_cost < min_cost:
                    min_cost = current_cost
                    selected_vehicle = vehicle_id
        if selected_vehicle is not None:
            solution[selected_vehicle].append(node)
            visited_nodes.add(node)
            vehicle_used[selected_vehicle] = 1  # 标记该车辆被使用
    return solution

def improved_initialize_solution():
    best_solution = None
    best_cost = float('inf')
    for _ in range(100):  # 运行多次贪心初始化
        solution = greedy_initialize_solution()
        cost = calculate_cost(solution)['total_cost']
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
    return best_solution

# 计算成本
def calculate_cost(solution):
    total_distance = 0
    distance_cost = 0
    penalty = 0
    fixed_cost = 0
    served_demand = 0  # 记录已服务的需求量
    total_demand = sum(demands)  # 所有需求点的总需求量
    node_arrival_times = {}  # 记录每个节点的到达时间
    node_battery_levels = {}  # 记录每个节点的剩余电量

    # 计算固定成本：固定成本与使用的车辆数目成正比
    used_vehicles = sum(vehicle_used)
    fixed_cost = used_vehicles * fixed_cost_per_vehicle

    # 检查所有需求点是否被服务，并计算已服务的需求量
    served_nodes = set()
    for vehicle_id, route in enumerate(solution):
        if not route:
            continue
        prev_node = 0  # 起点为配送中心
        load = 0
        battery_used = 0
        arrival_time = 0
        vehicle_battery_remaining = vehicle_battery  # 每辆车初始剩余电量为满电量

        for node in route:
            if prev_node == 0:
                travel_time = travel_times[prev_node][node]
                arrival_time = max(time_windows[node - 1][0] ,travel_time)
            else:
                travel_time = travel_times[prev_node][node]
                arrival_time = max(time_windows[node - 1][0], node_arrival_times[prev_node]+service_time+travel_time)
            node_arrival_times[node] = arrival_time  # 记录到达时间

            # 计算电量使用
            battery_used = distances[prev_node][node]
            vehicle_battery_remaining -= battery_used

            # 更新节点的剩余电量
            node_battery_levels[node] = vehicle_battery_remaining

            # 检查车辆容量约束
            load += demands[node - 1]
            if load > vehicle_capacity:
                penalty += 100000000000  # 违反容量约束

            # 检查时间窗约束
            if not (time_windows[node - 1][0] <= arrival_time <= time_windows[node - 1][1]):
                penalty += 110000  # 违反时间窗

            total_distance += distances[prev_node][node]
            distance_cost += distances[prev_node][node]
            prev_node = node

        # 计算返回配送中心的距离和成本
        total_distance += distances[prev_node][0]
        distance_cost += distances[prev_node][0]

        # 检查返回配送中心时的电量约束
        battery_used = distances[prev_node][0]
        vehicle_battery_remaining -= battery_used
        if vehicle_battery_remaining < 0:
            penalty += 10000000000  # 违反电量约束

        # 更新节点的剩余电量（返回配送中心）
        node_battery_levels[0] = vehicle_battery_remaining

    # 如果已服务的需求量不等于总需求量，添加惩罚成本
    served_demand = sum(demands[node - 1] for node in node_arrival_times)
    if served_demand != total_demand:
        penalty += 10000000000000000  # 非常大的惩罚成本，确保所有需求点都被服务

    return {
        'total_distance': total_distance,
        'distance_cost': distance_cost * 10,
        'penalty': penalty,
        'fixed_cost': fixed_cost,
        'total_cost': total_distance*10 + penalty + fixed_cost,
        'node_arrival_times': node_arrival_times,  # 返回节点到达时间信息
        'node_battery_levels': node_battery_levels  # 返回节点剩余电量信息
    }

# 扰动操作
def shake(solution, k):
    new_solution = [route[:] for route in solution]
    for _ in range(k):
        vehicle1, vehicle2 = np.random.choice(num_vehicles, 2, replace=False)
        if new_solution[vehicle1] and new_solution[vehicle2]:
            index1, index2 = np.random.randint(len(new_solution[vehicle1])), np.random.randint(len(new_solution[vehicle2]))
            # 交换操作
            new_solution[vehicle1][index1], new_solution[vehicle2][index2] = new_solution[vehicle2][index2], new_solution[vehicle1][index1]
        # 单一车辆内部的重新插入操作
        if new_solution[vehicle1]:
            route = new_solution[vehicle1]
            node = route.pop(np.random.randint(len(route)))
            route.insert(np.random.randint(len(route)+1), node)
    return new_solution

# 局部搜索：2-opt算法
def two_opt(route):
    best_route = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_cost = calculate_cost([new_route])['total_cost']
                if new_cost < calculate_cost([best_route])['total_cost']:
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
        route = best_route
    return best_route

# 局部搜索：对每个车辆的路线应用2-opt算法进行局部优化
def local_search(solution):
    improved_solution = []
    for route in solution:
        improved_route = two_opt(route)
        improved_solution.append(improved_route)
    return improved_solution

# VNS主循环
def variable_neighborhood_search(max_iterations=1000):
    current_solution = greedy_initialize_solution()
    update_vehicle_used(current_solution)  # 初始解后更新 vehicle_used
    best_solution = current_solution
    best_cost_details = calculate_cost(current_solution)
    best_cost = best_cost_details['total_cost']

    for iteration in range(max_iterations):
        for k in range(1, 5):  # 更换邻域结构
            new_solution = shake(current_solution, k)
            new_solution = local_search(new_solution)
            update_vehicle_used(new_solution)  # 每次迭代后更新 vehicle_used
            new_cost_details = calculate_cost(new_solution)
            new_cost = new_cost_details['total_cost']

            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
                best_cost_details = new_cost_details
                break

        current_solution = best_solution
        print(f"Iteration {iteration + 1}, Best Cost: {best_cost}")
    # 更新最佳解时，再次计算固定成本
    used_vehicles = sum(vehicle_used)
    best_cost_details['fixed_cost'] = used_vehicles * fixed_cost_per_vehicle
    best_cost_details['total_cost'] = best_cost_details['total_distance'] * 10 + best_cost_details['penalty'] + best_cost_details['fixed_cost']

    print(f"Optimized Total Cost: {best_cost}")
    print("Detailed Cost Breakdown:")
    print(f"  total_distance: {best_cost_details['total_distance']}")
    print(f"  Total Distance Cost: {best_cost_details['distance_cost']}")
    print(f"  Penalty Cost: {best_cost_details['penalty']}")
    print(f"  Fixed Cost: {best_cost_details['fixed_cost']}")

    return best_solution

# 更新 vehicle_used 函数
def update_vehicle_used(solution):
    global vehicle_used
    vehicle_used = [0] * num_vehicles
    for vehicle_id, route in enumerate(solution):
        if route:
            vehicle_used[vehicle_id] = 1

def hours_to_hms(node_arrival_times):
    # 将小时转换为秒
    total_seconds = int(node_arrival_times * 3600)
    # 创建一个 timedelta 对象来表示这些秒数
    delta = timedelta(seconds=total_seconds)
    # 用一个固定的日期（这里选择1970-01-01）加上这个 timedelta 得到一个 datetime 对象
    base_time = datetime(1970, 1, 1) + delta
    # 格式化输出为 HH:MM:SS
    formatted_time = base_time.strftime('%H:%M:%S')
    return formatted_time

# 可视化最终结果，包括节点电量信息和离开时间信息
def plot_solution(solution):
    plt.figure(figsize=(12, 8))
    for vehicle_id, route in enumerate(solution):
        x, y = [619 / 10000], [547 / 10000]  # depot coordinates
        node_arrival_times = calculate_cost([route])['node_arrival_times']
        node_battery_levels = calculate_cost([route])['node_battery_levels']
        departure_times = {node: 0 for node in route}  # 初始化离开时间为0
        prev_node = 0  # depot
        arrival_time = 0
        vehicle_battery_remaining = vehicle_battery

        for node in route:
            node_arrival_times[node] = hours_to_hms(node_arrival_times[node])
            x.append(coordinates[node][0])
            y.append(coordinates[node][1])
            plt.text(coordinates[node][0], coordinates[node][1], f"node: {node} \narrival_time: {node_arrival_times[node]:} \nbattery: {node_battery_levels[node]/vehicle_battery*100:.1f}%", fontsize=6)


        x.append(619/10000)
        y.append(547/10000)
        plt.plot(x, y, marker='o', label=f'Vehicle {vehicle_id}')

    plt.scatter([619/10000], [547/10000], color='red', label='Depot')
    plt.legend()
    plt.grid(True)
    plt.title("Final Solution\n")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# 执行VNS算法
final_solution = variable_neighborhood_search()

# 输出最终解决方案和详细成本信息
print("\nFinal Solution:")
for idx, route in enumerate(final_solution):
    print(f"Vehicle {idx}: {route}")

# 可视化最终结果
plot_solution(final_solution)
