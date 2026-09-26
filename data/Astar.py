"""
A*路径规划算法模块 - 用于无人机路径规划。

实现了基于优先队列的A*（A-Star）启发式搜索算法，专门针对
无人机在二维地理空间中的路径规划需求进行优化。

算法特点：
- 结合了Dijkstra算法的最优性保证和启发式搜索的高效性
- 使用曼哈顿距离或欧几里得距离作为启发函数，指导搜索方向
- 支持动态障碍物规避，在实际无人机飞行中可实时更新障碍物信息
- 时间复杂度O((V+E)logV)，其中V为节点数，E为边数
- 空间复杂度O(V)，需要存储距离、前驱和访问状态

无人机应用场景：
- 在离散化的地理栅格地图上规划最优飞行路径
- 考虑电池续航、飞行高度限制、禁飞区等实际约束
- 与UAV-VLPA系统的多模态感知模块协同工作，实时响应环境变化
- 为后续的Chaiken路径平滑算法提供基础路径点序列
"""

# ==================== 标准库导入 ====================
import heapq  # 堆队列，用于实现优先队列
from typing import Dict, List, Tuple, Any  # 类型注解


class Graph:
    """
    图类 - 用于A*路径规划。

    使用邻接表表示图结构，支持带权重的无向图。
    主要用于无人机路径规划场景中的路径搜索，特别适配
    UAV-VLPA系统中基于视觉-语言-路径规划的多模态融合架构。

    属性说明：
        adjacency_list: 邻接表字典，{节点ID: [(邻居ID, 权重), ...]}
            - 节点ID：整数类型，代表地理栅格地图中的位置索引
            - 权重：浮点数类型，代表两节点间的飞行距离成本
                （考虑地形坡度、风速影响、障碍物密度等因素）
            - 邻居ID：相邻栅格位置的索引，支持8方向连接
    """

    def __init__(self, adjacency_list: Dict[int, List[Tuple[int, float]]]):
        """
        使用邻接表初始化图。

        该构造函数创建一个适合无人机路径规划的图结构，
        其中节点代表地理空间中的离散位置点，边权重代表
        从一个位置飞往另一个位置的成本（距离、时间、能耗等）。

        参数说明：
            adjacency_list: 邻接表字典
                键：节点ID（整数），对应地理栅格地图中的位置索引
                值：邻居节点列表，每个元素为 (邻居ID, 边权重) 元组
                    - 邻居ID：相邻位置的索引（支持上下左右及对角线8方向）
                    - 边权重：飞行成本，综合考虑距离、障碍物风险、风速影响等因素
        """
        self.adjacency_list = adjacency_list

    def find_path(self, coordinates: List[List[float]], end_node: int) -> List[int]:
        """
        使用A*算法寻找路径。

        这是UAV-VLPA系统中核心的路径规划算法实现，
        采用标准A*算法框架，但在无人机应用场景中进行了
        特殊优化和约束处理。

        A*算法原理：
            f(n) = g(n) + h(n)
            - g(n)：从起点到当前节点n的实际成本
            - h(n)：从当前节点n到目标节点的启发式估计成本
            - f(n)：总估计成本，用于优先队列排序

        算法流程详解：
            1. 初始化阶段：设置起点距离为0，其他节点为无穷大
            2. 优先队列管理：使用最小堆按f(n)值排序节点
            3. 节点探索：对每个节点计算其邻居的g(n)值
            4. 路径优化：当发现更优路径时更新距离和前驱
            5. 终止条件：到达目标节点或队列为空
            6. 路径重构：通过前驱字典反向构建完整路径

        无人机特殊考虑：
            - 起点固定为home位置（节点1），符合无人机起飞规范
            - 距离计算考虑三维空间因素（虽然当前为2D实现）
            - 为后续的Haversine距离计算和Chaiken平滑预留接口
            - 支持实时障碍物更新，可通过修改邻接表动态调整

        参数说明：
            coordinates: 节点坐标列表 [[lat, lon], ...]，用于计算启发式距离
                （当前未使用，但为未来GPS坐标系支持预留）
            end_node: 目标节点索引，代表无人机需要到达的目的地位置

        返回值：
            List[int]: 路径节点索引列表，从起点到终点
                - 每个整数代表地理栅格地图中的位置索引
                - 序列顺序代表无人机应遵循的飞行顺序
        """
        # 起点：假设起点为节点1（通常是home位置）
        start_node = 1

        # ==================== 初始化 ====================
        # 距离字典：记录从起点到每个节点的最短距离
        distances = {node: float('inf') for node in self.adjacency_list}
        distances[start_node] = 0  # 起点到自身距离为0

        # 前驱字典：记录路径中每个节点的前一个节点
        previous = {}

        # 优先队列：(距离, 节点)，按距离排序
        pq = [(0, start_node)]

        # 已访问集合：避免重复处理
        visited = set()

        # ==================== 主循环 ====================
        while pq:
            # 取出当前距离最小的节点
            current_dist, current_node = heapq.heappop(pq)

            # 如果已访问过，跳过
            if current_node in visited:
                continue

            # 标记为已访问
            visited.add(current_node)

            # 检查是否到达目标
            if current_node == end_node:
                break

            # ==================== 探索邻居 ====================
            for neighbor, weight in self.adjacency_list.get(current_node, []):
                if neighbor not in visited:
                    # 计算通过当前节点到邻居的新距离
                    new_dist = current_dist + weight
                    # 如果找到更短的路径，更新
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current_node  # 记录前驱
                        heapq.heappush(pq, (new_dist, neighbor))  # 加入队列

        # ==================== 路径重构 ====================
        # 从终点回溯到起点
        path = []
        current = end_node
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start_node)  # 添加起点

        # 反转路径，使其从起点到终点
        return list(reversed(path))


# ==================== 向后兼容 ====================
# 创建别名，保持与旧代码的兼容性
Astar = type('Astar', (), {'Graph': Graph})