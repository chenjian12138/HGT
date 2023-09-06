import torch
import dhg 
from typing import List
def hspd_encoding(X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:

    num_nodes = hg.num_v
    device = hg.device

    hspd_enc = torch.zeros((num_nodes, X.size(1))).to(device)

    distances = compute_shortest_path_distances(hg)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                relative_dist = abs(distances[i][j] - distances[j][i])
                hspd_enc[i] += X[j] * relative_dist

    return hspd_enc

def compute_shortest_path_distances(hg: "dhg.Hypergraph") -> List[List[float]]:
    # 使用Dijkstra算法计算最短路径距离
    num_nodes = hg.num_v
    distances = [[float('inf')] * num_nodes for _ in range(num_nodes)]

    # 初始化距离矩阵，将节点与自身的距离设置为0
    for i in range(num_nodes):
        distances[i][i] = 0

    # 遍历超图的边，更新节点之间的距离
    for edge in hg.edges():
        # 假设每条边都连接两个节点，如果超图中的边连接多个节点，需要根据实际情况进行修改
        node1, node2 = edge.head, edge.tail[0]

        # 根据节点之间的连接关系更新距离矩阵
        distances[node1][node2] = 1
        distances[node2][node1] = 1

    # 使用Dijkstra算法计算最短路径距离
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]

    return distances