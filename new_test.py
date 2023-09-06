import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Graph,Hypergraph
from dhg.data import Cora,Cooking200,CoauthorshipCora,CocitationPubmed
from dhg.random import set_seed
from dhg.models import GCN,HGNN,HGNNP,HyperGCN,HNHN,GAT,GIN
from dhg.metrics import GraphVertexClassificationEvaluator as g_Evaluator
from dhg.metrics import HypergraphVertexClassificationEvaluator as hg_Evaluator
from new_HGT import HGT

torch.autograd.set_detect_anomaly(True)

def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    # loss = F.nll_loss(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res

if __name__ == "__main__":
    set_seed(2022)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = hg_Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    # evaluator = g_Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    # data = Cooking200()
    # data = CocitationPubmed()
    data = CoauthorshipCora()
    # print(data)
    
    #训练cooking200 
    # X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    # 训练其他数据集
    X, lbl = data["features"], data["labels"]
    
    G = Hypergraph(data["num_vertices"], data["edge_list"])
    # print(type(G))
    
    # G = Graph.from_hypergraph_clique(G, weighted=True)
    # print(type(G))

    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]
    
    # 下面为cooking200数据集最佳参数
    # net = GCN(X.shape[1], 32, data["num_classes"], use_bn=True)
    # net = GAT(X.shape[1], 32, data["num_classes"], num_heads=8, use_bn=True,drop_rate=0.05)
    # net = GIN(X.shape[1], 64, data["num_classes"], num_layers=5,num_mlp_layers=2,train_eps=True,use_bn=True,drop_rate=0.05)
    # net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=True)
    # net = HyperGCN(X.shape[1], 32, data["num_classes"], use_mediator=True,fast=True,drop_rate=0.01)
    # net = HNHN(X.shape[1], 32, data["num_classes"], use_bn=True,drop_rate=0.05)
    # net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=True)
    # net = HGT(X.shape[1], 32, data["num_classes"], num_heads=8,use_bn=True,drop_rate=0.01,atten_neg_slope=0.2)    
    
    # 下面调试Cora数据集
    # net = GCN(data["dim_features"], 32, data["num_classes"], use_bn=True)
    # net = GAT(data["dim_features"], 32, data["num_classes"], num_heads=8, use_bn=True,drop_rate=0.05)
    # net = GIN(data["dim_features"], 64, data["num_classes"], num_layers=5,num_mlp_layers=2,train_eps=True,use_bn=True,drop_rate=0.05) 
    # net = HGNN(data["dim_features"], 16, data["num_classes"],use_bn=False,drop_rate=0.01)
    # net = HGNNP(data["dim_features"], 16, data["num_classes"],use_bn=False,drop_rate=0.01)
    # net = HyperGCN(data["dim_features"], 32, data["num_classes"], use_mediator=False,fast=True,drop_rate=0.55)
    # net = HNHN(data["dim_features"], 4, data["num_classes"],use_bn=True,drop_rate=0.01)
    net = HGT(data["dim_features"], 64, data["num_classes"], num_heads=32,use_bn=False,drop_rate=0.005,atten_neg_slope=0.1)   
    # net = HGT(data["dim_features"], 64, data["num_classes"], num_heads=32,use_bn=False,drop_rate=0.01,atten_neg_slope=0.1) 
    # net = HGT(data["dim_features"], 32, data["num_classes"], num_heads=8,use_bn=False,drop_rate=0.01,atten_neg_slope=0.1)   
    
    optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=5e-4)

    X, lbl = X.cuda(), lbl.cuda()
    G = G.to(device)
    net = net.cuda()

    best_state = None
    best_epoch, best_val = 0, 0
    print("模型构建完毕！开始训练")
    # print(net)
    for epoch in range(500):
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, G, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)