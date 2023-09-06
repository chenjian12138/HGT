import torch
import torch.nn as nn

from dhg.structure.graphs import Graph
from dhg.structure.hypergraphs import Hypergraph



class HGTConv(nn.Module):
    """我们的模型

    Args:
        ``in_channels`` (``int``): :math:`C_` is the number of input channels.
        ``out_channels`` (``int``): :math:`C_` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. If ``dropout <= 0``, the layer will not drop values. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to ``0.2``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_src = nn.Linear(out_channels, 1, bias=False)
        self.atten_dst = nn.Linear(out_channels, 1, bias=False)


    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        x_for_src = self.atten_src(X)
        x_for_dst = self.atten_dst(X)
        e_atten_score = x_for_src[hg.e2v_src] + x_for_dst[hg.e2v_dst]
        e_atten_score = self.atten_dropout(self.atten_act(e_atten_score).squeeze())
        
        # # We suggest to add a clamp on attention weight to avoid Nan error in training.
        e_atten_score = torch.clamp(e_atten_score, min=0.001, max=5)
        
        X = hg.smoothing_with_HGNN(X)
        X = hg.v2v(X, aggr="softmax_then_sum", e2v_weight=e_atten_score)
        # X = hg.v2v(X, aggr="softmax_then_sum", e_weight=e_atten_score)
        X = hg.v2v(X, aggr="mean",v2e_aggr='softmax_then_sum')
        
        # if not self.is_last:
        #     X = self.act(X)
        
        if not self.is_last:
            X = self.drop(self.act(X))
        return X
