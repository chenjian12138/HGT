import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import dhg
from dhg.nn import MultiHeadWrapper
from dhg.nn import HGNNPConv
from HGT_conv import HGTConv
import math
from p_laplacian import p_laplacian
from hspd_encoding import hspd_encoding
import numpy as np

class HGT(nn.Module):
    """
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``num_heads`` (``int``): The Number of attention head in each layer.
        ``d_model`` (``int``): The dimension of the transformer input and output.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_heads: int,
        # d_model: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
        p: float = 1.0
    ) -> None:
        self.out_channels = hid_channels
        super().__init__()
        self.p = p
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            HGTConv,
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )

        #self.transformer_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        # self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=1)

        self.out_layer = HGTConv(
            hid_channels * num_heads,
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=True,
        )

    
    
    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        
        X = self.drop_layer(X)
        X_transformed = self.multi_head_layer(X=X, hg=hg)
        
        pos_enc  = self.positional_encoding(X_transformed,hg, X_transformed.size(1))  # 计算p-Laplacian作为Positional encoding

        X_transformed += pos_enc 


        X_encoded = self.drop_layer(X_transformed)
        hspd_enc = hspd_encoding(X_encoded, hg)
        X_encoded += hspd_enc
        X_encoded = self.out_layer(X_encoded, hg)

        return X_encoded


    def positional_encoding(self,X, g, channels):
        
            num_nodes = g.num_v
            device = g.device
           
            pos_enc = torch.zeros((num_nodes, channels)).to(device)
            inv_sqrt_d = 1.0 / (channels ** 0.5)

            # 计算位置编码
            positions = torch.arange(start=0, end=num_nodes, dtype=torch.float32).unsqueeze(1)
            div_term1 = torch.exp(torch.arange(start=0, end=channels, step=2).float() * (-math.log(10000.0) / channels))
            div_term2 = torch.exp(torch.arange(start=1, end=channels, step=2).float() * (-math.log(10000.0) / channels))
            
            p_laplace = p_laplacian(X, self.p)  # 计算p-Laplacian矩阵
            # positional_encoding = torch.matmul(pos_enc, torch.from_numpy(p_laplace))
            
            pos_enc[:, 0::2] = torch.sin(positions * div_term1) * inv_sqrt_d
            pos_enc[:, 1::2] = torch.cos(positions * div_term2) * inv_sqrt_d
            
            pos_enc = p_laplace * pos_enc
            
            return pos_enc
        

