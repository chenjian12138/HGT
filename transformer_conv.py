import torch
import torch.nn as nn

class TransformerConv(nn.Module):
    """
    Transformer-based Convolutional Layer for Graph Neural Networks (GNNs).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_bn (bool, optional): Whether to use batch normalization. Defaults to False.
        drop_rate (float, optional): Dropout rate. Defaults to 0.5.
        atten_neg_slope (float, optional): Negative slope of the LeakyReLU activation function used in attention mechanism. Defaults to 0.2.
        is_last (bool, optional): Whether this layer is the last layer. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
        is_last: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.drop_rate = drop_rate
        self.atten_neg_slope = atten_neg_slope
        self.is_last = is_last

        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=4,
            dropout=drop_rate
        )

        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(out_channels)

        self.relu = nn.LeakyReLU(negative_slope=atten_neg_slope)
        self.dropout = nn.Dropout(drop_rate)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        # Attention
        x = self.ln1(X)  # Layer Normalization
        x = self.attention(
            x.permute(1, 0, 2),
            x.permute(1, 0, 2),
            x.permute(1, 0, 2),
        )[0]
        x = x.permute(1, 0, 2)

        # Residual Connection
        X = X + x

        # Feedforward
        X = self.fc1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)

        # Residual Connection
        X = X + X

        # Layer Normalization
        X = self.ln2(X)

        if self.use_bn and not self.is_last:
            X = self.bn(X)

        return X
