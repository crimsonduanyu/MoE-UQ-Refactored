"""ST-MoE-RMQRN: Spatiotemporal Mixture-of-Experts with Residual Multi-Quantile Regression Network.

This module implements the ST-MoE-RMQRN model for multi-task demand forecasting
with probabilistic uncertainty quantification.
"""

import math
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp


class LaplacianFilter(nn.Module):
    """Learnable Laplacian filter for graph structure learning.
    
    Uses low-rank matrix decomposition to learn adaptive graph structure
    from prior adjacency matrix.
    
    Args:
        space_dim: Number of spatial nodes.
        rank: Rank for matrix decomposition.
        activation: Activation function ('softmax' or 'sigmoid').
        learnable: Whether the filter is learnable.
        prior_graph_path: Path to prior graph adjacency matrix.
    """
    
    def __init__(self, space_dim: int, rank: int, activation: str = 'softmax',
                 learnable: bool = True, prior_graph_path: str = None):
        super().__init__()
        self.space_dim = space_dim
        self.rank = rank
        self.activation = activation
        
        # Initialize from prior graph
        if prior_graph_path is not None:
            graph = np.load(prior_graph_path)
        else:
            # Default initialization
            graph = np.eye(space_dim)
            
        u, sig, v = np.linalg.svd(graph)
        u = u * sig
        
        # Keep components after rank (low-rank approximation)
        u = u[:, rank:]
        v = v[rank:, :]
        
        if learnable:
            self.m1 = nn.Parameter(torch.tensor(u).float())
            self.m2 = nn.Parameter(torch.tensor(v).float())
        else:
            self.register_buffer("m1", torch.tensor(u, dtype=torch.float32))
            self.register_buffer("m2", torch.tensor(v, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Compute the normalized adjacency matrix."""
        mat = torch.matmul(self.m1, self.m2)
        if self.activation == 'softmax':
            return F.softmax(mat, dim=1)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(mat)
        else:
            raise ValueError(f'Activation function {self.activation} not supported.')


class BidirectionalTCN(nn.Module):
    """Bidirectional Temporal Convolutional Network.
    
    Applies temporal convolution in both forward and backward directions
    to capture bidirectional temporal dependencies.
    
    Args:
        in_channels: Number of input channels (nodes).
        out_channels: Number of output features.
        kernel_size: Size of the 1D temporal kernel.
        activation: Activation function ('relu' or 'sigmoid').
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, activation: str = 'relu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Forward direction convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        
        # Backward direction convolutions
        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_timesteps, num_nodes).
            
        Returns:
            Output tensor of shape (batch_size, num_timesteps, out_channels).
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        xf = x.unsqueeze(1)  # (batch_size, 1, num_timesteps, num_nodes)

        # Reverse temporal direction
        inv_idx = torch.arange(xf.size(2) - 1, -1, -1).long().to(device=self.device)
        xb = xf.index_select(2, inv_idx)

        xf = xf.permute(0, 3, 1, 2)
        xb = xb.permute(0, 3, 1, 2)  # (batch_size, num_nodes, 1, num_timesteps)
        
        # Forward direction
        tempf = self.conv1(xf) * torch.sigmoid(self.conv2(xf))
        outf = tempf + self.conv3(xf)
        outf = outf.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels])

        # Backward direction
        tempb = self.conv1b(xb) * torch.sigmoid(self.conv2b(xb))
        outb = tempb + self.conv3b(xb)
        outb = outb.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels])

        # Padding for sequence length alignment
        rec = torch.zeros([batch_size, self.kernel_size - 1, self.out_channels]).to(device=self.device)
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)

        # Reverse backward output
        inv_idx = torch.arange(outb.size(1) - 1, -1, -1).long().to(device=self.device)
        outb = outb.index_select(1, inv_idx)
        
        # Combine directions
        if self.activation == 'relu':
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == 'sigmoid':
            out = torch.sigmoid(outf) + torch.sigmoid(outb)
        else:
            out = outf + outb
            
        return out


class ChebGraphConv(nn.Module):
    """Chebyshev Graph Convolution layer.
    
    Implements spectral graph convolution using Chebyshev polynomials
    for efficient approximation.
    
    Args:
        orders: Order of Chebyshev polynomials.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        activation: Activation function ('relu', 'selu', or 'None').
    """
    
    def __init__(self, orders: int, in_channels: int, out_channels: int, 
                 activation: str = 'None'):
        super().__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = orders + 1
        self.theta = nn.Parameter(
            torch.FloatTensor(in_channels * self.num_matrices, out_channels)
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def _concat(x: torch.Tensor, x_: torch.Tensor) -> torch.Tensor:
        """Concatenate tensors along first dimension."""
        return torch.cat([x, x_.unsqueeze(0)], dim=0)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features of shape (batch_size, num_nodes, in_channels).
            adj: Adjacency matrix of shape (num_nodes, num_nodes).
            
        Returns:
            Output features of shape (batch_size, num_nodes, out_channels).
        """
        batch_size, num_node, input_size = x.shape
        x0 = x.permute(1, 2, 0).reshape(num_node, input_size * batch_size)
        x_poly = x0.unsqueeze(0)
        
        # Compute Chebyshev polynomials
        for _ in range(1, self.orders + 1):
            x1 = torch.mm(adj, x0)
            x_poly = self._concat(x_poly, x1)
            x0 = x1

        x_poly = x_poly.reshape(self.num_matrices, num_node, input_size, batch_size)
        x_poly = x_poly.permute(3, 1, 2, 0)
        x_poly = x_poly.reshape(batch_size, num_node, input_size * self.num_matrices)
        
        out = torch.matmul(x_poly, self.theta) + self.bias
        
        if self.activation == 'relu':
            out = F.relu(out)
        elif self.activation == 'selu':
            out = F.selu(out)
            
        return out


class MultiLayerGCN(nn.Module):
    """Multi-layer Graph Convolutional Network with learnable graph structure.
    
    Args:
        space_dim: Number of spatial nodes.
        hidden_dim: Hidden dimension.
        rank: Rank for Laplacian decomposition.
        num_timesteps_input: Number of input timesteps.
        n_layer: Number of GCN layers.
        prior_graph: Prior graph adjacency matrix.
        learnable_graph: Whether to learn graph structure.
        prior_graph_path: Path to prior graph file.
    """
    
    def __init__(self, space_dim: int = None, hidden_dim: int = 100, rank: int = 30,
                 num_timesteps_input: int = 12, n_layer: int = 3,
                 prior_graph: torch.Tensor = None, learnable_graph: bool = True,
                 prior_graph_path: str = None):
        super().__init__()
        
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(
            ChebGraphConv(3, num_timesteps_input, hidden_dim, activation='relu')
        )
        for _ in range(n_layer - 3):
            self.gcn_layers.append(
                ChebGraphConv(3, hidden_dim, hidden_dim, activation='linear')
            )
        self.gcn_layers.append(ChebGraphConv(2, hidden_dim, rank, activation='linear'))
        self.gcn_layers.append(ChebGraphConv(2, rank, hidden_dim, activation='relu'))

        self.use_prior = prior_graph is not None
        if not self.use_prior:
            self.laplacian_filters = nn.ModuleList()
            laplacian_rank = int(math.ceil(math.sqrt(space_dim)) / 2)
            for _ in range(n_layer):
                self.laplacian_filters.append(
                    LaplacianFilter(space_dim, laplacian_rank, activation='softmax',
                                    learnable=learnable_graph, prior_graph_path=prior_graph_path)
                )
                laplacian_rank *= 2
        else:
            self.prior_graph = prior_graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all GCN layers."""
        for i, gcn in enumerate(self.gcn_layers):
            if self.use_prior:
                x = gcn(x, self.prior_graph)
            else:
                x = gcn(x, self.laplacian_filters[i]())
        return x


class MultiLayerTCN(nn.Module):
    """Multi-layer Temporal Convolutional Network.
    
    Args:
        space_dim: Number of spatial nodes.
        hidden_dim: Hidden dimension.
        rank: Bottleneck rank.
    """
    
    def __init__(self, space_dim: int = None, hidden_dim: int = 24, rank: int = 6):
        super().__init__()
        assert space_dim is not None, 'space_dim must be specified'
        
        self.layer1 = BidirectionalTCN(space_dim, hidden_dim, kernel_size=3, activation='relu')
        self.layer2 = BidirectionalTCN(hidden_dim, rank, kernel_size=3, activation='linear')
        self.layer3 = BidirectionalTCN(rank, hidden_dim, kernel_size=3, activation='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input of shape (batch_size, space_dim, timesteps).
            
        Returns:
            Output of shape (batch_size, timesteps, hidden_dim).
        """
        x = x.permute(0, 2, 1)  # (batch_size, timesteps, space_dim)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SpatioTemporalBackbone(nn.Module):
    """Spatiotemporal feature extraction backbone.
    
    Combines spatial (GCN) and temporal (TCN) feature extraction
    with multiplicative fusion.
    
    Args:
        space_dim: Number of spatial nodes.
        hidden_dim_t: Hidden dimension for temporal module.
        hidden_dim_s: Hidden dimension for spatial module.
        rank_t: Temporal bottleneck rank.
        rank_s: Spatial bottleneck rank.
        t_in: Number of input timesteps.
        t_out: Number of output timesteps.
        prior_graph: Prior graph adjacency matrix.
        learnable_graph: Whether to learn graph structure.
        prior_graph_path: Path to prior graph file.
    """
    
    def __init__(self, space_dim: int, hidden_dim_t: int = 48, hidden_dim_s: int = 128,
                 rank_t: int = 12, rank_s: int = 30, t_in: int = 12, t_out: int = 3,
                 prior_graph: torch.Tensor = None, learnable_graph: bool = True,
                 prior_graph_path: str = None):
        super().__init__()
        self.space_dim = space_dim
        
        self.temporal_encoder = MultiLayerTCN(space_dim, hidden_dim_t, rank_t)
        self.spatial_encoder = MultiLayerGCN(
            space_dim, hidden_dim_s, rank_s, t_in,
            prior_graph=prior_graph, learnable_graph=learnable_graph,
            prior_graph_path=prior_graph_path
        )

        self.temporal_mlp = nn.Sequential(
            Mlp(hidden_dim_t, int(space_dim * 1.5), space_dim, drop=0.05),
        )

        self.spatial_mlp = nn.Sequential(
            Mlp(hidden_dim_s, hidden_dim_s * 2, t_in, drop=0.05),
        )

        self.fusion_mlp = nn.Sequential(
            Mlp(t_in, hidden_dim_t * 2, hidden_dim_t, drop=0.1),
            Mlp(hidden_dim_t, hidden_dim_t, hidden_dim_t, drop=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spatiotemporal fusion.
        
        Args:
            x: Input of shape (batch_size, space_dim, timesteps).
            
        Returns:
            Fused features of shape (batch_size, space_dim, hidden_dim).
        """
        # Spatial path
        spatial_feat = self.spatial_encoder(x)
        spatial_feat = self.spatial_mlp(spatial_feat)
        
        # Temporal path
        temporal_feat = self.temporal_encoder(x)
        temporal_feat = self.temporal_mlp(temporal_feat)
        
        # Multiplicative fusion
        fused = spatial_feat * temporal_feat.permute(0, 2, 1)
        
        return self.fusion_mlp(fused)


class ResidualMultiQuantileHead(nn.Module):
    """Residual Multi-Quantile Regression head for uncertainty quantification.
    
    Produces quantile predictions with monotonicity constraints by
    Residually building upper and lower quantiles from the median.
    
    Args:
        quantile_list: List of quantile levels (must be odd length with 0.5 as median).
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (prediction horizon).
    """
    
    def __init__(self, quantile_list: list, input_dim: int, 
                 hidden_dim: int, output_dim: int):
        super().__init__()
        self.quantile_list = sorted(quantile_list)
        self.num_quantiles = len(quantile_list)
        assert self.num_quantiles % 2 == 1, 'Quantile number must be odd.'
        
        half_len = self.num_quantiles // 2
        self.median_head = Mlp(input_dim, hidden_dim, output_dim)
        self.lower_heads = nn.ModuleList([
            Mlp(input_dim, hidden_dim, output_dim, drop=0.05) 
            for _ in range(half_len)
        ])
        self.upper_heads = nn.ModuleList([
            Mlp(input_dim, hidden_dim, output_dim, drop=0.05) 
            for _ in range(half_len)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to produce quantile predictions.
        
        Args:
            x: Input features.
            
        Returns:
            Quantile predictions of shape (..., num_quantiles, output_dim).
        """
        median = self.median_head(x)
        lower_results, upper_results = [], []
        
        lower_prev = median
        upper_prev = median
        
        for i in range(self.num_quantiles // 2):
            lower_delta = self.lower_heads[i](x)
            upper_delta = self.upper_heads[i](x)
            
            # Ensure monotonicity via ReLU on deltas
            lower_prev = lower_prev - F.relu(lower_delta)
            upper_prev = upper_prev + F.relu(upper_delta)
            
            lower_results.append(lower_prev)
            upper_results.append(upper_prev)
            
        lower_results.reverse()
        return torch.stack(lower_results + [median] + upper_results, dim=1)


class ExpertGate(nn.Module):
    """Gating network for expert selection.
    
    Args:
        num_task: Number of tasks.
        num_nodes: Number of spatial nodes.
        num_timesteps: Number of timesteps.
        num_experts: Number of shared experts.
    """
    
    def __init__(self, num_task: int, num_nodes: int, 
                 num_timesteps: int, num_experts: int):
        super().__init__()
        self.reduce_t = nn.Linear(num_timesteps, 1)
        self.reduce_n = nn.Linear(num_nodes, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expert weights.
        
        Args:
            x: Input features.
            
        Returns:
            Expert weights of shape (batch, 1, 1, num_experts).
        """
        x = F.gelu(self.reduce_t(x).squeeze(-1))
        x = F.relu(self.reduce_n(x).squeeze(-1))
        x = F.softmax(x, dim=-1)
        x = rearrange(x, 'b e -> b 1 1 e')
        return x


class EnvironmentAwareRouter(nn.Module):
    """Environment-aware router for dynamic expert selection.
    
    Routes samples to experts based on contextual features
    (time embeddings + environmental conditions).
    
    Args:
        contextual_dim: Dimension of contextual features.
        t_in: Number of input timesteps.
        num_experts: Number of experts.
        top_k: Number of experts to activate.
        hidden_dim: Hidden layer dimension.
        tau: Gumbel-Softmax temperature.
    """
    
    def __init__(self, contextual_dim: int = 13, t_in: int = 24,
                 num_experts: int = 8, top_k: int = 3,
                 hidden_dim: int = 64, tau: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        # Ensure top_k doesn't exceed num_experts
        self.top_k = min(top_k, num_experts)
        self.tau = tau

        self.context_mlp = Mlp(contextual_dim, hidden_dim, num_experts, drop=0.2)
        self.temporal_mlp = Mlp(t_in, hidden_dim, 1, drop=0.2)

    def forward(self, contextual: torch.Tensor) -> tuple:
        """Compute expert routing.
        
        Args:
            contextual: Contextual features of shape (batch, context_dim, timesteps).
            
        Returns:
            Tuple of (mask_ste, probs) where mask_ste is the top-k mask
            with straight-through gradient and probs are soft probabilities.
        """
        contextual = self.temporal_mlp(contextual).squeeze(-1)
        logits = self.context_mlp(contextual)

        # Soft Gumbel-Softmax probabilities
        probs = F.gumbel_softmax(logits, tau=self.tau, hard=False)

        # Top-k mask
        topk_idx = torch.topk(probs, self.top_k, dim=-1).indices
        mask = torch.zeros_like(probs).scatter_(1, topk_idx, 1.0)

        # Straight-Through Estimator
        mask_ste = mask + (probs - probs.detach())

        return mask_ste, probs


class STMoERMQRN(nn.Module):
    """Spatiotemporal Mixture-of-Experts with Residual Multi-Quantile Regression Network.
    
    Main model combining:
    - Task-specific spatiotemporal experts
    - Shared spatiotemporal experts with environment-aware routing
    - Residual multi-quantile heads for uncertainty quantification
    
    Args:
        space_dim: Number of spatial nodes.
        contextual_dim: Dimension of contextual features.
        hidden_dim_t: Hidden dimension for temporal processing.
        hidden_dim_s: Hidden dimension for spatial processing.
        rank_t: Temporal bottleneck rank.
        rank_s: Spatial bottleneck rank.
        t_in: Number of input timesteps.
        t_out: Number of output timesteps.
        task_num: Number of tasks (data sources).
        quantile_list: List of quantile levels for prediction.
        prior_graph: Prior graph adjacency matrix.
        num_experts: Number of shared experts.
        prior_graph_path: Path to prior graph file.
    """
    
    def __init__(self, space_dim: int, contextual_dim: int,
                 hidden_dim_t: int = 64, hidden_dim_s: int = 128,
                 rank_t: int = 32, rank_s: int = 60,
                 t_in: int = 12, t_out: int = 3, task_num: int = 3,
                 quantile_list: list = None, prior_graph: torch.Tensor = None,
                 num_experts: int = 6, prior_graph_path: str = None):
        super().__init__()
        
        self.t_in = t_in
        self.t_out = t_out
        self.space_dim = space_dim
        self.contextual_dim = contextual_dim
        self.task_num = task_num
        self.hidden_dim_t = hidden_dim_t

        # Task-specific experts
        self.task_experts = nn.ModuleList([
            SpatioTemporalBackbone(
                space_dim, hidden_dim_t, hidden_dim_s, rank_t, rank_s,
                t_in, t_out, prior_graph, prior_graph_path=prior_graph_path
            ) for _ in range(task_num)
        ])
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            SpatioTemporalBackbone(
                space_dim, hidden_dim_t, hidden_dim_s, rank_t, rank_s,
                t_in, t_out, prior_graph, prior_graph_path=prior_graph_path
            ) for _ in range(num_experts)
        ])
        
        # Environment-aware router
        self.router = EnvironmentAwareRouter(
            contextual_dim=contextual_dim,
            num_experts=num_experts,
            top_k=3,
            hidden_dim=64
        )

        # Expert gates for each task
        self.gates = nn.ModuleList([
            ExpertGate(task_num, space_dim, hidden_dim_t, num_experts)
            for _ in range(task_num)
        ])
        
        # Quantile prediction towers
        self.quantile_towers = nn.ModuleList([
            ResidualMultiQuantileHead(quantile_list, hidden_dim_t, hidden_dim_t * 2, t_out)
            for _ in range(task_num)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, task_num, space_dim + contextual_dim, t_in).
            
        Returns:
            Quantile predictions of shape (batch, task_num, num_quantiles, space_dim, t_out).
        """
        # Split input into features and contextual information
        features, contextual = torch.split(
            x.float(), [self.space_dim, self.contextual_dim], dim=2
        )
        b, m, n, t_in = features.shape

        # Compute expert routing
        contextual_flat = rearrange(contextual, 'b m c t -> (b m) c t')
        mask, probs = self.router(contextual_flat)
        
        num_experts = mask.shape[-1]
        mask = rearrange(mask, '(b m) e -> b m e', b=b, m=m)
        probs = rearrange(probs, '(b m) e -> b m e', b=b, m=m)
        
        # Flatten for expert processing
        x_flat = rearrange(features, 'b m n t -> (b m) n t')
        
        # Process through shared experts
        expert_outputs = []
        for expert_idx in range(num_experts):
            sample_mask = mask[:, :, expert_idx].view(-1)
            selected_idx = torch.nonzero(sample_mask, as_tuple=True)[0]

            if len(selected_idx) == 0:
                expert_outputs.append(
                    x_flat.new_zeros((b, m, n, self.hidden_dim_t))
                )
                continue

            input_to_expert = x_flat[selected_idx]
            output_from_expert = self.shared_experts[expert_idx](input_to_expert)

            expert_output_flat = x_flat.new_zeros((b * m, n, self.hidden_dim_t))
            expert_output_flat[selected_idx] = output_from_expert
            expert_outputs.append(
                rearrange(expert_output_flat, '(b m) n t -> b m n t', b=b, m=m)
            )

        shared_expert_output = torch.stack(expert_outputs, dim=-1)
        
        # Process through task-specific experts
        task_features = [
            self.task_experts[i](features[:, i, ...]) 
            for i in range(self.task_num)
        ]

        # Combine experts and produce quantile predictions
        results = []
        for i in range(self.task_num):
            gate_weights = self.gates[i](task_features[i])
            weighted_shared = torch.sum(
                gate_weights * shared_expert_output[:, i, ...], dim=-1
            )
            combined = weighted_shared + task_features[i]
            results.append(self.quantile_towers[i](combined))

        return torch.stack(results, dim=1)
