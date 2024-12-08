import stblis_utils as stb
import torch
import torch.nn as nn
import torch.nn.functional as F


# Module that does the spatiotemporal BLIS calculation
class STBLISModule(nn.Module):
    def __init__(self, adj_s, adj_t, K, J, wavelet_type="W_2"):
        super(STBLISModule, self).__init__()
        if wavelet_type == "W_2":
            self.s_wav = stb.get_W_2(adj_s, K, True)
            self.t_wav = stb.get_W_2(adj_t, J, True)
        else:
            self.s_wav = stb.get_W_1(adj_s, K, True)
            self.t_wav = stb.get_W_1(adj_t, J, True)

    def forward(self, x):
        if len(x.shape) == 3:  # Add dimension if input has 3 dimensions
            x = x.unsqueeze(1)
        new_outputs = []
        B, C, N, T = x.shape
        x = x.reshape(B, C, N *T)
        kron_product = torch.kron(self.s_wav, self.t_wav)  # Kronecker product
        D, _, _ = kron_product.shape
        filtered_data = torch.einsum('ijk, mlk->mlij', kron_product, x)
        filtered_data = filtered_data.reshape(B, C * D, N, T)
        
        new_outputs.append(F.relu(filtered_data))
        new_outputs.append(F.relu(-filtered_data))
        
        output = torch.cat(new_outputs, dim=1)
        return output

# Define sum aggregation of the wavelet scattering outputs
class SumAgg(nn.Module):
    def forward(self, x):
        # Aggregate features across nodes
        aggregated_data = x.sum(dim=1, keepdim=False)
        return aggregated_data

# Define mean aggregation of the wavelet scattering outputs
class MeanAgg(nn.Module):
    def forward(self, x):
        # Aggregate features across nodes
        aggregated_data = x.mean(dim=1, keepdim=False)
        return aggregated_data

# Embedding layer for the MLP
class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EmbeddingLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = F.relu(self.bn(self.fc(x)))
        out = self.dropout(x)
        return out

# Regression layer for the MLP
class RegressionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes):
        super(RegressionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, num_nodes * output_dim)
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.num_nodes, self.output_dim)
        return x

# Define module to precompute the spatiotemporal BLIS module on the data
class STBLISPrecompute(nn.Module):
    def __init__(self, adj_s, adj_t, K, J, wavelet_type="W_2"):
        super(STBLISPrecompute, self).__init__()
        self.blis_module = STBLISModule(adj_s, adj_t, K, J, wavelet_type=wavelet_type)
        self.moment_aggregation = MeanAgg()
        
    def forward(self, x_in):
        x = self.blis_module(x_in)
        x = self.moment_aggregation(x)
        out = x + x_in # Because we are taking the mean, we can add in the input as a residual to retain more input information
        return out

# Defines the output network from the spatiotemporal BLIS module
class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, device):
        super(MLPNet, self).__init__()
        self.embedding_layer = EmbeddingLayer(input_dim, hidden_dim).to(device)
        self.regression_layer = RegressionLayer(hidden_dim, output_dim, num_nodes).to(device)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.regression_layer(x)
        return x