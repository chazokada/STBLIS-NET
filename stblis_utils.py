import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Function to remove 'module.' prefix when loading in DataParallel model
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict

# From https://github.com/KrishnaswamyLab/blis/blob/main/blis/models/wavelets.py
def get_P(A: torch.Tensor) -> torch.Tensor:
    d_arr = torch.sum(A, dim=0)  # Sum along the rows (axis 0)
    d_arr_inv = 1.0 / d_arr
    d_arr_inv[torch.isinf(d_arr_inv)] = 0  # Replace infinities with zeros
    D_inv = torch.diag(d_arr_inv)
    P = 0.5 * (torch.eye(D_inv.shape[0], dtype=torch.float64) + torch.matmul(A, D_inv))  # Ensure eye matrix is float64
    return P

# From https://github.com/KrishnaswamyLab/blis/blob/main/blis/models/wavelets.py
def get_T(A: torch.Tensor) -> torch.Tensor:
    d_arr = torch.sum(A, dim=1)  # Sum along the columns (axis 1)
    d_arr_inv = 1.0 / d_arr
    d_arr_inv[torch.isinf(d_arr_inv)] = 0  # Replace infinities with zeros
    D_inv_sqrt = torch.diag(torch.sqrt(d_arr_inv))
    T = 0.5 * (torch.eye(A.shape[0], dtype=torch.float64) + torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt))  # Ensure eye matrix is float64
    return T

# From https://github.com/KrishnaswamyLab/blis/blob/main/blis/models/wavelets.py
def get_M(A: torch.Tensor) -> torch.Tensor:
    M = torch.diag(1.0 / torch.sqrt(torch.sum(A, dim=1)))
    return M

# From https://github.com/KrishnaswamyLab/blis/blob/main/blis/models/wavelets.py
def get_W_2(A: torch.Tensor, largest_scale: int, low_pass_as_wavelet=False) -> torch.Tensor:
    P = get_P(A)
    N = P.shape[0]
    powered_P = P
    if low_pass_as_wavelet:
        wavelets = torch.zeros((largest_scale + 2, *P.shape), dtype=torch.float32)
    else:
        wavelets = torch.zeros((largest_scale + 1, *P.shape), dtype=torch.float32)
    wavelets[0, :, :] = torch.eye(N, dtype=torch.float32) - powered_P
    for scale in range(1, largest_scale + 1):
        Phi = torch.matmul(powered_P, (torch.eye(N, dtype=torch.float32) - powered_P))
        wavelets[scale, :, :] = Phi
        powered_P = torch.matmul(powered_P, powered_P)
    low_pass = powered_P
    if low_pass_as_wavelet:
        wavelets[-1, :, :] = low_pass
    return wavelets

# From https://github.com/KrishnaswamyLab/blis/blob/main/blis/models/wavelets.py
def get_W_1(A: torch.Tensor, largest_scale: int, low_pass_as_wavelet=False) -> torch.Tensor:
    T_matrix = get_T(A)
    w, U = torch.linalg.eigh(T_matrix)
    w = torch.maximum(w, torch.tensor(0.0, dtype=torch.float32))  # ReLU operation

    d_arr = torch.sum(A, dim=1)
    d_arr_inv = 1.0 / d_arr
    d_arr_inv[torch.isinf(d_arr_inv)] = 0
    M = torch.diag(torch.sqrt(d_arr_inv))
    M_inv = torch.diag(torch.sqrt(d_arr))
    if low_pass_as_wavelet:
        wavelets = torch.zeros((largest_scale + 2, *T_matrix.shape), dtype=torch.float32)
    else:
        wavelets = torch.zeros((largest_scale + 1, *T_matrix.shape), dtype=torch.float32)
    eig_filter = torch.sqrt(torch.maximum(1.0 - w, torch.tensor(0.0, dtype=torch.float32)))
    Psi = torch.matmul(M_inv, torch.matmul(U, torch.matmul(torch.diag(eig_filter), U.T))) @ M
    wavelets[0, :, :] = Psi
    for scale in range(1, largest_scale + 1):
        eig_filter = torch.sqrt(torch.maximum(w**(2**(scale - 1)) - w**(2**scale), torch.tensor(0.0, dtype=torch.float64)))
        Psi = torch.matmul(M_inv, torch.matmul(U, torch.matmul(torch.diag(eig_filter), U.T))) @ M
        wavelets[scale, :, :] = Psi
    low_pass = torch.matmul(M_inv, torch.matmul(U, torch.matmul(torch.diag(torch.sqrt(w**(2**largest_scale))), U.T))) @ M
    if low_pass_as_wavelet:
        wavelets[-1, :, :] = low_pass
    return wavelets

# Create an adjacency matrix for a linear time graph
def get_time_adj(time_steps: int) -> torch.Tensor:
    
    time_adj = torch.zeros((time_steps, time_steps), dtype=torch.float32)
    
    # Set the edges for a line graph (each node is connected to the next)
    for i in range(time_steps - 1):  # time_steps - 1 because the last node has no next node
        time_adj[i, i + 1] = 1.0
        time_adj[i + 1, i] = 1.0  # Since it's undirected, set both directions
    return time_adj

# Function to pad the input tensor with 0s so that all batches have the same size during training/testing
def pad_tensor(tensor, pad_to):
    # tensor: original tensor of shape (a, b, c)
    # pad_to: target size for the first dimension
    a, b, c = tensor.shape
    
    # If a1 is greater than a, pad with zeros along the first dimension
    if pad_to > a:
        padding = (0, 0, 0, 0, 0, pad_to - a)  # Padding format is (last_dim, second_last_dim, ..., first_dim)
        padded_tensor = F.pad(tensor, padding, "constant", 0)  # Pad with zeroes

    else:
        padded_tensor = tensor  # No padding needed, return as is

    return padded_tensor

# Function to plot outputs from the model
def plot_time_series(sample_index, node_index, input_series, actual_series, predicted_series, num_timesteps, num_nodes, output_dim, mean_ind_dict, std_ind_dict, zscore=False):
    plt.figure(figsize=(10, 6))

    # Plot the input time series (last num_timesteps timesteps before prediction)
    input_series = input_series[sample_index].view(-1, num_nodes, num_timesteps).numpy().squeeze(0)
    if not zscore:
        input_series = zscore_to_original(node_index, input_series, mean_ind_dict, std_ind_dict)
    
    # Plot the actual values (next output_dim timesteps after input)
    actual_series = actual_series[sample_index, node_index, :].numpy()
    if not zscore:
        actual_series = zscore_to_original(node_index, actual_series, mean_ind_dict, std_ind_dict)
    
    # Plot the predicted values (next output_dim timesteps after input)
    predicted_series = predicted_series[sample_index, node_index, :].numpy()
    if not zscore:
        predicted_series = zscore_to_original(node_index, predicted_series, mean_ind_dict, std_ind_dict)

    # Concatenate input_series with actual_series and predicted_series
    actual_with_input = np.concatenate((input_series[node_index, :], actual_series))
    predicted_with_input = np.concatenate((input_series[node_index, :], predicted_series))

    # Plot the input series, actual series, and predicted series with dashed lines
    plt.plot(range(0, 5 *  (num_timesteps + output_dim), 5), predicted_with_input, 'k:')
    plt.plot(range(0, 5 * (num_timesteps + output_dim), 5), actual_with_input, 'k--')
    
    # Scatter the input, actual, and predicted series
    plt.scatter(range(0, 5 * num_timesteps, 5), input_series[node_index, :], c='black', marker="o", label='Input Series', zorder=5)
    plt.scatter(range(5 * num_timesteps, 5 *(num_timesteps + output_dim), 5), actual_series, c='black', marker="^", label='Actual Series', zorder=5)
    plt.scatter(range(5 * num_timesteps, 5 * (num_timesteps + output_dim), 5), predicted_series, c='black', marker="x", label='Predicted Series', zorder=10)

    # Title and labels
    plt.title(f'Time Series for Sample {sample_index}, Node {node_index}')
    plt.xlabel('Time (mins)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Transforms data that is z-score normalized back to original values
def zscore_to_original(sensor_ind, z_score, mean_dict, std_dict):
    # Get the mean and standard deviation for the given sensor index
    mean = mean_dict.get(sensor_ind)
    std = std_dict.get(sensor_ind)
    
    # Reverse the z-score formula
    original_value = z_score * std + mean
    return original_value

def calculate_mape(preds, actuals):
    mask = (actuals != 0) # To avoid division by zero
    valid_actuals = actuals[mask]
    valid_preds = preds[mask]
    
    mape = (torch.abs((valid_actuals - valid_preds) / (valid_actuals))).mean()
    return mape.item()

def calculate_rmse(preds, actuals):
    # Calculate the squared differences
    squared_diff = (preds - actuals) ** 2

    # Calculate the mean of squared differences
    mse = squared_diff.mean()
    rmse = math.sqrt(mse)

    return rmse

def calculate_mae(preds, actuals):
    # Calculate the absolute differences
    abs_diff = torch.abs(preds - actuals)

    # Calculate the mean of absolute differences
    mae = abs_diff.mean()

    return mae.item()

# Function to calculate all relevant metrics (MAE, RMSE, and MAPE, given specific indices
def calculate_metrics_per_index(preds, target, indices=[2, 5, 11]):
    # Index 2 is 15 mins, index  5 is 30 minutes, index 11 is 60 minutes
    # This is because data is at 5 min intervals
    metrics = {}

    # Iterate over the specified indices
    for idx in indices:
        # Select values at the specified index (along dim 2 (size 12))
        target_selected = target[:, :, idx]
        preds_selected = preds[:, :, idx]
        
        # Compute MAE
        mae = calculate_mae(preds_selected, target_selected)
        
        # Compute RMSE
        rmse = calculate_rmse(preds_selected, target_selected)
        
        # Compute MAPE (Mean Absolute Percentage Error)
        # Exclude values where actual values are 0
        mape = calculate_mape(preds_selected, target_selected)

        # Store the metrics for the current index
        metrics[idx] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    return metrics