import torch
import numpy as np


def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.from_numpy(np_array).type(pth_dtype)

def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))

def to_np(torch_tensor):
    return torch_tensor.data.numpy()

def to_sqnp(torch_tensor, dtype=np.float64):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)

def create_block_matrix(m, n):
    """
    Creates an m x m block matrix with n blocks of ones along the diagonal.

    :param m: The size of the matrix (number of rows and columns).
    :param n: The number of blocks of ones along the diagonal.
    :return: An m x m numpy array representing the block matrix.

    # Example usage
    m = 6
    n = 2
    block_matrix = create_block_matrix(m, n)
    print(block_matrix)
    """
    if n > m:
        raise ValueError("Number of blocks cannot be greater than the matrix size.")

    # Initialize an m x m matrix of zeros
    matrix = np.zeros((m, m))

    # Calculate the size of each block
    block_size = m // n

    # Fill diagonal blocks with ones
    for i in range(n):
        start_index = i * block_size
        end_index = start_index + block_size
        matrix[start_index:end_index, start_index:end_index] = np.ones((block_size, block_size))

    return matrix
