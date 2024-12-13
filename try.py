import torch
tensor1 = torch.tensor([[1],[2],[3]])     # shape (3, 1)
tensor2 = torch.tensor([[4, 5, 6],    # shape (num_sample, 3)
                        [7, 8, 9],
                        [10, 11, 12],
                        [13, 14, 15]])  # shape (num_sample, 3)
print("tensor1 shape:", tensor1.shape)
print("tensor2 shape:", tensor2.T.shape)
# 交换 tensor1 和 tensor2 的轴顺序，使其形状兼容
result = torch.matmul(tensor1.T, tensor2.T)

# 转置结果回到预期形状
result = result.T


print("Result shape:", result.shape)
print(result)