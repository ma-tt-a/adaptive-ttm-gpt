import torch as t
import torch.nn as nn
import torch.nn.functional as F

from tensorized_layers import TTLinearConfig, TTLinear
from utils import tt_2_matrix

t.manual_seed(42)
batch = int(input("batch:"))
X = t.randn(batch, 256, requires_grad=True).xpu()


rank = t.Size([1, 4, 4, 8, 4, 4, 1])
in_shape = t.Size([4, 8, 8])
out_shape = t.Size([8, 8, 16])
cfg = TTLinearConfig(
    in_shape=in_shape,
    out_shape=out_shape,
    rank=rank,
    adaptive=False
)
shape = cfg.in_shape + cfg.out_shape
layer = TTLinear(cfg).xpu()

t.xpu.reset_peak_memory_stats()
print('Running tensorized...')
start = t.xpu.Event(enable_timing=True)
end = t.xpu.Event(enable_timing=True)
start.record()
Y = layer(X)
l = t.linalg.norm(Y)
l.backward()
end.record()
t.xpu.synchronize()
ten_time = start.elapsed_time(end) / 1000
ten_mem = t.xpu.max_memory_allocated()
print("Tensorized memory:, ", ten_mem)
print("Tensorized time: ", ten_time)

t.xpu.reset_peak_memory_stats()

print('Running dense...')
start = t.xpu.Event(enable_timing=True)
end = t.xpu.Event(enable_timing=True)
start.record()
mat = tt_2_matrix(layer.get_cores(masked=False))
Y_mat = X @ mat
l = t.linalg.norm(Y_mat)
l.backward()
end.record()
t.xpu.synchronize()
dense_time = start.elapsed_time(end) / 1000
dense_mem = t.xpu.max_memory_allocated()
print("Dense memory:, ", dense_mem)
print("Dense time: ", ten_time)

print(f"CR: {dense_mem / ten_mem}")
print(f"Speedup: {dense_time / ten_time}")

print(t.linalg.norm(Y - Y_mat) / t.linalg.norm(Y_mat))


# t.autograd.gradcheck(TTMatVec.apply, [X, *cores])
