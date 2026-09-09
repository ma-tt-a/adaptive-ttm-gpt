
# Adaptive TT Layer

All model projections are tensorized via TT decomposition. Adaptiveness comes from diagonal additions between cores.

## NanoGPT (5M)

### Results
![](imgs/pareto_nanogpt.png)

### Wall-clock time

One step:
![](imgs/nanogpt_steptime.png)

<!-- One forward epoch:
![](imgs/nanogpt_forward_epoch.png)

One backward epoch:
![](imgs/nanogpt_backward_epoch.png) -->

### Memory

#### Static memory

![](imgs/nanogpt_static_memory.png)

#### Peak memory

For inference step:
![](imgs/nanogpt_peak_memory_inference.png)

For training step:
![](imgs/nanogpt_peak_memory_training.png)

### Ranks distribution

- c_attn - Q, K, V projections in attention
- attn_proj - output projection in attention
- c_fc - up projection in FFN
- mlp_proj - down projection in FFN

Pruned ranks distribution:

![](imgs/nanogpt_rank_distribution.png)

Unpruned/pruned ranks heatmap:

![](imgs/nanogpt_rank_fracs.png)

### Training dynamics

Left diagrams - quantiles of core entries distribution.

Right diagrams - gradients of cores and mean rank dynamics.
![](imgs/nanogpt_optim_dynamics.png)

## GPT-2 small (125M)

### Results

![](imgs/pareto_gpt2.png)

### Wall-clock time

One step:
![](imgs/gpt2_steptime.png)

<!-- One forward epoch:
![](imgs/gpt2_forward_epoch.png)

One backward epoch:
![](imgs/gpt2_backward_epoch.png) -->

### Memory

#### Peak memory

For inference step:
![](imgs/gpt2_peak_memory_inference.png)

For training step:
![](imgs/gpt2_peak_memory_training.png)