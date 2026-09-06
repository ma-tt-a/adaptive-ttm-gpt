import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def tt_2_tensor(cores):
    """
    Contract a chain of 3-rd order cores
    """
    # assert cores[0].shape[0] == 1 and cores[-1].shape[-1] == 1, \
    #     "it is not a tt"
    N = len(cores)
    offset = N + 1
    expr = []
    res = [0]
    for k, G in enumerate(cores):
        expr += [G, (k, k + offset, k + 1)]
        res += [k + offset]
    res += [N]
    return t.einsum(*expr, res)


def tt_2_matrix(cores: List[t.Tensor]):
    """
    Contract a chain of 3-rd order cores, then reshape to a matrix
    """
    assert len(cores) % 2 == 0, "# cores is odd"
    assert cores[0].shape[0] == 1 and cores[-1].shape[-1] == 1, \
        "it is not a tt"
    d = len(cores) // 2
    X = tt_2_tensor(cores)
    sh = X.shape
    return X.reshape(sh[:d+1].numel(), sh[d+1:].numel())


def build_cores_gauss(shape: t.Size, rank: t.Size, std: float = 1.0):
    """
    Initialize TT cores with given shape and ranks from a centered normal
    """
    N = len(shape)
    cores = []
    for n in range(1, N + 1):
        cores.append(std * t.randn(rank[n-1], shape[n-1], rank[n]))
    return cores


def get_xavier_std(rank: t.Size, target_std: float):
    """
    Core std such that the contracted matrix has entries with std target_std.

    Var(W) = prod(rank) * std ** (2N).
    """
    N = len(rank) - 1
    R = 1
    for r in rank[1:-1]:
        R *= r
    return (target_std ** 2 / R) ** (1 / (2 * N))


def get_uniform_rank(in_shape: t.Size, out_shape: t.Size, max_rank: int) -> t.Size:
    """
    Uniform TT-rank.
    Clipped with max_rank and R_i <= I_1 x ... x I_i, R_i <= I_i+1 x ... x I_2d
    """
    dims = in_shape + out_shape
    N = len(dims)
    rank = [1]
    for n in range(1, N):
        rank.append(min(max_rank, dims[:n].numel(), dims[n:].numel()))
    rank.append(1)
    return t.Size(rank)


def get_device():
    """
    Pick the available accelerator: cuda (colab), xpu (local intel), else cpu
    """
    if t.cuda.is_available():
        return t.device("cuda")
    if hasattr(t, "xpu") and t.xpu.is_available():
        return t.device("xpu")
    return t.device("cpu")


def device_module(device: t.device):
    """
    torch.cuda / torch.xpu for the given device, or None on cpu
    """
    return getattr(t, device.type, None) if device.type != "cpu" else None


def tt_svd(X: t.Tensor):

    def k_unfolding(X: t.Tensor, k=2):
        sh = X.shape
        return X.reshape(sh[:k].numel(), sh[k:].numel())

    def flip_k_leg(M: t.Tensor, shapes, k):
        return M.reshape(M.shape[0] * shapes[k-1], shapes[k:].numel())

    N = X.ndim
    shapes = X.shape
    cores = []
    U, S, VT = t.linalg.svd(k_unfolding(X, k=1), full_matrices=False)
    cores.append(U.unsqueeze(0))
    for k in range(2, N):
        r = VT.shape[0]
        U, S, VT = t.linalg.svd(flip_k_leg(
            S[:, None] * VT, shapes, k), full_matrices=False)
        cores.append(U.reshape(r, U.shape[0] // r, U.shape[1]))
    cores.append((S[:, None] * VT).unsqueeze(-1))
    return cores


class TTMatVec(t.autograd.Function):

    @staticmethod
    def A_i(i: int, *cores: List[t.Tensor]):
        return tt_2_tensor(cores[:i])

    @staticmethod
    def B_i(i: int, *cores: List[t.Tensor]):
        d = len(cores) // 2
        return tt_2_tensor(cores[d:d+i])

    @staticmethod
    def A_inv_i(i: int, *cores: List[t.Tensor]):
        d = len(cores) // 2
        return tt_2_tensor(cores[d-i:d])

    @staticmethod
    def B_inv_i(i: int, *cores: List[t.Tensor]):
        d = len(cores) // 2
        return tt_2_tensor(cores[2*d-i:])

    @staticmethod
    def forward(ctx, X: t.Tensor, *cores: List[t.Tensor]):
        d = len(cores) // 2
        A_d = TTMatVec.A_i(d, *cores)
        B_d = TTMatVec.B_i(d, *cores)
        a_sh = A_d.shape
        b_sh = B_d.shape
        T_1 = X @ A_d.reshape((a_sh[:-1].numel(), a_sh[-1]))
        Y = T_1 @ B_d.reshape((b_sh[0], b_sh[1:].numel()))
        ctx.save_for_backward(X, T_1, *cores)
        return Y

    @staticmethod
    def backward(ctx, *g_Y):
        X, T_1, *cores = ctx.saved_tensors
        d = len(cores) // 2
        r_d = t.Size([cores[d-1].shape[2]])
        in_shape, out_shape = t.Size([]), t.Size([])
        for n in range(d):
            in_shape += t.Size([cores[n].shape[1]])
        for n in range(d, 2*d):
            out_shape += t.Size([cores[n].shape[1]])

        # =============

        # g_X
        A_d = TTMatVec.A_i(d, *cores)
        B_d = TTMatVec.B_i(d, *cores)
        a_sh = A_d.shape
        b_sh = B_d.shape

        U_1 = g_Y[0] @ B_d.reshape((b_sh[0], b_sh[1:].numel())).T
        g_X = U_1 @ A_d.reshape((a_sh[:-1].numel(), a_sh[-1])).T

        # =============

        # g_G_i: i <= d
        U_2 = (U_1.T @ X).reshape(r_d + in_shape)
        g_G_left = []

        # i = 1
        i = 1
        expr_U_2 = [0] + [k for k in range(1, d + 1)]
        expr_A_inv_i = [2 * d + 4] + [k for k in range(i + 1, d + 1)] + [0]
        expr_g_G = [i, 2 * d + 4]

        g_G_1 = t.einsum(
            U_2, expr_U_2,
            TTMatVec.A_inv_i(d-i, *cores), expr_A_inv_i,
            expr_g_G
        )[None, :, :]
        g_G_left.append(g_G_1)

        # 1 < i < d
        for i in range(2, d):
            expr_U_2 = [0] + [k for k in range(1, d + 1)]
            expr_A_i = [2 * d + 2] + [k for k in range(1, i)] + [2 * d + 3]
            expr_A_inv_i = [2 * d + 4] + [k for k in range(i + 1, d + 1)] + [0]
            expr_g_G = [2 * d + 3, i, 2 * d + 4]

            g_G_i = t.einsum(
                U_2, expr_U_2,
                TTMatVec.A_i(i-1, *cores), expr_A_i,
                TTMatVec.A_inv_i(d-i, *cores), expr_A_inv_i,
                expr_g_G
            )
            g_G_left.append(g_G_i)

        # i = d
        i = d
        expr_U_2 = [0] + [k for k in range(1, d + 1)]
        expr_A_i = [2 * d + 2] + [k for k in range(1, i)] + [2 * d + 3]
        expr_g_G = [2 * d + 3, i, 0]

        g_G_d = t.einsum(
            U_2, expr_U_2,
            TTMatVec.A_i(i-1, *cores), expr_A_i,
            expr_g_G
        )
        g_G_left.append(g_G_d)

        # =============

        # g_Gi: i >= d + 1
        T_2 = (g_Y[0].T @ T_1).reshape(out_shape + r_d)
        g_G_right = []

        # i = d + 1
        i = d + 1
        expr_T_2 = [k for k in range(d + 1, 2 * d + 1)] + [d]
        expr_B_inv_i = [2 * d + 3] + [k for k in range(i + 1, 2 * d + 2)]
        expr_g_G = [d, i, 2 * d + 3]

        g_G_d1 = t.einsum(
            T_2, expr_T_2,
            TTMatVec.B_inv_i(2*d-i, *cores), expr_B_inv_i,
            expr_g_G
        )
        g_G_right.append(g_G_d1)

        # d + 1 < i < 2d
        for i in range(d+2, 2*d):
            expr_T_2 = [k for k in range(d + 1, 2 * d + 1)] + [d]
            expr_B_i = [d] + [k for k in range(d + 1, i)] + [2 * d + 2]
            expr_B_inv_i = [2 * d + 3] + [k for k in range(i + 1, 2 * d + 2)]
            expr_g_G = [2 * d + 2, i, 2 * d + 3]

            g_G_di = t.einsum(
                T_2, expr_T_2,
                TTMatVec.B_i(i-1-d, *cores), expr_B_i,
                TTMatVec.B_inv_i(2*d-i, *cores), expr_B_inv_i,
                expr_g_G
            )
            g_G_right.append(g_G_di)

        # i = 2d
        i = 2 * d
        expr_T_2 = [k for k in range(d + 1, 2 * d + 1)] + [d]
        expr_B_i = [d] + [k for k in range(d + 1, i)] + [2 * d + 2]
        expr_g_G = [2 * d + 2, i]

        g_G_2d = t.einsum(
            T_2, expr_T_2,
            TTMatVec.B_i(i-1-d, *cores), expr_B_i,
            expr_g_G
        )[:, :, None]
        g_G_right.append(g_G_2d)

        # =============

        # g = [g_X] + g_G_left + g_G_right
        return g_X, *g_G_left, *g_G_right
