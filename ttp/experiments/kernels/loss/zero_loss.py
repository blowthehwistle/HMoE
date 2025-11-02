import torch


class ZeroLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pred: torch.Tensor):
        ctx.input_shape = pred.shape
        return torch.zeros([], dtype=pred.dtype, device=pred.device)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return torch.zeros(ctx.input_shape, dtype=grad.dtype, device=grad.device)


def zero_loss(pred: torch.Tensor):
    return ZeroLoss.apply(pred)
