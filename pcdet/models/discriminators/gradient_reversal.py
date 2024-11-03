import torch
import torch.nn as nn

from torch.autograd import Function

"""
Gradient Reversal Layer
https://github.com/tadeephuy/GradientReversal
https://github.com/jvanvugt/pytorch-domain-adaptation
"""

class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = torch.tensor(0.0, requires_grad = False)

    def update_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_, requires_grad = False)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

"""
Gradient Reversal Function
https://github.com/tadeephuy/GradientReversal
https://github.com/jvanvugt/pytorch-domain-adaptation
"""

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(x, lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        _, lambda_ = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = - lambda_ * grad_output

        return grad_input, None

