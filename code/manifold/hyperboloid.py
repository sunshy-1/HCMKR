"""
This is the implementation of Hyperboloid manifold.
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time
from manifold.base import Manifold
from utility.math_utils import arcosh, cosh, sinh, tanh


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.
    -x0^2 + x1^2 + ... + xd^2 = -K and c = 1 / K is the hyperbolic curvature.
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res


    def minkowski_mutdot(self, x, y, keepdim=True):
        """
        conduct matrix multiplication between two matrices.
        For example: x = [t1, d], y = [t2, d]
        """
        res = torch.mm(x, y.t())
        x_0 = x[..., 0].view(-1, 1)
        y_0 = y[..., 0].view(-1, 1)
        res_0 = torch.mm(x_0, y_0.t())
        res = res - 2 * res_0
        if keepdim:
            res = res.view(res.shape + (1,))
        return res


    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        """Squared distance between pairs of points."""
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    def sqdist_mut(self, x, y, c):
        """
        Squared distance between two matrix with different dimensions
        For example, x = [t1, d], y = [t2, d]
        """
        K = 1. / c
        prod = self.minkowski_mutdot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    def egrad2rgrad(self, x, grad, k, dim=-1):
        grad.narrow(-1, 0, 1).mul_(-1)
        grad = grad.addcmul(self.inner(x, grad, dim=dim, keepdim=True), x / k)
        return grad

    def inner(self, u, v, keepdim: bool = False, dim: int = -1):
        d = u.size(dim) - 1
        uv = u * v
        if keepdim is False:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
                dim, 1, d
            ).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
                dim=dim, keepdim=True
            )

    

    def proj(self, x, c):
        """Projects point p on the manifold."""
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=-1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[..., 0] = 0
        vals = torch.zeros_like(x)
        vals[..., 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        """Projects u on the tangent space of x."""
        d = x.size(-1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=-1, keepdim=True)
        mask = torch.ones_like(u)
        mask[..., 0] = 0
        vals = torch.zeros_like(u)
        vals[..., 0:1] = ux / torch.clamp(x[..., 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[..., 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c):
        """Exponential map of u at point x."""
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        """Logarithmic map of point y at point x."""
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        # x = u.narrow(-1, 1, d).view(-1, d)
        x = u.narrow(-1, 1, d)
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[..., 0:1] = sqrtK * cosh(theta)
        res[..., 1:] = sqrtK * sinh(theta) * x / x_norm

        return self.proj(res, c)

    def logmap0(self, x, c):
        """Logarithmic map of point p at the origin."""
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        # y = x.narrow(-1, 1, d).view(-1, d)
        y = x.narrow(-1, 1, d)
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[..., 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[..., 1:] = sqrtK * arcosh(theta) * y / y_norm
        return self.proj_tan0(res, c)

    def mobius_add(self, x, y, c):
        """Adds points x and y."""
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner_product(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[..., 0:1] = - y_norm
        v[..., 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[..., 1:], dim=-1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    # def ptransp0(self, x, u, c):
    #     K = 1. / c
    #     o = torch.zeros_like(x)
    #     o[..., 0:1] = K ** 0.5
    #     return self.ptransp(o, x, u, c)


class Hyp_Linear(nn.Module):
    """
    Hyperbolic linear layer.
    """
    def __init__(self, in_features, out_features, c, dropout, use_bias):
        super(Hyp_Linear, self).__init__()
        self.manifold = Hyperboloid()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

        """
        The input should add one extra dimension
        """

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.matvec(drop_weight, x, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            res = self.manifold.add(res, hyp_bias, c=self.c)
        return res

    def get_weight(self):
        return self.weight