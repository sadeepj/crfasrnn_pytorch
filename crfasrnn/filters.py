"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

try:
    import permuto_cpp
except ImportError as e:
    raise (e, "Did you import `torch` first?")

_CPU = torch.device("cpu")
_EPS = np.finfo("float").eps


class PermutoFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_in, features):
        q_out = permuto_cpp.forward(q_in, features)[0]
        ctx.save_for_backward(features)
        return q_out

    @staticmethod
    def backward(ctx, grad_q_out):
        feature_saved = ctx.saved_tensors[0]
        grad_q_back = permuto_cpp.backward(
            grad_q_out.contiguous(), feature_saved.contiguous()
        )[0]
        return grad_q_back, None  # No need of grads w.r.t. features


def _spatial_features(image, sigma):
    """
    Return the spatial features as a Tensor

    Args:
        image:  Image as a Tensor of shape (channels, height, wight)
        sigma:  Bandwidth parameter

    Returns:
        Tensor of shape [h, w, 2] with spatial features
    """
    sigma = float(sigma)
    _, h, w = image.size()
    x = torch.arange(start=0, end=w, dtype=torch.float32, device=_CPU)
    xx = x.repeat([h, 1]) / sigma

    y = torch.arange(
        start=0, end=h, dtype=torch.float32, device=torch.device("cpu")
    ).view(-1, 1)
    yy = y.repeat([1, w]) / sigma

    return torch.stack([xx, yy], dim=2)


class AbstractFilter(ABC):
    """
    Super-class for permutohedral-based Gaussian filters
    """

    def __init__(self, image):
        self.features = self._calc_features(image)
        self.norm = self._calc_norm(image)

    def apply(self, input_):
        output = PermutoFunction.apply(input_, self.features)
        return output * self.norm

    @abstractmethod
    def _calc_features(self, image):
        pass

    def _calc_norm(self, image):
        _, h, w = image.size()
        all_ones = torch.ones((1, h, w), dtype=torch.float32, device=_CPU)
        norm = PermutoFunction.apply(all_ones, self.features)
        return 1.0 / (norm + _EPS)


class SpatialFilter(AbstractFilter):
    """
    Gaussian filter in the spatial ([x, y]) domain
    """

    def __init__(self, image, gamma):
        """
        Create new instance

        Args:
            image:  Image tensor of shape (3, height, width)
            gamma:  Standard deviation
        """
        self.gamma = gamma
        super(SpatialFilter, self).__init__(image)

    def _calc_features(self, image):
        return _spatial_features(image, self.gamma)


class BilateralFilter(AbstractFilter):
    """
    Gaussian filter in the bilateral ([r, g, b, x, y]) domain
    """

    def __init__(self, image, alpha, beta):
        """
        Create new instance

        Args:
            image:  Image tensor of shape (3, height, width)
            alpha:  Smoothness (spatial) sigma
            beta:   Appearance (color) sigma
        """
        self.alpha = alpha
        self.beta = beta
        super(BilateralFilter, self).__init__(image)

    def _calc_features(self, image):
        xy = _spatial_features(
            image, self.alpha
        )  # TODO Possible optimisation, was calculated in the spatial kernel
        rgb = (image / float(self.beta)).permute(1, 2, 0)  # Channel last order
        return torch.cat([xy, rgb], dim=2)
