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

import torch
import torch.nn as nn

from crfasrnn.filters import SpatialFilter, BilateralFilter
from crfasrnn.params import DenseCRFParams


class CrfRnn(nn.Module):
    """
    PyTorch implementation of the CRF-RNN module described in the paper:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
    """

    def __init__(self, num_labels, num_iterations=5, crf_init_params=None):
        """
        Create a new instance of the CRF-RNN layer.

        Args:
            num_labels:         Number of semantic labels in the dataset
            num_iterations:     Number of mean-field iterations to perform
            crf_init_params:    CRF initialization parameters
        """
        super(CrfRnn, self).__init__()

        if crf_init_params is None:
            crf_init_params = DenseCRFParams()

        self.params = crf_init_params
        self.num_iterations = num_iterations

        self._softmax = torch.nn.Softmax(dim=0)

        self.num_labels = num_labels

        # --------------------------------------------------------------------------------------------
        # --------------------------------- Trainable Parameters -------------------------------------
        # --------------------------------------------------------------------------------------------

        # Spatial kernel weights
        self.spatial_ker_weights = nn.Parameter(
            crf_init_params.spatial_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Bilateral kernel weights
        self.bilateral_ker_weights = nn.Parameter(
            crf_init_params.bilateral_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Compatibility transform matrix
        self.compatibility_matrix = nn.Parameter(
            torch.eye(num_labels, dtype=torch.float32)
        )

    def forward(self, image, logits):
        """
        Perform CRF inference.

        Args:
            image:  Tensor of shape (3, h, w) containing the RGB image
            logits: Tensor of shape (num_classes, h, w) containing the unary logits
        Returns:
            log-Q distributions (logits) after CRF inference
        """
        if logits.shape[0] != 1:
            raise ValueError("Only batch size 1 is currently supported!")

        image = image[0]
        logits = logits[0]

        spatial_filter = SpatialFilter(image, gamma=self.params.gamma)
        bilateral_filter = BilateralFilter(
            image, alpha=self.params.alpha, beta=self.params.beta
        )
        _, h, w = image.shape
        cur_logits = logits

        for _ in range(self.num_iterations):
            # Normalization
            q_values = self._softmax(cur_logits)

            # Spatial filtering
            spatial_out = torch.mm(
                self.spatial_ker_weights,
                spatial_filter.apply(q_values).view(self.num_labels, -1),
            )

            # Bilateral filtering
            bilateral_out = torch.mm(
                self.bilateral_ker_weights,
                bilateral_filter.apply(q_values).view(self.num_labels, -1),
            )

            # Compatibility transform
            msg_passing_out = (
                spatial_out + bilateral_out
            )  # Shape: (self.num_labels, -1)
            msg_passing_out = torch.mm(self.compatibility_matrix, msg_passing_out).view(
                self.num_labels, h, w
            )

            # Adding unary potentials
            cur_logits = msg_passing_out + logits

        return torch.unsqueeze(cur_logits, 0)
