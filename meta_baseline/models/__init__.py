"""
Meta-learning model architectures.
"""

from .conv2lr import SameDifferentCNN as Conv2CNN
from .conv4lr import SameDifferentCNN as Conv4CNN
from .conv6lr import SameDifferentCNN as Conv6CNN

__all__ = ['Conv2CNN', 'Conv4CNN', 'Conv6CNN'] 