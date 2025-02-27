"""
baseline models
"""

from .conv2 import SameDifferentCNN as Conv2CNN
from .conv4 import SameDifferentCNN as Conv4CNN
from .conv6 import SameDifferentCNN as Conv6CNN

__all__ = ['Conv2CNN', 'Conv4CNN', 'Conv6CNN'] 