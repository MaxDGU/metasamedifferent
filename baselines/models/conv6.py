import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import glob
from PIL import Image
from torchvision import transforms
import json
import argparse
#reused methods across all models 
from .utils import train_epoch, validate, EarlyStopping, train_model, SameDifferentDataset


class SameDifferentCNN(nn.Module):
    def __init__(self):
        super(SameDifferentCNN, self).__init__()
        
        # First layer: 6x6 filters with 18 filters
        self.conv1 = nn.Conv2d(3, 18, kernel_size=6, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(18, track_running_stats=False)
        
        # Subsequent layers: 2x2 filters with doubling filter counts
        self.conv2 = nn.Conv2d(18, 36, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(36, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(36, 72, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(72, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(72, 144, kernel_size=2, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(144, track_running_stats=False)
        
        self.conv5 = nn.Conv2d(144, 288, kernel_size=2, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(288, track_running_stats=False)
        
        self.conv6 = nn.Conv2d(288, 576, kernel_size=2, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(576, track_running_stats=False)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout2d = nn.Dropout2d(0.1)  # Add spatial dropout
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._initialize_size()
        
        # Three FC layers with 1024 units each
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        
        # Dropouts for FC layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        
        # Final classification layer
        self.classifier = nn.Linear(1024, 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_size(self):
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)
    
    def _initialize_weights(self):
        # Initialize convolutional layers
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # Initialize fully connected layers
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(fc.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(fc.bias, 0)
        
        # Initialize classifier
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        # Convolutional layers with spatial dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout2d(x)
        
        x = x.reshape(x.size(0), -1)
        
        # FC layers with dropout
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        
        return self.classifier(x)


#main just to test...
def main(args):
    """Main entry point for training Conv6 model."""
    args.dataset_class = SameDifferentDataset
    train_model(SameDifferentCNN, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='SVRT task number')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    args = parser.parse_args()
    
    print(f"\nStarting training for task {args.task} with seed {args.seed}")
    main(args) 