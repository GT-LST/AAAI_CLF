import torch
import torch.nn as nn
from transformers import XLNetModel


class ClassificationXLNet(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(ClassificationXLNet, self).__init__()

        self.transformer = XLNetModel.from_pretrained(model_name)
        self.max_pool = nn.MaxPool1d(64)
        self.drop = nn.Dropout(0.3)
        self.linear = nn.Sequential(nn.Linear(768, num_labels))

    def forward(self, x):
        all_hidden = self.transformer(x)
        pooled_output = torch.mean(all_hidden[0], 1)
        predict = self.linear(pooled_output)
        predict = self.drop(predict)

        return predict