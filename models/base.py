import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward_train(self, input, target, data_meta, **kwargs):
        pass

    def simple_test(self, input, target, data_meta, **kwargs):
        pass

    def forward(self, input, target, data_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(input, target, data_meta, **kwargs)
        else:
            return self.simple_test(input, target, data_meta, **kwargs)
