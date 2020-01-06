import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


'''
参考：
- https://github.com/ajbrock/BigGAN-PyTorch
'''


# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.softplus(-dis_real))
    loss_fake = torch.mean(F.softplus(dis_fake))
    return loss_real, loss_fake


def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


# Perceptual Similarity
# http://arxiv.org/abs/1801.03924
# https://github.com/richzhang/PerceptualSimilarity/blob/master/models/__init__.py
# https://github.com/richzhang/PerceptualSimilarity/blob/master/models/dist_model.py
# https://github.com/richzhang/PerceptualSimilarity/blob/master/models/pretrained_networks.py


class VGGFeatureExtractor(nn.Module):

    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        model = models.vgg19(pretrained=True)
        features = list(model.features.children())
        self.features = nn.Sequential(*features[:35])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, input):
        if input.size(1) == 1:
            input = torch.cat((input, input, input), 1)
        output = self.features(input)
        return output.view(output.size(0), -1)


class ResNet101FeatureExtractor(nn.Module):

    def __init__(self):
        super(ResNet101FeatureExtractor, self).__init__()
        model = models.resnet101(pretrained=True)
        features = list(model.children())
        self.features = nn.Sequential(*features[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, input):
        if input.size(1) == 1:
            input = torch.cat([input, input, input], 1)
        output = self.features(input)
        return output.view(output.size(0), -1)


class PerceptualLoss(nn.Module):

    def __init__(self, net='vgg19', loss='L2', device='cuda'):
        super(PerceptualLoss, self).__init__()
        net = net.lower()
        if net == 'vgg19':
            self.net = VGGFeatureExtractor().to(device)
        elif net == 'resnet101':
            self.net = ResNet101FeatureExtractor().to(device)
        else:
            raise NotImplementedError('Net [%s] is not found' % net)

        loss = loss.lower()
        if loss == 'l1':
            self.loss = nn.L1Loss()
        elif loss == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [%s] is not found' % loss)

        self.net.eval()

    def __call__(self, fake, real):
        pred_fake = self.net(fake)
        pred_real = self.net(real)
        loss = self.loss(pred_fake, pred_real.detach())
        return loss
