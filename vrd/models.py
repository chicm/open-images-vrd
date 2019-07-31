import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from net.senet import se_resnext50_32x4d, se_resnet50, senet154, se_resnet152, se_resnext101_32x4d, se_resnet101
from net.densenet import densenet121, densenet161, densenet169, densenet201
from net.nasnet import nasnetalarge
from net.inceptionresnetv2 import inceptionresnetv2
from net.inceptionv4 import inceptionv4
from net.dpn import dpn98, dpn107, dpn131, dpn92


import settings

c = nn.CrossEntropyLoss()

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def get_num_features(backbone_name):
    if backbone_name in ['resnet18', 'resnet34']:
        ftr_num = 512
    elif backbone_name =='nasnetmobile':
        ftr_num = 1056
    elif backbone_name == 'mobilenet':
        ftr_num = 1280
    elif backbone_name == 'densenet161':
        ftr_num = 2208
    elif backbone_name == 'densenet121':
        ftr_num = 1024
    elif backbone_name == 'densenet169':
        ftr_num = 1664
    elif backbone_name == 'densenet201':
        ftr_num = 1920
    elif backbone_name == 'nasnetalarge':
        ftr_num = 4032
    elif backbone_name in ['inceptionresnetv2', 'inceptionv4']:
        ftr_num = 1536
    elif backbone_name in ['dpn98', 'dpn92', 'dpn107', 'dpn131']:
        ftr_num = 2688
    elif backbone_name in ['bninception']:
        ftr_num = 1024
    else:
        ftr_num = 2048  # xception, res50, etc...

    return ftr_num

def create_imagenet_backbone(backbone_name, pretrained=True):
    if backbone_name in [
        'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnet50', 'senet154', 'se_resnet101',
        'se_resnet152', 'nasnetmobile', 'mobilenet', 'nasnetalarge', 'inceptionresnetv2',
        'dpn98', 'dpn107', 'inceptionv4']:
        backbone = eval(backbone_name)()
    elif backbone_name in ['resnet34', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']:
        backbone = eval(backbone_name)(pretrained=pretrained)
    else:
        raise ValueError('unsupported backbone name {}'.format(backbone_name))
    return backbone

class RelIsNet(nn.Module):
    def __init__(self, backbone_name, num_classes=6, pretrained=True):
        super(RelIsNet, self).__init__()
        print('num_classes:', num_classes)
        self.backbone = create_imagenet_backbone(backbone_name)
        self.label1_emb = nn.Embedding(23, 512, padding_idx=-1)
        ftr_num = get_num_features(backbone_name)

        self.ftr_num = ftr_num
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ftr_num+512, num_classes)
        self.name = 'RelIsNet_{}_{}'.format(backbone_name, num_classes)

    #def logits(self, x):
    #    x = self.avg_pool(x)
    #    x = F.dropout2d(x, 0.4, self.training)
    #    x = x.view(x.size(0), -1)
    #    return self.logit(x)
    
    def forward(self, img, label1):
        x1 = self.backbone.features(img)
        x1 = self.avg_pool(x1).view(x1.size(0), -1)
        x1 = F.dropout(x1, 0.2, self.training)

        x2 = self.label1_emb(label1)
        #print(x1.size(), x2.size())
        x = torch.cat([x1, x2], 1)
        x = F.dropout(x, 0.2, self.training)

        return self.fc(x)

IS_MODEL_DIR = './output'

def create_model(args):
    if args.init_ckp is not None:
        model = RelIsNet(backbone_name=args.backbone, num_classes=args.init_num_classes)
        model.load_state_dict(torch.load(args.init_ckp))
        if args.init_num_classes != args.num_classes:
            model.logit = nn.Linear(model.ftr_num, args.num_classes)
            model.name = '{}_{}_{}'.format(suffix_name, args.backbone, args.num_classes)
    else:
        model = RelIsNet(backbone_name=args.backbone, num_classes=args.num_classes)

    model_file = os.path.join(IS_MODEL_DIR, model.name, args.ckp_name)

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    if args.predict and (not os.path.exists(model_file)):
        raise AttributeError('model file does not exist: {}'.format(model_file))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    
    return model, model_file

from argparse import Namespace

def test():
    x = torch.randn(2, 3, 224, 224).cuda()
    args = Namespace()
    args.init_ckp = None
    args.backbone = 'se_resnext50_32x4d'
    args.ckp_name = 'best_model.pth'
    args.predict = False
    args.num_classes = 6
    label1 = torch.tensor([10]*2).cuda()
    print(label1.size())

    model = create_model(args)[0].cuda()
    y = model(x, label1)
    print(y.size(), y)


if __name__ == '__main__':
    test()
    #convert_model4()
