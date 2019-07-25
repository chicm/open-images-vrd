import os
import glob
import argparse
import numpy as np
import pandas as pd
import logging as log
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import settings
from is_loader import get_val_loader, get_train_loader, get_test_loader
import cv2
from models import create_model
from torch.nn import DataParallel
from tqdm import tqdm
from apex import amp

class FocalLoss(nn.Module):
    def forward(self, x, y):
        alpha = 0.25
        gamma = 2

        p = x.sigmoid()
        pt = p*y + (1-p)*(1-y)       # pt = p if t > 0 else 1-p
        w = alpha*y + (1-alpha)*(1-y)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        w = w.detach()
        #w.requires_grad = False
        #return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
        return F.binary_cross_entropy_with_logits(x, y, w, reduction='none')

class FocalLoss2(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        #if len(loss.size())==2:
        #    loss = loss.sum(dim=1)
        return loss

#criterion = nn.BCEWithLogitsLoss(weight=cls_weights, reduction='none')
criterion = nn.CrossEntropyLoss() #reduction='none')
#criterion = FocalLoss2()

def train(args):
    print('start training...')
    model, model_file = create_model(args)
    #model = model.cuda()
    
    model = model.cuda()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name

    val_loader = get_val_loader(batch_size=args.val_batch_size, val_num=args.val_num, dev_mode=args.dev_mode)
    train_loader = get_train_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)

    best_metrics = 0.
    best_key = 'acc'

    print('epoch |    lr    |      %        |  loss  |  avg   |  loss  |  acc   |   best  | time |  save |')

    if not args.no_first_val:
        val_metrics = validate(args, model, val_loader)
        print('val   |          |               |        |        | {:.4f} | {:.4f} | {:.4f} |        |        |'.format(
            val_metrics['valid_loss'], val_metrics['acc'], val_metrics[best_key] ))

        best_metrics = val_metrics[best_key]

    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_metrics)
    else:
        lr_scheduler.step()
    train_iter = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = 0
        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target  = data
            img, target = img.cuda(), target.cuda()
            
            output = model(img)

            loss = criterion(output, target)
            batch_size = img.size(0)
            #(batch_size * loss).backward()

            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            print('\r {:4d} | {:.6f} | {:06d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                if isinstance(model, DataParallel):
                    torch.save(model.module.state_dict(), model_file+'_latest')
                else:
                    torch.save(model.state_dict(), model_file+'_latest')

                val_metrics = validate(args, model, val_loader)
                
                _save_ckp = ''
                if args.always_save or val_metrics[best_key] > best_metrics:
                    best_metrics = val_metrics[best_key]
                    if isinstance(model, DataParallel):
                        torch.save(model.module.state_dict(), model_file)
                    else:
                        torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                print(' {:.4f} | {:.4f} | {:.4f} | {:.2f} |  {:4s} |'.format(
                    val_metrics['valid_loss'], val_metrics['acc'], best_metrics,
                    (time.time() - bg) / 60, _save_ckp))

                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(best_metrics)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    #del model, optimizer, lr_scheduler
        
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def accuracy(output, label, topk=(1,)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    return res


def validate(args, model: nn.Module, valid_loader):
    #criterion_val = nn.BCEWithLogitsLoss(reduction='none')
    model.eval()
    all_losses, corrects, total_num = [], 0, 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            all_losses.append(loss.item())
            #print(targets.size(), outputs.size())
            #_, preds = outputs.max(1)
            #print('targets:', targets.cpu().numpy())
            #print('preds:', preds.cpu().numpy())
            #break

            corrects += accuracy(outputs, targets)[0]
            total_num += len(inputs)

    metrics = {}
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['acc'] = corrects / total_num
    return metrics


def _reduce_loss(loss):
    #print('loss shape:', loss.shape)
    return loss.sum() / loss.shape[0]

def pred_model_output(model, loader, labeled=True):
    model.eval()
    scores, preds, labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, total=loader.num // loader.batch_size):
            if labeled:
                img = batch[0].cuda()
                labels.append(batch[1])
            else:
                img = batch.cuda()
            output = model(img)

            score, pred = output.max(1)
            scores.append(score.cpu())
            preds.append(pred.cpu())

    scores = torch.cat(scores).numpy()
    preds = torch.cat(preds).numpy()
    print(preds.shape)

    if labeled:
        labels = torch.cat(labels).numpy()
        return preds, scores, labels
    else:
        return preds, scores

def tta_validate(args):
    model, _ = create_model(args)
    #model = model.cuda()
    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name
    model = model.cuda()

    #preds = []
    labels = None
    for i in range(args.tta_num):
        _, val_loader = get_train_val_loaders(val_batch_size=args.val_batch_size, val_num=args.val_num, dev_mode=args.dev_mode, val_tta=i)
        pred, labels = pred_model_output(model, val_loader)
        #preds.append(pred)
        np.save('output/val/val_tta_pred_{}.npy'.format(i), pred)
    #tta_pred = np.mean(preds, 0)
    #np.save('val_tta_pred.npy', tta_pred)
    np.save('output/val/val_labels.npy', labels)
    print('computing score...')
    calc_val_score(args.tta_num)


def predict(args):
    assert args.dets_file_name is not None
    model, _ = create_model(args)
    #model = model.cuda()
    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name
    model = model.cuda()

    preds, scores = [], []
    for i in range(args.tta_num):
        test_loader = get_test_loader(args.dets_file_name, batch_size=args.val_batch_size, dev_mode=args.dev_mode)
        pred, score = pred_model_output(model, test_loader, labeled=False)
        preds.append(pred)

    tta_pred = np.mean(preds, 0).astype(np.int32)
    tta_score = np.mean(scores, 0)

    print(tta_pred.shape)
    print(tta_pred[:2])

    create_submission(args, test_loader.df, tta_pred, tta_score)

classes = ['none', '/m/083vt', '/m/02gy9n', '/m/05z87', '/m/04lbp', '/m/0dnr7']

def create_submission(args, meta_df, preds, scores):
    pred_dict = {}
    for i, row in meta_df.iterrows():
        if args.dev_mode and i == len(preds):
            break
        img_id = row.ImageID
        if img_id not in pred_dict:
            pred_dict[img_id] = []

        if preds[i] != 0:
            label = classes[preds[i]]
            det = [row.Confidence, row.LabelName, row.XMin, row.YMin, row.XMax, row.YMax, label, row.XMin, row.YMin, row.XMax, row.YMax, 'is']
            pred_dict[img_id].extend(det)
            
    df_test = pd.read_csv(os.path.join(settings.DATA_DIR, 'VRD_sample_submission.csv'))
    df_test['PredictionString'] = df_test.ImageId.map(lambda x: ' '.join(pred_dict[x]) if x in pred_dict else '')
    df_test.to_csv(args.sub_file, index=False)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='backbone')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=768, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=1024, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=200, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='Adam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=3, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=8, type=int, help='lr scheduler patience')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--init_num_classes', type=int, default=6, help='init num classes')
    parser.add_argument('--num_classes', type=int, default=6, help='init num classes')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--val_score', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--apex', action='store_true')
    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_pretrained.pth',help='check point file name')
    parser.add_argument('--sub_file', type=str, default='sub1.csv')
    parser.add_argument('--dets_file_name', type=str, default=None)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    parser.add_argument('--val_num', default=10000, type=int, help='number of val data')
    parser.add_argument('--tta_num', default=1, type=int, help='number of val data')
    #parser.add_argument('--img_sz', default=256, type=int, help='image size')
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)

    if args.predict:
        predict(args)
    else:
        train(args)
