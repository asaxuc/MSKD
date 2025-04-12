import argparse 
import glob
import os
import warnings
import pickle
import sys
import logging
import torch
import torch.distributed 
import torch.nn as nn
from mmengine import Config
from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from assets import *
import models
import time  
from tools.set_up_seed import setup_seed
import glob
from assets.MSKD import MSKD
import setproctitle
setproctitle.setproctitle("python")
from torch.utils.tensorboard import SummaryWriter 

class BCE(nn.Module):
    def __init__(self, reduction="sum", eps=1e-8):
        super(BCE, self).__init__()
        self.reduction = reduction
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.loss = None

    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        if self.reduction == "sum":
            return -self.loss.sum()
        elif self.reduction == "none":
            return -self.loss
        else:
            raise AttributeError


def criterion_s2(student, teacher):
    N, C = student.shape
    student = torch.sigmoid(student)
    teacher = torch.sigmoid(teacher).detach()
    student = torch.clamp(student, min=1e-9, max=1 - 1e-9)
    teacher = torch.clamp(teacher, min=1e-9, max=1 - 1e-9)
    loss = nn.KLDivLoss(reduction="none")(torch.log(student), teacher) + nn.KLDivLoss(reduction="none")(
        torch.log(1 - student), 1 - teacher)
    loss = loss.sum() / N
    return loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="./configs/coco/config_all.py", type=str, help="path of cfg file")
    parser.add_argument("--mode", default="train-V13", type=str, help="path of data files")
    parser.add_argument("--use-parallel",default=True,type=bool)
    parser.add_argument("--local-rank",type=int) 
    parser.add_argument("--world_size",type=int) 
    parser.add_argument("--method", default="sd", type=str)
    parser.add_argument("--model", default="resnet34", type=str, help="sd, pskd, ud, ddgsd, baseline, kd, byot, clv, dlb")
    parser.add_argument("--ngpu",type=int) 
    parser.add_argument("--para",type=float,default=1) 
    args = parser.parse_args()
    return args


def main_train(args, backbone, train_loader, test_loader, writer, cfg, device, method, **kwargs): 
    criterion =  nn.MultiLabelSoftMarginLoss() 
    if args.use_parallel:
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank], find_unused_parameters=True,output_device=local_rank, broadcast_buffers=True)  
        kwargs["model_teacher"] = torch.nn.parallel.DistributedDataParallel(kwargs["model_teacher"], device_ids=[local_rank], find_unused_parameters=True,output_device=local_rank, broadcast_buffers=True)  

    lr = cfg.lr
    lrp = cfg.lrp
    p1 = []
    p2 = []
    alls = {"resnet34":"fc","mobilenet_v2":"classifier","swin_t":"head"}
    for name, params in backbone.named_parameters():    
        if alls[args.model] not in name:
            p1.append(params)
        else:
            p2.append(params) 
    lla =   [{'params': p1, 'lr': lr*lrp},  {'params': p2, 'lr': lr}] 
    
    optimizer_sp0 = torch.optim.SGD(lla, lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler_sp0 = lr_scheduler.CosineAnnealingLR(optimizer_sp0, T_max=len(train_loader) * cfg.max_epoch,  last_epoch=-1, verbose=False)

    step0 = MSKD(backbone, optimizer_sp0, criterion, scheduler_sp0, writer, device, kwargs["datas"])
    if not cfg.evo:
        step0.learn(args, backbone, cfg.max_epoch, train_loader, test_loader, cfg, method, **kwargs)
    else:
        with torch.no_grad():
            step0.visualize(args, backbone, cfg.max_epoch, train_loader, test_loader, cfg, method, **kwargs)


def main(args, TIMESTAMP, device, datas):
    cfg = Config.fromfile(args.cfg_file)
    print(" %s =%s=> model:%s |%d %d"
          % (cfg.dataset,
             args.method, args.model, cfg.img_size, cfg.batch_size))
    torch.cuda.empty_cache()

    setup_seed(0) 
    if ((args.local_rank==0 and args.use_parallel) or (not args.use_parallel)):
        writer = SummaryWriter(f"./log/{TIMESTAMP}_{args.method}_{args.model}_{cfg.dataset}")
    else:
        writer = None

    train_loader, test_loader = DATA[cfg.dataset](cfg, cfg.data_root, args)

    # teacher model & student model
    if "swin" not in args.model:
        model_student = models.__dict__[args.model](train_loader.num_classes, pretrained=cfg.pretrained)
    else:
        model_student = models.__dict__[args.model](train_loader.num_classes, pretrained=cfg.pretrained, img_size=cfg.img_size)

    model_student = model_student.to(device)

    if args.method == "kd":
        tea = "swin_t"
    else:
        tea = args.model
        
    if args.method:
        if "swin" not in tea:
            model_teacher = models.__dict__[tea](train_loader.num_classes, pretrained=cfg.pretrained)
        else:
            model_teacher = models.__dict__[tea](train_loader.num_classes, pretrained=cfg.pretrained, img_size=cfg.img_size)

        model_teacher = model_teacher.to(device)
    else:
        model_teacher = None 
    alls = {"resnet34":"fc","mobilenet_v2":"classifier","swin_t":"head"}
    for name, params in model_teacher.named_parameters(): # head
        if alls[args.model] not in name:  
            params.requires_grad = False
         
    main_train(args, model_student, train_loader, test_loader, writer, cfg, device, args.method, model_teacher=model_teacher, alpha_t=0.8 ,datas=datas)

    print("Finished!\n")


if __name__ == "__main__":
    args = get_args()  
    if args.use_parallel: 
        local_rank = args.local_rank    
        torch.cuda.set_device(local_rank)  
        print(torch.cuda.current_device())
        torch.distributed.init_process_group("nccl", world_size=args.world_size, rank=local_rank)  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    logger = logging.getLogger('')
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
    file_handler = logging.FileHandler(f'./memo/{args.method}_{TIMESTAMP}.log')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    datas = None

    warnings.filterwarnings("ignore")
    if not os.path.exists("runs"):
        os.mkdir("runs")

    main(args, TIMESTAMP, device, datas)


