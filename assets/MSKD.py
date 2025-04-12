import copy
import glob
import pickle
import os
import time
import torch.distributed
import torchvision 
import torch.nn as nn
from metrics import *
from einops import rearrange, repeat
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torchvision.transforms import Resize
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm
from peerwork import *
import torch.nn.functional as F
from utils import *
from torch.nn import KLDivLoss

def MLD(student, teacher):
    '''
    rectified LD for multi-label
    '''
    eps = 1e-7
    N, C = student.shape
    student = torch.sigmoid(student)
    teacher = torch.sigmoid(teacher)
    student = torch.clamp(student, min=eps, max=1-eps)
    teacher = torch.clamp(teacher, min=eps, max=1-eps)
    loss = KLDivLoss(reduction="none")(torch.log(student), teacher) + KLDivLoss(reduction="none")(torch.log(1 - student), 1 - teacher)  
    loss = loss.sum() / N 
    return loss

def easy_bce(output, targets):
    log_probs = F.log_softmax(output,dim=-1)
    loss = (- targets * log_probs).mean(0).sum()
    return loss

class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, output, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(output)
        loss = (- targets * log_probs).mean(0).sum()
        return loss


def partial_softmax(outputs, targets, **kwargs):
    selective_fc_loss = 0
    for softmax_fc_out_i, target_i in zip(outputs, targets):
        tmp_out = []
        tmp_target = []
        labels = target_i.nonzero()
        for label in labels:
            mask = target_i.clone().byte()
            mask[label] = 0
            mask = ~mask
            tmp_target.append(target_i[mask].nonzero()[0])
            tmp_out.append(softmax_fc_out_i[mask].view((1, -1)))
        tmp_out = torch.cat(tmp_out, dim=0)
        tmp_target = torch.cat(tmp_target, dim=0)
        selective_fc_loss += F.cross_entropy(tmp_out, tmp_target, reduction="sum")
    selective_fc_loss /= len(labels)
    return selective_fc_loss


def my_kl(predicted, target, T):
    predicted = predicted.clamp(min=1e-7) 
    target = target.clamp(min=1e-7) 
    return -(target * torch.log(predicted)).sum(dim=1).mean() * (T**2) #+  (target * torch.log(target)).sum(dim=1).mean()

class Distill_MBD(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(Distill_MBD, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, **kwargs): 
        if "w" not in kwargs.keys():
            p_s = F.log_softmax(y_s / self.T, dim=1)
            p_t = F.softmax(y_t / self.T, dim=1)
            if "coe" in kwargs.keys():
                loss = F.kl_div(p_s, p_t, reduction='mean') * (self.T ** 2)
            else:
                loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
            return loss
        else:
            targets = kwargs["w"]
            p_s =  partial_softmax(y_s / self.T, targets)
            p_t =  partial_softmax(y_t / self.T, targets)
            
            p_t_1 = p_t.clamp(max=1) 
            p_s_1 = p_s.clamp(max=1)    
            
            # add labels:
            # get the 0-1 preds
            x =  p_s_1 
            idxs = torch.sum(targets == 1, dim=1).to(torch.int)
            sorted_outputs = torch.sort(-x, dim=1)[0]
            thr = -sorted_outputs[range(len(targets)), idxs].reshape(len(sorted_outputs), 1)
            preds = torch.zeros(x.shape, dtype=torch.int).cuda()
            preds[x > thr] = 1

            # re-form 0-1 preds
            x = F.sigmoid(x)
            plmax = x.max(dim=-1)[0].detach()
            plmin = x.min(dim=-1)[0].detach() 
            max_min = (targets * plmax.unsqueeze(dim=1) + (1-targets) * plmin.unsqueeze(dim=1)).detach()
            
            # every norm score:
            pos_expo =  targets * torch.abs((p_t_1+1)/2 - p_s_1).clamp(min=0,max=1)**(1/4)
            neg_expo =  (1-targets) * torch.abs((p_t_1)/2- p_s_1).clamp(min=0,max=1)**(1)      
                
            expt = (pos_expo + neg_expo) 
            base = p_s_1    
            p_s = (base**(expt)).clamp_min(1e-7)   
            p_t = (p_t).clamp_min(1e-7)  
            loss = my_kl(p_s, p_t, self.T)  
            return loss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def rkd_3d(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=2)
    prod = torch.bmm(e, e.transpose(1,2))
    res = (e_square.unsqueeze(2) + e_square.unsqueeze(1) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[:,range(e.shape[1]), range(e.shape[1])] = 0
    return res 

def RIS(student, teacher):
    with torch.no_grad():
        t_d = pdist(teacher, squared=False)
        mean_td = t_d[t_d>0].mean()
        t_d = t_d / mean_td

    d = pdist(student, squared=False)
    mean_d = d[d>0].mean()
    d = d / mean_d

    loss = nn.HuberLoss()(d, t_d )
    return loss
 
def relational_3d(student, teacher):
    with torch.no_grad():
        t_d = rkd_3d(teacher, squared=False)
        mean_td = t_d[t_d>0].mean()
        t_d = t_d / mean_td

    d = rkd_3d(student, squared=False)
    mean_d = d[d>0].mean()
    d = d / mean_d

    loss = nn.HuberLoss()(d, t_d)
    return loss
 
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt
    
class MSKD():
    def __init__(self, backbone, optimizer, criterion, scheduler, writer, device, datas):
        self.backbone: nn.Module = backbone
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.writer = writer
        self.scaler = GradScaler()
        self.grad_block = None
        self.fmap_block = None 
        self.mir = AveragePrecisionMeter(False)
        self.resize = Resize([224, 224])
        self.datas = datas
        self.pipe = PipeLine()
        self.la = []

    def learn(self, args, backbone, epoch, train_loader, test_loader, cfg, udec="sd", **kwargs):
        ttime = None  
        for e in range(epoch):
            self.mir.reset()
            self.train(args, cfg, backbone, epoch, e, train_loader, method=udec, **kwargs)
            self.evaluate(args, backbone, test_loader, e)

            # if False:
            map = 100 * self.mir.value().mean() 
            OP, OR, OF1, CP, CR, CF1 = self.mir.overall()
            OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.mir.overall_topk(3)
            
            if ((args.local_rank==0 and args.use_parallel) or (not args.use_parallel)):
                print('Epoch: [{0}]\t'
                    'mAP {map:.3f}'.format(e, map=map))
                print('OP: {OP:.4f}\t'
                'OR: {OR:.4f}\t'
                'OF1: {OF1:.4f}\t'
                'CP: {CP:.4f}\t'
                'CR: {CR:.4f}\t'
                'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                self.writer.add_scalars("VA-Evaluate", {"mAP":map, "OP":OP*100, "OR":OR*100, "OF1":OF1*100, "CP":CP*100, "CR":CR*100, "CF1":CF1*100}, e)
                 
                ttime = self.save(args, backbone, cfg, ttime, udec)
             

    def load(self, backbone):
        backbone.load_state_dict(torch.load(glob.glob("./save/*")[-1]))

    def save(self, args, backbone, cfg, ttime, use):
        if not ttime:
            timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        else:
            timestamp = ttime
            
        state = backbone.state_dict() 
        torch.save(state, "./save/{}-{}_{}_{}_{}_lr{}_epoch{}.pth".format(use, cfg.step, timestamp, cfg.dataset, args.model, cfg.lr, cfg.max_epoch))
        return timestamp


    def random_resized_crop(self, vanillaware, img_size, iteration):
        # return: 3*B len turple
        rrp = torchvision.transforms.Compose([
            RandomResizedCrop((img_size, img_size), scale=(0.2, 0.3),
                                                     ratio=(2.0 / 5.0, 5.0 / 3.0)),
        ])
        batch_loc = []
        for _ in range(iteration):
            for va in vanillaware:
                li = list(rrp(va)) 
                li[-1] = list(li[-1])
                li[-1][2] += li[-1][0]
                li[-1][3] += li[-1][1]  
                batch_loc.append(li) 
        assert len(batch_loc) == 3*vanillaware.shape[0]
        return batch_loc

    def five_crop(self, ref, img_size, use_crop="random"):
        # return: 3*B len turple
        if use_crop == "random":
            fcrop = torchvision.transforms.Compose([
                FiveCrop(ref.shape[-2] // 2, re_size=img_size)  # resizing is contained.
            ])
        elif use_crop == "five":
            fcrop = torchvision.transforms.Compose([
                FiveCrop(ref.shape[-2] // 2, re_size=img_size)   # resizing is contained.
            ])
        else:
            raise NotImplementedError()

        batch_loc = [] 
        for va in ref:
            li = fcrop(va)
            batch_loc += li[:-1] 
        assert len(batch_loc) == 4 * ref.shape[0]
        return batch_loc

    def obtain_diff(self,pred, preds, targets):
        # x 3*B
        iteration = preds.shape[0] // pred.shape[0]
        mp = pred.repeat(iteration, 1)
        diff = preds - mp
        fla = diff.flatten()
        thrp = fla.sort(dim=0)[0][-(fla.shape[0]*targets.sum())//targets.numel()]
        thrn = fla.sort(dim=0)[0][(fla.shape[0]*targets.sum())//targets.numel()]
        t = torch.zeros_like(diff).to(self.device)
        t[diff > thrp] = 1
        t[diff < thrn] = -1  # reverse adding
         
        return t  # the value of label graph
    
    def save_iter(self,  t, targets): 
        '''
        unit-test
        '''
        for idx,target in enumerate(targets):
            if torch.equal(target.detach(), torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0]).cuda()):
                print("the epoch")
                os.makedirs("/home/xuc/others/",exist_ok=True)
                with open(f"/home/xuc/others/label.pth","wb") as f:
                    for tt in t.chunk(4,dim=0):
                        self.la.append(tt[idx])
                    pickle.dump(self.la, f)
                f.close()
    
    def get_adj_matrix(self,x):
        ma = rkd_3d(x) / (torch.abs(rkd_3d(x)).max())
        return ma
    
    def label_graph_prop(self, x, fes):
        ma = self.get_adj_matrix(fes)
        ma_deg = ma.sum(dim=-1) 
        ma_deg = torch.diag_embed(ma_deg,0).to(self.device) + torch.eye(ma_deg.shape[1]).to(self.device) + 1e-6
        o = (torch.bmm(torch.bmm(torch.pow(ma_deg,-1/2), ma), torch.pow(ma_deg,-1/2))) 
        o = torch.bmm(o, x) 
        return o  

    def train(self, args, cfg, backbone, epochs, epoch, train_loader, method='sd', **kwargs):
        lcr = [0, 0, 0, 0]
        register_1 = None
        register_2 = None  
        use_five = True 
        if not use_five:
            iteration = cfg.iteration
        else:
            iteration = 4
 
        tra_function = lambda x: rearrange(x,"(A B) C-> A B C",A=iteration).transpose(0,1) 
        model_teacher = kwargs["model_teacher"]
        if method == "pskd":
            alpha_T = kwargs["alpha_t"]
            alpha_t = alpha_T * ((epoch + 1) / epochs)
            alpha_t = max(0, alpha_t)
            if epoch != 0:
                model_teacher.load_state_dict(torch.load(glob.glob(f"./save/pskd_{args.model}/*")[-1]))
            elif epoch == 0:
                model_teacher.load_state_dict(backbone.state_dict()) 
                
        if method == "ud":
            if cfg.dataset == "voc":  
                model_teacher.load_state_dict(torch.load(f"./save/ud/udp-{args.model}.pth") )
                # model_teacher.load_state_dict(torch.load("./save/ud/nnn-decoder.pth") )
                print("voc udp loaded1")
            elif cfg.dataset == "fli":
                model_teacher.load_state_dict(torch.load(f"./save/ud/fli-{args.model}.pth") )
                print("flickr udp loaded")
            elif cfg.dataset == "coco":
                if args.model == "resnet34":
                    model_teacher.load_state_dict(torch.load(f"./save/coco34.pth") )
                    print("coco udp 34 loaded")
                else: 
                    model_teacher.load_state_dict(torch.load(f"./save/ud/coco-{args.model}.pth") )
                    print("coco udp swin loaded")

        for i, (inputs, targets) in tqdm(enumerate(train_loader)):
            backbone.train()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device) 
            B, C = targets.shape 
            
            if method== "ddgsd":
                inputs = torch.cat((inputs[:,0,:,:,:],inputs[:,1,:,:,:]))  
                tgt = torch.cat((targets, targets),dim=0)
            else:
                tgt = targets.float()
            
            loss = torch.FloatTensor([0.]).to(self.device)
            if method == "sd":
                if step == 1:
                    outputs = backbone(inputs, method=method)[0]
                    loss = self.criterion(outputs, tgt) 
                elif step == 2:    
                    model_teacher.load_state_dict(backbone.state_dict()) 
                    model_teacher.eval()
                    # step 1:
                    with torch.no_grad(): 
                        if not use_five:
                            batch_loc = self.random_resized_crop(inputs, cfg.img_size,  cfg.iteration)   
                        else:
                            batch_loc = self.five_crop(inputs, cfg.img_size)  
                        pin1 = torch.stack([i[0].to(self.device) for i in batch_loc],dim=0)
                        loc = [torch.tensor(i[1]).to(self.device).unsqueeze(dim=0).float() for i in batch_loc] 
                        [patcho, dfea, rrh] = model_teacher.forward(pin1, method=args.udec)     
                         
                    backbone.train()   
                    [outputs, slicep, sfea, rh, h]  = backbone.forward(inputs, loc=loc, method=args.udec, iter=iteration)   # rh, h : B 20 768
                       
                    # step 2:
                    pot = tra_function(patcho) 
                    sop = tra_function(slicep)
                     
                    with torch.no_grad(): 
                        value_of_label_graph = self.obtain_diff(outputs, patcho, targets)
                        # self.save_iter( value_of_label_graph, targets)
                        value_of_label_graph = rearrange(value_of_label_graph,"(A B) C-> A B C",A=iteration).transpose(0,1) # B A C 
                        o1 = value_of_label_graph
                        for _ in range(int(args.para)):
                            o1 = self.label_graph_prop(o1, sop)     
                        o_all = (pot * (o1).softmax(dim=1)).mean(dim=1)     
                        o1 = o1.unsqueeze(dim=-1)
                        # o_ml = (poh * (o1).softmax(dim=1)).mean(dim=1)               

                    l1 = relational_3d(sop, pot)
                    loss += l1    
                    if register_2 is not None:   
                        [xo, _, _] = model_teacher.forward(register_1, method=args.udec)
                        l2 = Distill_MBD(4)(outputs, xo.detach(), w=targets)
                        loss += l2 
                         
                    l3 = Distill_MBD(4)(outputs, o_all, w=targets)
                    loss += l3
                    loss += self.criterion(outputs, tgt)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(backbone.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step()
                outputs = outputs[:B]   
                
            elif method == 'baseline':
                outputs = backbone(inputs, method=method)[0]
                loss =  self.criterion(outputs, tgt)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(backbone.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step() 
            elif method == "kd":
                with torch.no_grad():
                    ot = model_teacher(inputs, method=method)[0]
                outputs = backbone(inputs, method=method)[0]
                loss = self.criterion(outputs, tgt)
                loss += MLD(outputs, ot.detach())
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(backbone.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step()
                
            elif method == "byot":
                outputs, logits, features = backbone(inputs, method=method)
                loss = self.criterion(outputs, tgt) # /len(train_loader)
                
                loss_div = torch.tensor(0.).to(self.device)
                loss_cls = torch.tensor(0.).to(self.device)
                for k in range(len(logits)):
                    loss_cls += self.criterion(logits[k], targets)
                    loss_div += Distill_MBD(1)(logits[k], outputs.detach())

                for j in range(len(features)-1): 
                    loss_div += 0.5 * 0.1 * ((features[j] - features[len(features)-1].detach()) ** 2).mean()
                loss += loss_cls
                loss += loss_div
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(backbone.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step()
            elif method == "pskd":
                # with torch.no_grad():
                outputs = backbone(inputs, method=method)[0]
                loss = self.criterion(outputs, tgt) # /len(train_loader)
                with torch.no_grad():
                    outputs_p =  model_teacher(inputs, method=method)[0]
                soft_targets = ((1 - alpha_t) * targets) + (alpha_t * F.sigmoid(outputs_p).detach())
                soft_targets = soft_targets.to(self.device)
                softmax_outputs = outputs
                loss1 = 1e-3 * Distill_MBD(4)(softmax_outputs, soft_targets)
                loss += loss1
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(backbone.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step()
            elif method == 'dlb':
                if i != 0:
                    logits_1 = backbone(torch.cat((inputs, register_1)), method=method)[0]
                    outputs = logits_1[:int(logits_1.shape[0]/2)]
                    logits_pre = logits_1[int(logits_1.shape[0]/2):]
                    
                    loss = self.criterion(outputs, tgt)
                    loss += MLD(logits_pre, register_2.detach())
                else:
                    outputs = backbone(inputs, method=method)[0]
                    loss = self.criterion(outputs, tgt)
                    
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(backbone.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step() 
                        
            elif method == "ddgsd":
                outputs,features = backbone(inputs, method=method)
                logit = outputs
                batch_size = logit.size(0) // 2 
                loss = self.criterion(logit, tgt) # /len(train_loader) 
                loss_div = torch.tensor(0.).to(self.device)
                loss_div += Distill_MBD(4)(logit[batch_size:], logit[:batch_size].detach())
                loss_div += Distill_MBD(4)(logit[:batch_size], logit[batch_size:].detach())
                loss_div += 5e-4 * (features[:batch_size].mean() - features[batch_size:].mean()) ** 2
                loss += 0.1*loss_div
                outputs = outputs[:batch_size]
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(backbone.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step() 
                
            elif method == "uskd":
                outputs, usko = backbone(inputs, method=method)
                loss = easy_bce(outputs, tgt)
                loss += USKDLoss(num_classes=train_loader.num_classes)(usko, outputs, tgt)
                    
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(backbone.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step()
            else:
                pass

            # sync:
            if True: #not args.use_parallel:   
                outputs = outputs.clamp(max=100,min=-100)
                lcr[0] += loss.detach().item()
                try:
                    lcr[3] += self.compute_mAP(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy())[-1] 
                except:
                    lcr[3] += self.compute_mAP(targets.detach().cpu().numpy(), targets.detach().cpu().numpy())[-1] 
            else:
                torch.distributed.barrier() 
                reduced_loss = reduce_mean(loss.detach(), args.ngpu) 
                reduced_outputs = reduce_mean(outputs.detach(), args.ngpu) 
                lcr[0] += reduced_loss.detach().item()
                lcr[3] += self.compute_mAP(targets.detach().cpu().numpy(), reduced_outputs.detach().cpu().numpy())[-1]
            
            if method == 'dlb':
                register_1 = inputs.detach()
                register_2 = outputs.detach()
            if method == 'dlb+':
                register_1 = inputs.detach()
                register_2 = outputs.detach()
            if method == 'sd':
                register_1 = inputs.detach() 
                register_2 = outputs.detach()

            
            if (i + 1) % 20 == 0 and ((args.local_rank==0 and args.use_parallel) or (not args.use_parallel)):    
                print('mAP:{:>5,.2f}%   | Loss: [ Total: {li[0]:^10,.4f} | Positive: {li[1]:^10,.4f} | Negative: {li[2]:^10,.4f}]'.format(
                        100. * lcr[3] / (i + 1),
                        li=[j / (i + 1) for j in lcr[:3]]))

        if method == "pskd":
            os.makedirs(f"./save/pskd_{args.model}",exist_ok=True)
            exist_name = glob.glob(f"./save/pskd_{args.model}/*") 
            strftime = time.strftime("%Y-%m-%d_%H-%M-%S")
            if epoch == 0: 
                myn = f"./save/pskd_{args.model}/mini_{strftime}.pth" 
            else:
                myn = exist_name[-1]
            torch.save(backbone.state_dict(),myn)

    def fit_zipfs_distribution(self, targets, pred, thr, logi):
        # return a zipf's distribution function. size: B, C
        tgt_msk = targets.to(torch.bool)
        corr_msk = pred.eq(targets)
        c0 = (tgt_msk) & corr_msk
        c1 = (~tgt_msk) & corr_msk
        c2 = (tgt_msk) & (~corr_msk)
        c3 = (~tgt_msk) & (~corr_msk)

        delta = c0 * ((1 - thr) + (-logi) - 1) + c1 * ((thr) + logi - 1) + c2 * ((1 - thr) + (-logi)) + c3 * (
                (thr) + logi) 
        
        delta[targets] = -2
        max_delta = delta.max(dim=-1)[0]
        min_delta = delta.topk(k=2, dim=-1, largest=False)[0][:, 1]
        # b = 0.1 + max_delta - min_delta
        b = 0.1 + 0 - min_delta
        x_every = 0.1 + delta - min_delta.unsqueeze(dim=-1)
        return b, delta, x_every

    def sample_from_zipfs_d(self, targets, pred, thr, logi, num_of_blocks=4, min_eps=0.2, mid_eps=0.5):
        b, delta, x_every = self.fit_zipfs_distribution(targets, pred, thr, logi)
        # prob = (0.8*(b-0.1)**2).unsqueeze(dim=-1) / (x_every - (0.2-b.unsqueeze(dim=-1))) ** 2  # threshold
        bly = x_every * (mid_eps - min_eps) / (b.unsqueeze(dim=-1) - 0.1) + (0.2 - 0.1 * (mid_eps - min_eps) / (b.unsqueeze(dim=-1) - 0.1))  # probability
        prob = bly
        sample = torch.rand((num_of_blocks, *prob.size())).to(self.device)
        prob = prob.unsqueeze(dim=0)
        uns_targets = repeat(targets, "B C -> N B C", N=num_of_blocks)
        # n layers. Expectations of n times separate sampling.
        probas = torch.zeros_like(uns_targets).to(self.device)
        for u in range(uns_targets.shape[0]-1, -1, -1):
            if u == uns_targets.shape[0] -1 :
                probas[u] = (bly)
            else:
                probas[u] = probas[u+1] + (1 - bly) ** (uns_targets.shape[0]-1-u) * (bly)

        lay_existence = (sample >= probas)
        tot_existence = ((sample <= prob) & lay_existence)
        uns_targets = probas

        assert uns_targets.shape == tot_existence.shape
        tailored_msk = tot_existence
        print(tailored_msk.to(torch.int).sum(), tailored_msk.numel())
        return uns_targets, tailored_msk
  
    def tailor_targets(self, targets1, outputs1, mio1):
        targets = targets1.detach().cpu().numpy()

        preds = torch.zeros_like(outputs1, dtype=torch.int).to(self.device)
        spreds = torch.zeros_like(outputs1, dtype=torch.int).to(self.device)
        accu = torch.zeros_like(outputs1, dtype=torch.int).to(torch.bool).to(self.device)
        rem = torch.zeros_like(outputs1, dtype=torch.int).to(torch.bool).to(self.device)
        ret = []

        outputs = outputs1.detach().cpu().numpy()
        idxs = np.sum(targets == 1, axis=1).astype(int)
        s_outputs = np.sort(-outputs, axis=1)
        sthr = -s_outputs[range(len(targets)), idxs].reshape(len(s_outputs), 1)
        logi = outputs - sthr
        spreds[outputs > sthr] = 1

        for ii,mio_o in enumerate(mio1):
            mio_outputs = mio_o.detach().cpu().numpy()
            idxs = np.sum(targets == 1, axis=1).astype(int)
            sorted_outputs = np.sort(-mio_outputs, axis=1)
            thr = -sorted_outputs[range(len(targets)), idxs].reshape(len(sorted_outputs), 1)
            logi = mio_outputs - thr
            preds[mio_outputs > thr] = 1

            corr_msk = preds.eq(targets1)
            if ii == len(mio1)-1:
                ret.append(accu | ~accu)
            else:
                ret.append(corr_msk | accu)  
            # logging 
            accu = accu | corr_msk 

        return ret

    def compute_mAP(self, labels, outputs):
        APs = []
        for j in range(labels.shape[1]):
            new_AP = average_precision_score(labels[:, j], outputs[:, j])
            APs.append(new_AP)
        mAP = np.mean(APs)
        return APs, mAP

    def test(self, outputs, targets):
        idxs = np.sum(targets == 1, axis=1).astype(int)
        sorted_outputs = np.sort(-outputs, axis=1)
        thr = -sorted_outputs[range(len(targets)), idxs].reshape(len(sorted_outputs), 1)

        preds = np.zeros(outputs.shape, dtype=np.int64)
        preds[outputs > thr] = 1
        # print(preds.sum(axis=0)/len(preds))
        APs, mAP = self.compute_mAP(targets, outputs)  # average precision & mean average precision
        of1 = f1_score(targets, preds, average="micro")  # overall f1 score
        cf1 = f1_score(targets, preds, average="macro")  # per-class f1 score

        print("mAP: {:.2f}  OF1: {:.2f}  CF1: {:.2f}".format(mAP * 100, of1 * 100, cf1 * 100))

        return APs, mAP, of1, cf1
    
    @torch.no_grad()
    def evaluate(self, args, backbone, eval_loader, e): 
        print(f"Epoch {e}: Evaluation: ", end="") 
        backbone.eval()
 
        acl = 0
        for i, (input, target) in tqdm(enumerate(eval_loader)):
            input = input.to(self.device)
            target = target.to(self.device)

            odd = backbone(input, method=args.udec)[0]
            # output = torch.sigmoid(odd.detach()) 
            
            # #all-reduce
            # torch.distributed.barrier() 
            # odd = reduce_mean(odd.detach(), args.ngpu)  
            
            
            self.mir.add(odd.detach(), target.float().detach())
            # acl += self.compute_mAP(target.detach().cpu().numpy(), odd.detach().cpu().numpy())[-1] 
            
        # if args.local_rank == 0:
        #     print(f"mAP: {acl/len(eval_loader)}")
            
        # outputs = np.concatenate(outputs)
        # odds = torch.cat(odds).cpu().detach()
        # targets = torch.cat(targets).cpu().detach()

        # APs, mAP, of1, cf1 = self.test(odds, targets)
        APs, mAP, of1, cf1 = [None] * 4

        return APs, mAP, of1, cf1

    def visual_eval(self, args, backbone, eval_loader, e):
        model = backbone
        print(f"Epoch {e}: Evaluation: ", end="")

        deploy_model = model
        deploy_model.eval()
 
        for i, (input, target) in tqdm(enumerate(eval_loader)):
            input = input.to(self.device)
            target = target.to(self.device)

            odd = deploy_model(input, method=args.udec)[0]
            # output = torch.sigmoid(odd.detach())  
            self.mir.add(odd.detach(), target.float().detach()) 


    def visualize(self, args, backbone, epoch, train_loader, test_loader, cfg, udec="sd", **kwargs):
        path = "/home/xuc/MSKD/save/baseline_2024-07-07-12-20_voc_resnet34_lr0.01_epoch100.pth"
        backbone.load_state_dict(torch.load(path))
        for e in range(1):
            self.mir.reset()  
            self.visual_eval(args, backbone, test_loader, e)
            lalla = 100 * self.mir.value()
            map = lalla.mean() 
            OP, OR, OF1, CP, CR, CF1 = self.mir.overall()
            OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.mir.overall_topk(3)
            
            if ((args.local_rank==0 and args.use_parallel) or (not args.use_parallel)):   
                print(lalla)
                print('Epoch: [{0}]\t'
                    'mAP {map:.3f}'.format(e, map=map))
                print('OP: {OP:.4f}\t'
                'OR: {OR:.4f}\t'
                'OF1: {OF1:.4f}\t'
                'CP: {CP:.4f}\t'
                'CR: {CR:.4f}\t'
                'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                 
