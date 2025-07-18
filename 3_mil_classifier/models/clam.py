import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..topk.utils.utils import initialize_weights
import numpy as np
import os
# from ..topk import SmoothTop1SVM
from einops import rearrange 

"""
Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M. and Mahmood, F., 2021. 
Data-efficient and weakly supervised computational pathology on whole-slide images. 
Nature biomedical engineering, 5(6), pp.555-570.
"""

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):    

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):  
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):  
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)   # element-wise multiplication
        A = self.attention_c(A)  # N x n_classes
        return A, x  


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
    def __init__(self,
                 feat_dim=1024,
                 hidden_feat=256,
                 gate=True,
                 size_arg="small",
                 dropout=False,
                 k_sample=8,
                 n_classes=2,
                 subtyping=False,
                 instance_eval=True
                 ):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [feat_dim, hidden_feat, 256], "big": [feat_dim, hidden_feat, 384]}
        size = self.size_dict[size_arg]  # size [1024,256,256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.instance_eval = instance_eval

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device, dtype=torch.uint8).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device, dtype=torch.uint8).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # print(f"inst_eval top k: {A.shape}")
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        instance_loss_fn = SmoothTop1SVM(n_classes=self.n_classes).cuda(device)
        instance_loss = instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        instance_loss_fn = SmoothTop1SVM(n_classes=self.n_classes).cuda(device)
        instance_loss = instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, batch, return_global_feature=False):
        h, label = batch[0], batch[1]
        h = h.squeeze()
        # print(f"raw h: {h.shape}")
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        if self.instance_eval:
            # an ugly walkaround to deal with small bag size
            original_k_sample = self.k_sample
            if A.shape[1] < 2 * self.k_sample:
                self.k_sample = (A.shape[1] - 1) // 2

            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            self.k_sample = original_k_sample

        M = torch.mm(A, h)

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if self.instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}

        if return_global_feature:
            return M, results_dict
        else:
            return logits, Y_prob, Y_hat, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(
        self, 
        feat_dim=1024, 
        hidden_feat=256,
        gate=True, 
        size_arg="small", 
        dropout=False, 
        k_sample=8, 
        n_classes=2,
        subtyping=False, 
        instance_eval=True
        ):
        nn.Module.__init__(self)
        self.size_dict = {"small": [feat_dim, hidden_feat, 256], "big": [feat_dim, hidden_feat, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.instance_eval = instance_eval

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        h, label = batch[0], batch[1]
        device = h.device
        h = h.squeeze()
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        if self.instance_eval:
            # an ugly walkaround to deal with small bag size
            original_k_sample = self.k_sample
            if A.shape[1] < 2 * self.k_sample:
                self.k_sample = (A.shape[1] - 1) // 2

            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            self.k_sample = original_k_sample

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if self.instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}

        return logits, Y_prob, Y_hat, results_dict



class CLAM_Batch(nn.Module):
    def __init__(self,
                 feat_dim=768,
                 n_classes=2,
                 hidden_feat=256,
                 gate=True,
                 size_arg="small",
                 dropout=True,
                 k_sample=16,
                 subtyping=False,
                 instance_eval=False
                 ):
        super(CLAM_Batch, self).__init__()
        self.size_dict = {"small": [feat_dim, hidden_feat, 256], "big": [feat_dim, hidden_feat, 384]}
        size = self.size_dict[size_arg]    # [2048, 768, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]  # 每个instance的维度先从1024降到512维度
        if dropout:  # 2048维度使用了dropout
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Sequential(nn.LayerNorm(size[1]),# 768
                                         nn.Linear(size[1], 128),  # 768-->128
                                         nn.Sigmoid(),
                                        #  nn.Dropout(0.25),
                                         nn.Linear(128,n_classes)
                                         )

        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        self.k_sample = k_sample
        
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.instance_eval = instance_eval

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device, dtype=torch.uint8).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device, dtype=torch.uint8).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # print(f"inst_eval top k: {A.shape}")
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        instance_loss_fn = SmoothTop1SVM(n_classes=self.n_classes).cuda(device)
        instance_loss = instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        instance_loss_fn = SmoothTop1SVM(n_classes=self.n_classes).cuda(device)
        instance_loss = instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, batch, return_global_feature=False):
        h, label = batch  
        
        bsz = h.shape[0]   
        h = rearrange(h, 'b n d -> (b n) d')
        A, h = self.attention_net(h)  

        A = rearrange(A, '(b n) d -> b d n', b=bsz) 
        h = rearrange(h, '(b n) d -> b n d', b=bsz) 

        A = F.softmax(A, dim=-1)  # softmax over N
        
        if self.instance_eval: 

            total_inst_loss = 0.0
            all_preds = []
            all_targets = []

            for inst in range(bsz):  

                A_inst = A[inst]  
                h_inst = h[inst]
                label_inst = label[inst]

                # an ugly walkaround to deal with small bag size
                original_k_sample = self.k_sample 
                if A_inst.shape[1] < 2 * self.k_sample:
                    self.k_sample = (A_inst.shape[1] - 1) // 2   # //  整除  比如  n=16   则k=7 取前7个和后7个  n=17的时候，取前8个和后8个


                inst_labels = F.one_hot(label_inst, num_classes=self.n_classes).squeeze()  # binarize label  

                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item()
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1:  # in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A_inst, h_inst, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:  # out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A_inst, h_inst, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss

                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)

                self.k_sample = original_k_sample

        M = torch.bmm(A, h).squeeze(1) 

        logits = self.classifiers(M) 
        Y_hat = torch.topk(logits, 1, dim=1)[1]   #
        Y_prob = F.softmax(logits, dim=1)
        if self.instance_eval:  
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}

        if return_global_feature:
            return M, results_dict
        else:
            return logits, Y_prob, Y_hat, results_dict




if __name__ == "__main__":
    bag_size = 100
    feat_dim = 1024
    x = torch.randn(3,100,768)
    y = torch.tensor([0,1,0])
    net = CLAM_Batch()
    logits, Y_prob, Y_hat,result_dict = net(batch=(x,y))     
    print(f"logits: {logits} Y_prob: {Y_prob} Y_hatL {Y_hat}")
