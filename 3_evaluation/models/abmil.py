import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import initialize_weights
###
"""
Ilse, M., Tomczak, J. and Welling, M., 2018, July. 
Attention-based deep multiple instance learning. 
In International conference on machine learning (pp. 2127-2136). PMLR.
"""

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
# 1024 --> 256 --> n_classs=1   两个FC层
class Attn_Net(nn.Module):
    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes, N x 1024


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
# 1024 --> 256 --> n_classs=1   3个FC层
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=384, dropout=False, n_classes=1):
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
        A = a.mul(b)    # a 和 b 逐元素相乘  相当于注意力了
        A = self.attention_c(A)  # N x n_classes
        return A, x   # A:N x n_classes    x: N x 1024


class AbMIL(nn.Module):  # Attention-based Multiple Instance Learning    多实例学习 MIL
    def __init__(self, gate=True, feat_dim=768, dropout=False, n_classes=2):
        super(AbMIL, self).__init__()
        size = [feat_dim, 512, 384]  # [768,512,384]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]  # 768 --> 512  统一将每个instance的向量长度设置为512
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:  # 默认是带 Gate的网络
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1) # 512->384->1
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)  # 512->384->1
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)  # 768 --> 512 -->384 -->1
        self.classifiers = nn.Linear(size[1], n_classes)   # 512 --> 2
        initialize_weights(self)  # 设置 Linear和Batchnorm1D初始化

    def forward(self, h):   # 这个是基于单个Bag的而非基于Batch的，其实CLAM中不考虑instance_eval的话就是Batch版本的abmil

        h = h.squeeze()  # [B,N,D]   # B：batch   N：一个bag中的instance  D：每个instance的特征向量维度
        # print(f"raw h: {h.shape}")
        A, h = self.attention_net(h)  # A:[N,C,1]    h:[N,C,512]
        A = torch.transpose(A, 1, 0)  # [N,1,C]
        A = F.softmax(A, dim=1)  # softmax over N     [1,N]

        M = torch.mm(A, h)   # [1,N]x[N,512]-->[1,512]
        logits = self.classifiers(M)   # [512 --> 2]  #  原始logits值 不需要全为正数，也不需要相加和为1

        Y_hat = torch.topk(logits, 1, dim=1)[1]   # 返回前 K 个最大元素及其索引
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat


if __name__ == "__main__":
    bag_size = 100
    feat_dim = 2048
    x = torch.randn(3,100,768)   # [100,2048]
    net = AbMIL()
    logits, Y_prob, Y_hat = net(h=x)
    print(f"logits: {logits} Y_prob: {Y_prob} Y_hatL {Y_hat}")
