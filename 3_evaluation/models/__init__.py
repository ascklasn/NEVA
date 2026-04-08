import pytorch_lightning as pl
import torchmetrics.functional as tf
import importlib
import torch.nn.functional as F
import torch
from sksurv.metrics import concordance_index_censored
# from models.model_utils import get_rank
# from . import model_utils
# from model_utils import get_rank
import os
from models.clam import CLAM_MB, CLAM_SB, CLAM_Batch
from models.mmclassifier import MMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.distributed as dist
import numpy as np

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def calculate_auc(logits, labels):
    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(logits, dim=1)
    # Convert probabilities and labels to CPU tensors
    probabilities = probabilities.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    
    num_classes = probabilities.shape[1]
    
    if num_classes == 2:
        # Binary classification case
        binary_probabilities = probabilities[:, 1]
        auc = roc_auc_score(labels, binary_probabilities)  # scikit-learn中的一个函数
    else:
        # Multiclass case
        aucs = []
        unique_labels = np.unique(labels)
        for i in unique_labels:
            binary_labels = (labels == i).astype(int)
            class_probabilities = probabilities[:, i]
            auc = roc_auc_score(binary_labels, class_probabilities)
            aucs.append(auc)
        auc = sum(aucs) / len(unique_labels)  # Macro-average AUC  取平均
    
    return auc

def calculate_accuracy(logits, labels):
    # Get the predicted class by taking the argmax
    _, predicted = torch.max(logits, 1)
    # Convert predictions and labels to CPU tensors
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    # Calculate accuracy
    acc = accuracy_score(labels, predicted)
    return acc


class MILModel(pl.LightningModule):  
    def __init__(self, config, save_path):
        super().__init__()
        self.config = config
        self.save_path = save_path

        self.criterion = get_obj_from_str(config["Loss"]["name"])(**config['Loss']['params'])

        self.model = get_obj_from_str(config["Model"]["name"])(**config["Model"]["params"])

        self.validation_step_outputs = []   # 用于记录验证集的每个step的输出，最后拼接起来然后计算metric
        self.test_step_outputs = []         #  用于记录test集的每个step的输出，最后拼接起来然后计算metric
        self.test_performance = 0  # save the final test performance of one fold

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):  # -->metrics : metrics = {"loss": loss, "logits": logits, "labels": label}, 包括logits和labels是为了后面的valid和test进行计算
        
        img, report, _, _ = batch  # 因为数据集的输出是 return feat_image, feat_report, time, status
        
        if self.config['Data']['dataset_type'] == 'coxreg':

            logits = self((img,report))
            time, status = batch[-2:] 
            label = batch[-1].long()
            loss = self.criterion(logits, time, status)
            metrics = {"loss": loss, "risks": logits, "time": time, "status": status}

        elif self.config['Data']['dataset_type'] == 'cls':

            logits = self((img,report))
            label = batch[-1].long()
            loss = self.criterion(logits, label)
            metrics = {"loss": loss, "logits": logits, "labels": label}

        else:
            raise NotImplementedError

        return metrics

    def training_step(self, batch):   # 一般训练时是不会记录下来 loss 或者其他metrics
        return self.compute_loss(batch)["loss"]   #  只返回损失值，也可以返回一个字典，但是必须有"loss" 这个键


    def on_train_epoch_end(self,):  # 在每个训练周期结束后执行， 一般就是更新学习率，trian的每个epoch结束的时候也不会记录loss或者其他metrics
        self.lr_scheduler.step()  # 每个周期结束后更新学习率


    def eval_epoch(self, mode='eval'):  # 在验证集和测试集上操作

        step_outputs = self.validation_step_outputs if mode == 'eval' else self.test_step_outputs
        
        # coxreg：验证集会记录 "val_loss"、"val_cindex"   early-stop和modelcheckpoint采用的都是 'val_cindex'
        if self.config['Data']['dataset_type'] == 'coxreg':
                
            # gather all validation results
            risks_list = torch.cat([out["risks"] for out in step_outputs], dim=0)
            time_list = torch.cat([out["time"] for out in step_outputs], dim=0)
            status_list = torch.cat([out["status"] for out in step_outputs], dim=0)

            eval_loss = self.criterion(risks_list, time_list, status_list)

            # self.log("val_loss", eval_loss)
            self.log("val_loss", eval_loss,sync_dist=True,on_epoch=True)

            c_index = round(compute_c_index(risks_list, time_list, status_list),4)
            # self.log("val_cindex", c_index)
            self.log("val_cindex", c_index,sync_dist=True)  # 这个是不支持的   sync_dist 不是这么用的


        # cls： 验证集在每个epoch的结束的时候会记录  "val_loss"、"val_acc"、"val_auc"  early-stop和modelcheckpoint采用的都是 "val_auc"
        elif self.config['Data']['dataset_type'] == 'cls':

            # gather all validation results
            logits_list = torch.cat([out["logits"] for out in step_outputs], dim=0)
            label_list = torch.cat([out["labels"] for out in step_outputs], dim=0)

            eval_loss = self.criterion(logits_list, label_list)  # 其实就是 conpute_loss时候的loss

            self.log("val_loss", eval_loss,sync_dist=True)   # 验证集上的所有损失
            # self.log("val_loss", eval_loss)

            acc = calculate_accuracy(logits_list, label_list)
            auc = round(calculate_auc(logits_list, label_list),4)
            print(f"acc:{acc},auc:{auc}")

            self.log("val_acc",acc,sync_dist=True)
            # self.log("val_acc", acc)
            self.log("val_auc", auc,sync_dist=True)
           #  self.log("val_auc", auc)

        else: 
            raise NotImplementedError


        # 只让一个GPU进行保存，在单机多卡的时候，每个 GPU 上都会执行 on_test_epoch_end，所以通过get_rank()==0,让GPU_0进行保存，GPU_1不进行保存即可
        # if get_rank() == 0:  # 还是传统的Pytorch的用法
        if self.global_rank == 0:  # Lightning官方的用法
            if mode == 'test':  # 用于测试集   测试集来保存
                if self.config['Data']['dataset_type'] == 'coxreg':
                        
                    self.test_performance = c_index
                    torch.save(risks_list, f'{self.save_path}/risk.pt')
                    torch.save(time_list, f'{self.save_path}/pfs.pt')
                    torch.save(status_list, f'{self.save_path}/status.pt')
                    print('-----'*10)
                    print("prob_list shape", risks_list.shape[0])
                    print(f"performance c_index: {c_index:.4f}, loss: {eval_loss: .4f}")
                
                elif self.config['Data']['dataset_type'] == 'cls':
                    self.test_performance = auc     # 我们最终只使用auc作为测试集的性能指标
                    print('#####'*10)
                    print(f'{self.save_path}/logits.pt')
                    torch.save(logits_list, f'{self.save_path}/logits.pt')
                    torch.save(label_list, f'{self.save_path}/labels.pt')
                    print(f"performance auc: {auc:.4f}, acc: {acc: .4f}")

                else:
                    raise NotImplementedError


    def validation_step(self, batch, batch_idx):  # 一般在validation的step时也不会记录log
        # Compute loss and metrics for the current validation batch
        with torch.inference_mode():  # 是 torch.no_grad() 的优化版本，也是禁用梯度，但速度更快，节省内存且加快速度
            outputs = self.compute_loss(batch)  # 输出一个字典：  metrics = {"loss": loss, "logits": logits, "labels": label}
        self.validation_step_outputs.append(outputs)  # 在列表中添加验证集时的metrics，metrics是一个字典，即列表中的每个元素就是每个验证step的字典


    def on_validation_epoch_end(self):  # 一般在 validation_epoch_end
        self.eval_epoch(mode='eval')    #  用于记录整个epoch的 loss、cindex、auc、acc等指标
        self.validation_step_outputs.clear()   #  清空列表，用于下一个epoch

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            ret = self.compute_loss(batch)
        self.test_step_outputs.append(ret)
        return ret     #返回一个字典，字典里包括 key为 “loss”

    def on_test_epoch_end(self):
        self.eval_epoch(mode='test')   
        self.test_step_outputs.clear()   # 一般在epoch_end的时候都会将一些变量清除，我们在这之前已经用self.log记录下来了


    def configure_optimizers(self):
        conf_optim = self.config["Optimizer"]
        name = conf_optim["optimizer"]["name"]
        optimizer_cls = getattr(torch.optim, name)  # Adam
        scheduler_cls = getattr(torch.optim.lr_scheduler, conf_optim["lr_scheduler"]["name"])  # CosineAnnealingLR
        # CosineAnnealingLR  采用余弦退火的方式逐渐降低学习率

        # train only trainable parameters
        optim = optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), **conf_optim["optimizer"]["params"])
        '''
        adam优化器的参数
        lr: 0.001
        weight_decay: 0.01
        amsgrad: False
        '''

        self.lr_scheduler = scheduler_cls(optim, **conf_optim["lr_scheduler"]["params"])
        # 这个学习率是不需要在 configure_optimizers 函数中返回的，他会在on_training_epoch_end的时候更新学习率

        return optim   # 是要返回一个优化器

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def compute_c_index(risks, durations, events):

    cindex = concordance_index_censored(
        events.cpu().bool(), 
        durations.cpu(), 
        risks.squeeze().cpu(), 
        tied_tol=1e-08
        )[0]
    return cindex