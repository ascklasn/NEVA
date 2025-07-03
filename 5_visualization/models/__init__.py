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

# def calculate_auc(logits, labels):
#     # Convert logits to probabilities using softmax
#     probabilities = torch.softmax(logits, dim=1)[:, 1]
#     # Convert probabilities and labels to CPU tensors
#     probabilities = probabilities.cpu().detach().numpy()
#     labels = labels.cpu().numpy()
#     # Calculate AUC
#     auc = roc_auc_score(labels, probabilities)
#     return auc

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
        auc = roc_auc_score(labels, binary_probabilities)  
    else:
        # Multiclass case
        aucs = []
        unique_labels = np.unique(labels)
        for i in unique_labels:
            binary_labels = (labels == i).astype(int)
            class_probabilities = probabilities[:, i]
            auc = roc_auc_score(binary_labels, class_probabilities)
            aucs.append(auc)
        auc = sum(aucs) / len(unique_labels)  # Macro-average AUC  
    
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
            
        self.criterion = get_obj_from_str(
            self.config["Loss"]["name"]
            )(**self.config["Loss"]["params"])

        self.model = get_obj_from_str(config["Model"]["name"])(**config["Model"]["params"])

        self.validation_step_outputs = []   # Used to record the output of each step of the validation set,
        self.test_step_outputs = []         # Used to record the output of each step of the test set for use in the final epoch_end.
        self.test_performance = 0  # save the final test performance of one fold

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):  # -->metrics : metrics = {"loss": loss, "logits": logits, "labels": label}
        
        img, report, _, _ = batch  #  feat_image, feat_report, time, status / feat_image, feat_report, torch.tensor([0]).float(), label

        
        if self.config['Data']['dataset_name'] == 'coxreg':
            if self.config['type']['type']== 'report':  
                pfs, status = batch[-2:]  # feat_image, feat_report, time, status
                logits = self(report)
                loss = self.criterion(logits, pfs, status)
                '''
                CoxSurvLoss's three parameters
                logits: The risk of model output
                pfs: The observed survival times
                status: The event indicators (1 if event occurred, 0 for censored)
                '''
                metrics = {"loss": loss, "risks": logits, "pfs": pfs, "status": status}


            elif self.config['type']['type']== 'vision':  
                if self.config['type']['use_model'] == 'clam':
                    pfs, status = batch[-2:]  
                    label = batch[-1].long()
                    logits, Y_prob, Y_hat, results_dict = self((img,status))
                    loss = self.criterion(logits, pfs, status)
                    metrics = {"loss": loss, "risks": logits, "pfs": pfs, "status": status}
                elif self.config['type']['use_model'] == 'longvit':
                    pfs, status = batch[-2:]  
                    label = batch[-1].long()
                    logits = self(img)
                    loss = self.criterion(logits, pfs, status)
                    metrics = {"loss": loss, "risks": logits, "pfs": pfs, "status": status}

            elif self.config['type']['type']== 'MultiModal' :  
                pfs, status = batch[-2:]
                label = batch[-1].long()
                logits, results_dict = self((img,report))  
                loss = self.criterion(logits, pfs, status)

                # additional loss for vision branch
                if 'logits_vision' in results_dict.keys():
                    loss += 0.3 * self.criterion(results_dict['logits_vision'], pfs, status)

                # additional loss for language branch
                if 'logits_report' in results_dict.keys():
                    loss += 0.3 * self.criterion(results_dict['logits_report'], pfs, status)

                metrics = {"loss": loss, "risks": logits, "pfs": pfs, "status": status}



        elif self.config['Data']['dataset_name'] == 'cls':
            if self.config['type']['type']== 'report':  
                label = batch[-1]
                logits = self(report)
                loss = self.criterion(logits, label)

                metrics = {"loss": loss, "logits": logits, "labels": label}

            elif self.config['type']['type']== 'vision':  
                if self.config['type']['use_model']=='clam':
                    label = batch[-1].long()
                    logits, Y_prob, Y_hat, results_dict = self((img,label))
                    loss = self.criterion(logits, label)
                    metrics = {"loss": loss, "logits": logits, "labels": label}
                elif self.config['type']['use_model'] == 'longvit':
                    label = batch[-1].long()
                    logits = self(img)
                    loss = self.criterion(logits, label)
                    metrics = {"loss": loss, "logits": logits, "labels": label}

            elif self.config['type']['type']=='MultiModal' :  
                label = batch[-1].long()
                logits, results_dict = self((img,report))  
                loss = self.criterion(logits, label)
                metrics = {"loss": loss, "logits": logits, "labels": label}

        else:
            raise NotImplementedError

        return metrics

    def training_step(self, batch, batch_idx,):   
        return self.compute_loss(batch)["loss"]  


    def on_train_epoch_end(self,):  
        self.lr_scheduler.step()  # Update the learning rate after each cycleearning rate after each cycle


    def eval_epoch(self, mode='eval'):  

        step_outputs = self.validation_step_outputs if mode == 'eval' else self.test_step_outputs
        
        if self.config['Data']['dataset_name'] == 'coxreg':
                
            # gather all validation results
            risks_list = torch.cat([out["risks"] for out in step_outputs], dim=0)
            pfs_list = torch.cat([out["pfs"] for out in step_outputs], dim=0)
            status_list = torch.cat([out["status"] for out in step_outputs], dim=0)

            eval_loss = self.criterion(risks_list, pfs_list, status_list)

            # self.log("val_loss", eval_loss)
            self.log("val_loss", eval_loss,sync_dist=True,on_epoch=True)

            c_index = round(compute_c_index(risks_list, pfs_list, status_list),4)
            # self.log("val_cindex", c_index)
            self.log("val_cindex", c_index,sync_dist=True) 


        elif self.config['Data']['dataset_name'] == 'cls':

            # gather all validation results
            logits_list = torch.cat([out["logits"] for out in step_outputs], dim=0)
            label_list = torch.cat([out["labels"] for out in step_outputs], dim=0)

            eval_loss = self.criterion(logits_list, label_list)  

            self.log("val_loss", eval_loss,sync_dist=True)  
            # self.log("val_loss", eval_loss)

            acc = calculate_accuracy(logits_list, label_list)
            auc = round(calculate_auc(logits_list, label_list),4)

            self.log("val_acc",acc,sync_dist=True)
            # self.log("val_acc", acc)
            self.log("val_auc", auc,sync_dist=True)
           #  self.log("val_auc", auc)

        else: 
            raise NotImplementedError


        if get_rank() == 0:
            if mode == 'test': 
                if self.config['Data']['dataset_name'] == 'coxreg':
                        
                    self.test_performance = c_index
                    torch.save(risks_list, f'{self.save_path}/risk.pt')
                    torch.save(pfs_list, f'{self.save_path}/pfs.pt')
                    torch.save(status_list, f'{self.save_path}/status.pt')
                    print('-----'*10)
                    print("prob_list shape", risks_list.shape[0])
                    print(f"performance c_index: {c_index:.4f}, loss: {eval_loss: .4f}")
                
                elif self.config['Data']['dataset_name'] == 'cls':
                    self.test_performance = auc    
                    print('#####'*10)
                    print(f'{self.save_path}/logits.pt')
                    torch.save(logits_list, f'{self.save_path}/logits.pt')
                    torch.save(label_list, f'{self.save_path}/labels.pt')

                    print(f"performance auc: {auc:.4f}, acc: {acc: .4f}")

                else:
                    raise NotImplementedError


    def validation_step(self, batch, batch_idx): 

        # Compute loss and metrics for the current validation batch
        with torch.inference_mode(): 
            outputs = self.compute_loss(batch) 
        self.validation_step_outputs.append(outputs)  


    def on_validation_epoch_end(self):  
        self.eval_epoch(mode='eval')   
        self.validation_step_outputs.clear()  

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            ret = self.compute_loss(batch)
        self.test_step_outputs.append(ret)
        return ret     

    def on_test_epoch_end(self):
        self.eval_epoch(mode='test')    
        self.test_step_outputs.clear()  


    def configure_optimizers(self):
        conf_optim = self.config["Optimizer"]
        name = conf_optim["optimizer"]["name"]
        optimizer_cls = getattr(torch.optim, name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, conf_optim["lr_scheduler"]["name"])  # CosineAnnealingLR
        # CosineAnnealingLR method to gradually reduce learning rate

        # train only trainable parameters
        optim = optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), **conf_optim["optimizer"]["params"])
        '''
        adam optimizer
        lr: 0.001
        weight_decay: 0.01
        amsgrad: False
        '''

        self.lr_scheduler = scheduler_cls(optim, **conf_optim["lr_scheduler"]["params"])

        return optim

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