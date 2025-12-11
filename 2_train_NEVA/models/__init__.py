import pytorch_lightning as pl
import torchmetrics.functional as tf
import importlib
import torch.nn.functional as F
import torch
from pytorch_lightning.loggers import WandbLogger
from sksurv.metrics import concordance_index_censored
import os
from models.clam import CLAM_MB, CLAM_SB, CLAM_Batch
from models.neva import NEVA
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.distributed as dist
import numpy as np
import warnings
from typing import Dict, Mapping, Set


def _collect_trainable_names(module: torch.nn.Module) -> Set[str]:
    """Return stateful names (parameters/buffers) that should be persisted."""
    param_names = {name for name, param in module.named_parameters() if param.requires_grad}
    buffer_names = {name for name, buf in module.named_buffers() if getattr(buf, "requires_grad", False)}
    return param_names | buffer_names


def extract_trainable_state_dict(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Extract only trainable entries from a Lightning checkpoint state_dict."""
    trainable_names = _collect_trainable_names(module)
    return {name: tensor for name, tensor in state_dict.items() if name in trainable_names}


def save_trainable_state_dict(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    destination: os.PathLike,
) -> None:
    """Persist a reduced checkpoint containing only trainable tensors.

    Reload with ``module.load_state_dict(torch.load(path), strict=False)`` to
    update adapters and heads without touching frozen backbone weights.
    """
    filtered = extract_trainable_state_dict(module, state_dict)
    torch.save(filtered, destination)

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

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        warnings.warn("AUC is undefined when only one class is present; returning 0.5.", RuntimeWarning)
        return 0.5

    num_classes = probabilities.shape[1]

    if num_classes == 2:
        # Binary classification case
        binary_probabilities = probabilities[:, 1]
        return roc_auc_score(labels, binary_probabilities)

    # Multiclass case
    aucs = []
    for i in unique_labels:
        binary_labels = (labels == i).astype(int)
        class_probabilities = probabilities[:, i]
        aucs.append(roc_auc_score(binary_labels, class_probabilities))
    return sum(aucs) / len(unique_labels)


def calculate_accuracy(logits, labels):
    # Get the predicted class by taking the argmax
    _, predicted = torch.max(logits, 1)
    # Convert predictions and labels to CPU tensors
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    # Calculate accuracy
    acc = accuracy_score(labels, predicted)
    return acc


def aggregate_patient_outputs(case_ids, mean_tensors=None, first_tensors=None):
    """Aggregate slide-level outputs into patient-level tensors.

    Args:
        case_ids: Sequence of case identifiers aligned with batch rows.
        mean_tensors: Mapping of name -> tensor to average across slides.
        first_tensors: Mapping of name -> tensor to keep the first slide entry.

    Returns:
        ordered_case_ids: unique case ids in order of first appearance.
        aggregated: dict containing aggregated tensors for provided names.
    """
    mean_tensors = mean_tensors or {}
    first_tensors = first_tensors or {}

    if not case_ids:
        raise ValueError("case_ids must not be empty.")

    reference_shape = next(iter(mean_tensors.values()), next(iter(first_tensors.values()), None))
    if reference_shape is None:
        raise ValueError("At least one tensor must be provided for aggregation.")

    expected_length = reference_shape.shape[0]
    if any(t.shape[0] != expected_length for t in mean_tensors.values()) or \
       any(t.shape[0] != expected_length for t in first_tensors.values()):
        raise ValueError("All tensors must have the same leading dimension as case_ids.")

    if len(case_ids) != expected_length:
        raise ValueError("Number of case IDs must match tensor batch dimension.")

    id_to_indices = {}
    ordered_ids = []
    for idx, cid in enumerate(case_ids):
        if cid not in id_to_indices:
            id_to_indices[cid] = []
            ordered_ids.append(cid)
        id_to_indices[cid].append(idx)

    aggregated_mean_lists = {name: [] for name in mean_tensors}
    aggregated_first_lists = {name: [] for name in first_tensors}

    index_device = next(iter(mean_tensors.values()), next(iter(first_tensors.values()))).device

    for cid in ordered_ids:
        indices = id_to_indices[cid]
        index_tensor = torch.as_tensor(indices, device=index_device, dtype=torch.long)

        for name, tensor in mean_tensors.items():
            patient_tensor = torch.index_select(tensor, 0, index_tensor)
            aggregated_mean_lists[name].append(patient_tensor.mean(dim=0))

        for name, tensor in first_tensors.items():
            aggregated_first_lists[name].append(tensor[indices[0]])

    aggregated = {}
    for name, values in aggregated_mean_lists.items():
        aggregated[name] = torch.stack(values, dim=0)

    for name, values in aggregated_first_lists.items():
        sample_tensor = first_tensors[name]
        aggregated[name] = torch.stack(values, dim=0).to(device=sample_tensor.device, dtype=sample_tensor.dtype)

    return ordered_ids, aggregated

def _gather_strings_across_processes(strings):
    if not dist.is_available() or not dist.is_initialized():
        return list(strings)
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, list(strings))
    merged = []
    for chunk in gathered:
        if chunk:
            merged.extend(chunk)
    return merged


class MILModel(pl.LightningModule):  
    def __init__(self, config, save_path):
        super().__init__()
        self.config = config
        self.save_path = save_path
            
        self.criterion = get_obj_from_str(
            self.config["Loss"]["name"]
            )(**self.config["Loss"]["params"])

        self.model = get_obj_from_str(config["Model"]["name"])(**config["Model"]["params"])

        self.validation_step_outputs = []   # 用于记录验证集的每个step的输出，最后拼接起来然后计算metric
        self.test_step_outputs = []         #  用于记录test集的每个step的输出，最后拼接起来然后计算metric
        self.test_performance = 0  # save the final test performance of one fold

    @property
    def _is_distributed(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        return getattr(trainer, "world_size", 1) > 1

    def _gather_tensor(self, tensor):
        if not self._is_distributed:
            return tensor
        gathered = self.all_gather(tensor, sync_grads=False)
        if isinstance(gathered, torch.Tensor):
            if gathered.dim() == tensor.dim():
                return gathered
            new_shape = (-1,) + tuple(tensor.shape[1:])
            return gathered.reshape(new_shape)
        return torch.cat(list(gathered), dim=0)

    def _log_epoch_metric_to_wandb(self, metric_name: str) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None or not getattr(trainer, "is_global_zero", True):
            return

        logger_instance = getattr(trainer, "logger", None)
        if not isinstance(logger_instance, WandbLogger):
            return

        metric_value = trainer.callback_metrics.get(metric_name)
        if metric_value is None:
            metric_value = trainer.callback_metrics.get(f"{metric_name}_epoch")
        if metric_value is None:
            return

        if isinstance(metric_value, torch.Tensor):
            if metric_value.numel() == 0:
                return
            if metric_value.numel() > 1:
                metric_value = metric_value.detach().mean()
            metric_value = metric_value.detach().float().item()
        else:
            try:
                metric_value = float(metric_value)
            except (TypeError, ValueError):
                return

        run = getattr(logger_instance, "experiment", None)
        if run is None:
            return

        step = getattr(trainer, "current_epoch", 0)
        try:
            run.log({metric_name: metric_value, "epoch": step}, step=step)
            run.summary[f"last_{metric_name}"] = metric_value
        except Exception:
            return

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):  # -->metrics : metrics = {"loss": loss, "logits": logits, "labels": label}, 包括logits和labels是为了后面的valid和test进行计算
        
        img, report, *others = batch  # 数据集输出包含图像、报告、时间、状态以及case_id

        if self.config['Data']['dataset_name'] == 'coxreg':
            time, status, case_ids = others
            case_ids = list(case_ids)
            if self.config['type']['type']== 'report':  # report_only  文本单模态
                logits = self(report)
                loss = self.criterion(logits, time, status)
                '''
                CoxSurvLoss的三个参数
                第一个参数： 模型输出发生风险的值
                第二个参数： pfs  观察时间
                第三个参数： 事件是否发生
                '''
                metrics = {"loss": loss, "risks": logits, "pfs": time, "status": status,
                           "case_ids": case_ids}

            elif self.config['type']['type']== 'vision':  # vision_only 图像单模态
                logits, Y_prob, Y_hat, results_dict = self((img,status))
                loss = self.criterion(logits, time, status)
                metrics = {"loss": loss, "risks": logits, "pfs": time, "status": status,
                           "case_ids": case_ids}

            elif self.config['type']['type']=='MultiModal' :  # MultiModal 多模态
                logits, results_dict = self((img,report)) 
                loss = self.criterion(logits, time, status)

                # additional loss for vision branch
                if 'logits_vision' in results_dict.keys():
                    loss += 0.3 * self.criterion(results_dict['logits_vision'], time, status)

                # additional loss for language branch
                if 'logits_report' in results_dict.keys():
                    loss += 0.3 * self.criterion(results_dict['logits_report'], time, status)

                metrics = {"loss": loss, "risks": logits, "pfs": time, "status": status,
                           "case_ids": case_ids}


        elif self.config['Data']['dataset_name'] == 'cls':
            _, label, case_ids = others
            case_ids = list(case_ids)
            label = label.long()
            if self.config['type']['type']== 'report':   # report_only  文本单模态
                logits = self(report)
                loss = self.criterion(logits, label)
                metrics = {"loss": loss, "logits": logits, "labels": label,
                           "case_ids": case_ids}

            elif self.config['type']['type']== 'vision':  # vision_only 图像单模态
                logits, Y_prob, Y_hat, results_dict = self((img,label))
                loss = self.criterion(logits, label)
                metrics = {"loss": loss, "logits": logits, "labels": label,
                           "case_ids": case_ids}

            elif self.config['type']['type']=='MultiModal' :  # MultiModal 多模态
                logits, results_dict = self.model((img,report)) 
                loss = self.criterion(logits, label)
                metrics = {"loss": loss, "logits": logits, "labels": label,
                           "case_ids": case_ids}

        else:
            raise NotImplementedError

        return metrics

    def training_step(self, batch):   # 一般训练时是不会记录下来 loss 或者其他metrics
        loss = self.compute_loss(batch)["loss"]   #  只返回损失值，也可以返回一个字典，但是必须有"loss" 这个键
        self.log(
            "train_loss",
            loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss


    def on_train_epoch_end(self,):  # 在每个训练周期结束后执行， 一般就是更新学习率，trian的每个epoch结束的时候也不会记录loss或者其他metrics
        self.lr_scheduler.step()  # 每个周期结束后更新学习率
        self._log_epoch_metric_to_wandb("train_loss")


    def eval_epoch(self, mode='eval'):  # 在验证集和测试集上操作

        step_outputs = self.validation_step_outputs if mode == 'eval' else self.test_step_outputs
        ordered_case_ids = None
        
        # coxreg：验证集会记录 "val_loss"、"val_cindex"   early-stop和modelcheckpoint采用的都是 'val_cindex'
        if self.config['Data']['dataset_name'] == 'coxreg':
                
            # gather all validation results
            risks_list = torch.cat([out["risks"] for out in step_outputs], dim=0)
            pfs_list = torch.cat([out["pfs"] for out in step_outputs], dim=0)
            status_list = torch.cat([out["status"] for out in step_outputs], dim=0)
            case_ids = []
            for out in step_outputs:
                case_ids.extend(out.get("case_ids", []))

            risks_list = self._gather_tensor(risks_list)
            pfs_list = self._gather_tensor(pfs_list)
            status_list = self._gather_tensor(status_list)
            case_ids = _gather_strings_across_processes(case_ids)

            if len(case_ids) != risks_list.shape[0]:
                raise ValueError("Mismatch between number of collected case ids and risk predictions.")

            ordered_case_ids, aggregated = aggregate_patient_outputs(
                case_ids,
                mean_tensors={"risks": risks_list},
                first_tensors={"pfs": pfs_list, "status": status_list}
            )
            risks_list = aggregated["risks"]
            pfs_list = aggregated["pfs"]
            status_list = aggregated["status"]

            eval_loss = self.criterion(risks_list, pfs_list, status_list)

            # self.log("val_loss", eval_loss)
            self.log("val_loss", eval_loss,sync_dist=True,on_epoch=True)

            c_index = round(compute_c_index(risks_list, pfs_list, status_list),4)
            # self.log("val_cindex", c_index)
            self.log("val_cindex", c_index,sync_dist=True)  # 这个是不支持的   sync_dist 不是这么用的


        # cls： 验证集在每个epoch的结束的时候会记录  "val_loss"、"val_acc"、"val_auc"  early-stop和modelcheckpoint采用的都是 "val_auc"
        elif self.config['Data']['dataset_name'] == 'cls':

            # gather all validation results
            logits_list = torch.cat([out["logits"] for out in step_outputs], dim=0)
            label_list = torch.cat([out["labels"] for out in step_outputs], dim=0)
            case_ids = []
            for out in step_outputs:
                case_ids.extend(out.get("case_ids", []))

            logits_list = self._gather_tensor(logits_list)
            label_list = self._gather_tensor(label_list)
            case_ids = _gather_strings_across_processes(case_ids)

            if len(case_ids) != logits_list.shape[0]:
                raise ValueError("Mismatch between number of collected case ids and logits.")

            ordered_case_ids, aggregated = aggregate_patient_outputs(
                case_ids,
                mean_tensors={"logits": logits_list},
                first_tensors={"labels": label_list}
            )
            logits_list = aggregated["logits"]
            label_list = aggregated["labels"]

            eval_loss = self.criterion(logits_list, label_list)  # 其实就是 conpute_loss时候的loss

            self.log("val_loss", eval_loss,sync_dist=True,on_epoch=True)   # 验证集上的所有损失
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
                if self.config['Data']['dataset_name'] == 'coxreg':
                        
                    self.test_performance = c_index
                    torch.save(risks_list, f'{self.save_path}/risk.pt')
                    torch.save(pfs_list, f'{self.save_path}/pfs.pt')
                    torch.save(status_list, f'{self.save_path}/status.pt')
                    torch.save(ordered_case_ids, f'{self.save_path}/case_ids.pt')
                    print('-----'*10)
                    print("prob_list shape", risks_list.shape[0])
                    print(f"performance c_index: {c_index:.4f}, loss: {eval_loss: .4f}")
                
                elif self.config['Data']['dataset_name'] == 'cls':
                    self.test_performance = auc     # 我们最终只使用auc作为测试集的性能指标
                    print('#####'*10)
                    print(f'{self.save_path}/logits.pt')
                    torch.save(logits_list, f'{self.save_path}/logits.pt')
                    torch.save(label_list, f'{self.save_path}/labels.pt')
                    if ordered_case_ids is not None:
                        torch.save(ordered_case_ids, f'{self.save_path}/case_ids.pt')
                    print(f"performance auc: {auc:.4f}, acc: {acc: .4f}")

                else:
                    raise NotImplementedError


    def validation_step(self, batch, batch_idx):  # 一般在validation的step时也不会记录log
        # Compute loss and metrics for the current validation batch
        with torch.inference_mode():  # 是 torch.no_grad() 的优化版本，也是禁用梯度，但速度更快，节省内存且加快速度
            outputs = self.compute_loss(batch)  # 输出一个字典：  metrics = {"loss": loss, "logits": logits, "labels": label}
        self.log(
            "val_loss",
            outputs["loss"],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.validation_step_outputs.append(outputs)  # 在列表中添加验证集时的metrics，metrics是一个字典，即列表中的每个元素就是每个验证step的字典


    def on_validation_epoch_end(self):  # 一般在 validation_epoch_end
        self.eval_epoch(mode='eval')    #  用于记录整个epoch的 loss、cindex、auc、acc等指标
        self._log_epoch_metric_to_wandb("val_loss")
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
