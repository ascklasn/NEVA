import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sksurv.metrics import concordance_index_censored
from matplotlib import colormaps
from lifelines.statistics import logrank_test

# 使用bootstrap计算AUC
def bootstrap_auc(y_true: np.ndarray | torch.Tensor | list, 
                  y_prob: np.ndarray | torch.Tensor | list, 
                  n_bootstrap: int = 3000, 
                  alpha: float = 0.05) -> tuple[float, float, float, float, np.ndarray]:
    """
    Compute the 95% confidence interval (CI), mean AUC, standard deviation (std), and AUC values using the bootstrap method for binary or multiclass classification.

    Args:
        y_true (array-like): True labels. Can be a numpy array, torch tensor, or list.
            - For binary classification: shape = [n_samples], elements are 0 or 1.
            - For multiclass classification: shape = [n_samples], elements are integers in the range [0, n_classes-1].
        y_prob (array-like): Predicted probabilities. Can be a numpy array, torch tensor, or list.
            - For binary classification: shape = [n_samples, 2], where the second column represents the probability of the positive class.
            - For multiclass classification: shape = [n_samples, n_classes].
        n_bootstrap (int, optional): Number of bootstrap resampling iterations. Default is 3000.
        alpha (float, optional): Significance level for the confidence interval. Default is 0.05 (corresponding to a 95% CI).    

    NOTES:
        - For binary classification, the AUC is computed directly for the positive class.
        - For multiclass classification:
            - If the number of unique classes in `y_true` equals `n_classes`, the Macro-AUC is computed by averaging the AUC for each class.
            - If the number of unique classes in `y_true` is less than `n_classes`, the AUC is computed only for the existing classes and averaged.

    Returns:
        tuple: A tuple containing auc_mean, ci_lower, ci_upper, auc_std, auc_values:
            - auc_mean (float): Mean AUC for binary classification or mean Macro-AUC for multiclass classification.
            - ci_lower (float): Lower bound of the 95% confidence interval.
            - ci_upper (float): Upper bound of the 95% confidence interval.
            - auc_std (float): Standard deviation of AUC values from bootstrap samples.
            - auc_values (np.ndarray): Array of AUC values from bootstrap samples.

    Raises:
        ValueError: If `y_true` contains only one unique class or if bootstrap resampling does not generate valid samples.

    Examples:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_prob = np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.2, 0.8], [0.3, 0.7]])
        >>> auc_mean, ci_lower, ci_upper, auc_std, auc_values = bootstrap_auc(y_true, y_prob)
        >>> print(f"AUC Mean: {auc_mean}, 95% CI: ({ci_lower}, {ci_upper}), AUC Std: {auc_std}")

    Todo:
        - 添加Recall、F1-score等指标的计算
    """

    def to_numpy(x):
        """Convert input to numpy array if not already."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.array(x) if isinstance(x, list) else x

    y_true = to_numpy(y_true)
    y_prob = to_numpy(y_prob)

    assert len(np.unique(y_true)) >= 2, "y_true must contain at least two unique classes for AUC computation."  # 至少要有两个类别，如果不满足则直接报错并输出‘y_true must contain at least two unique classes for AUC computation.’的错误信息
    assert y_prob.ndim == 2, "y_prob must be a 2D array for AUC computation."  # y_prob必须是二维数组
    assert y_true.shape[0] == y_prob.shape[0], "y_true and y_prob must have the same number of samples."  # y_true和y_prob的样本数必须相同
    assert y_prob.shape[1] >= 2, "y_prob must have at least two columns for binary classification."  # y_prob至少要有两列

    n_samples = len(y_true)
    n_classes = y_prob.shape[1]
    auc_values = []
    existing_classes = np.unique(y_true)

    def compute_binary_auc(indices):
        """Compute AUC for binary classification."""
        return roc_auc_score(y_true[indices], y_prob[indices][:, 1])

    def compute_multiclass_auc(indices, existing_classes=None):
        """Compute Macro-AUC for multiclass classification."""
        if existing_classes is not None:
            # Handle case where len(np.unique(y_true)) < n_classes
            class_aucs = [
                roc_auc_score((y_true[indices] == cls).astype(int), y_prob[indices][:, cls])
                for cls in existing_classes
                if len(np.unique((y_true[indices] == cls).astype(int))) == 2
            ]
            return np.mean(class_aucs) if class_aucs else None
        else:
            # Handle case where len(np.unique(y_true)) == n_classes
            return roc_auc_score(y_true[indices], y_prob[indices], average='macro', multi_class='ovr')

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        auc = None  # Initialize auc to None for each iteration
        if n_classes == 2:
            # Binary classification
            if len(np.unique(y_true[indices])) == 2:
                auc = compute_binary_auc(indices)
        elif n_classes > 2:
            # Multiclass classification
            if len(existing_classes) < n_classes:
                auc = compute_multiclass_auc(indices, existing_classes)
            else:
                auc = compute_multiclass_auc(indices)
        if auc is not None:
            auc_values.append(auc)

    if not auc_values:
        raise ValueError("Bootstrap did not generate valid samples. This may occur if the data contains only one class or is highly imbalanced.")

    auc_values = np.array(auc_values)
    ci_lower = np.percentile(auc_values, 100 * alpha / 2)
    ci_upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
    auc_mean = np.mean(auc_values)
    auc_std = np.std(auc_values)

    return auc_mean, ci_lower, ci_upper, auc_std, auc_values


# 计算C-index,一致性指数 Concidence Index
def compute_c_index(risks, durations, events):

    cindex = concordance_index_censored(
        events.astype(bool), 
        durations, 
        risks.squeeze(), 
        tied_tol=1e-08
        )[0]
    return cindex

def bootstrap_cindex(risks: np.ndarray | torch.Tensor | list,
                        pfs: np.ndarray | torch.Tensor | list, 
                        status: np.ndarray | torch.Tensor | list, 
                        n_bootstrap: int = 3000,
                        alpha: float = 0.05) -> tuple[float, float, float, float]:
        """
        Compute the 95% CI, Cindex_mean and Cindex_std using the Bootstrap method.
    
        Parameters:
            risks (array-like): Predicted risks. Can be a numpy array, torch tensor, or list.
            pfs (array-like): True survival times. Can be a numpy array, torch tensor, or list.
            status (array-like): Event indicators (1 if event occurred, 0 otherwise). Can be a numpy array, torch tensor, or list.
            n_bootstrap (int, optional): Number of bootstrap resampling iterations. Default is 3000.
    
        Returns:
            tuple: A tuple containing:
                - cindex_mean (float): Mean C-index from bootstrap samples.
                - ci_lower (float): Lower bound of the 95% confidence interval.
                - ci_upper (float): Upper bound of the 95% confidence interval.
                - cindex_std (float): Standard deviation of C-index values from bootstrap samples.
        """
        # Convert inputs to numpy arrays if they are not already
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy() # 将tensor先切换到cpu上然后转成np.ndarray
            return np.array(x) if isinstance(x, list) else x
    
        risks = to_numpy(risks)
        pfs = to_numpy(pfs)
        status = to_numpy(status)
    
        cindex_values = []
        
        # Bootstrap resampling
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(risks), size=len(risks), replace=True)
            if len(np.unique(status[indices])) > 1:
                cindex = concordance_index_censored(status[indices], pfs[indices], risks[indices])[0]
                cindex_values.append(cindex)
        
        if len(cindex_values) == 0:
            raise ValueError("Bootstrap did not generate valid samples. This may occur if the data contains only one class or is highly imbalanced.")
        
        cindex_values = np.array(cindex_values)
    
                                 
        # Calculate 95% CI and mean, standard deviation
        ci_lower = np.percentile(cindex_values, 100 * alpha / 2)  # Lower bound of CI
        ci_upper = np.percentile(cindex_values, 100 * (1 - alpha / 2))  # Upper bound of CI
        cindex_mean = np.mean(cindex_values)  # Mean C-index
        cindex_std = np.std(cindex_values)  # Standard deviation of C-index values
        return  cindex_mean, ci_lower, ci_upper, cindex_std, cindex_values

# todo 有待完善
def bootstrap_f1score(y_true: np.ndarray | torch.Tensor | list, 
                      y_proba: np.ndarray | torch.Tensor | list, 
                      n_bootstrap: int = 1000, 
                      alpha: float = 0.05) -> tuple[float, float, float, float]:
    """
    Compute the 95% CI, f1score_mean and f1score_std using the Bootstrap method for multi-class classification.
    The input `y_pred` contains predicted probabilities, not labels.

    Parameters:
        y_true (array-like): True labels. Can be a numpy array, torch tensor, or list.
        y_proba (array-like): Predicted probabilities. Can be a numpy array, torch tensor, or list.
        n_bootstrap (int, optional): Number of bootstrap resampling iterations. Default is 5000.
        alpha (float, optional): Significance level for confidence interval. Default is 0.05 (corresponding to 95% CI).

    Returns:
        tuple: A tuple containing:
            - f1score_mean (float): Mean F1-score from bootstrap samples.
            - ci_lower (float): Lower bound of the 95% confidence interval.
            - ci_upper (float): Upper bound of the 95% confidence interval.
            - f1score_std (float): Standard deviation of F1-scores from bootstrap samples.
    """
    # Convert inputs to numpy arrays if they are not already
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()  # 将tensor先切换到cpu上然后转成np.ndarray
        return np.array(x) if isinstance(x, list) else x

    y_true = to_numpy(y_true)
    y_proba = to_numpy(y_proba)

    # Convert predicted probabilities to predicted labels by taking the argmax
    y_pred = np.argmax(y_proba, axis=1)

    f1_scores = []
    n_samples = len(y_true)
    
    # Bootstrap resampling
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) > 1:  # Ensure that both classes are represented in the sample
            f1 = f1_score(y_true[indices], y_pred[indices], average='macro')
            f1_scores.append(f1)
    
    if len(f1_scores) == 0:
        raise ValueError("Bootstrap did not generate valid samples. This may occur if the data contains only one class or is highly imbalanced.")
    
    f1_scores = np.array(f1_scores)
    
    # Calculate 95% CI and mean, standard deviation
    ci_lower = np.percentile(f1_scores, 100 * alpha / 2)  # Lower bound of CI
    ci_upper = np.percentile(f1_scores, 100 * (1 - alpha / 2))  # Upper bound of CI
    f1score_mean = np.mean(f1_scores)  # Mean F1-score
    f1score_std = np.std(f1_scores)  # Standard deviation of F1-scores
    
    return f1score_mean, ci_lower, ci_upper, f1score_std

# todo 有待完善
def evaluate_binary(y_labels, y_probs, return_detail_metric=False): 
    """
    Evaluate binary classification tasks by calculating the mean AUC, 95% confidence interval (CI), and AUC standard deviation (std). 
    Optionally, compute additional metrics such as precision, recall, and F1-score.

    Parameters:
        y_labels (array-like): True labels. Can be a numpy array, torch tensor, or list.
            - Shape: [n_samples].
        y_probs (array-like): Predicted probabilities. Can be a numpy array, torch tensor, or list.
            - Shape: [n_samples, n_classes].
        return_detail_metric (bool, optional): Whether to return additional metrics (precision, recall, F1-score). Default is False.

    Returns:
        tuple: A tuple containing:
            - auc_mean (float): Mean AUC from bootstrap samples.
            - ci_lower (float): Lower bound of the 95% confidence interval.
            - ci_upper (float): Upper bound of the 95% confidence interval.
            - auc_std (float): Standard deviation of AUC from bootstrap samples.
            - auc_values (np.ndarray): Array of AUC values from bootstrap samples.
            - detail_metrics : A dictionary containing additional metrics (precision, recall, F1-score) if return_detail_metric is True.
    """

    n_samples = y_probs.shape[0]
    auc_mean, ci_lower, ci_upper, auc_std, auc_values = bootstrap_auc(y_labels, y_probs) # compute 95% CI and std_auc

    # Compute the False Positive Rate (FPR) and True Positive Rate (TPR) for the ROC curve
    binary_prob = y_probs[:, 1]  # Assuming binary classification, take the probability of the positive class
    fpr, tpr, _ = roc_curve(y_labels, binary_prob)
    
    # todo: 决定这个precision recall f1 是否也需要进行bootstrap,这个 if 有待修改
    if return_detail_metric:
        # Compute precision, recall, and F1-score for the binary classification
        precision = precision_score(y_labels, (binary_prob > 0.5).astype(int))  # Precision: TP / (TP + FP)
        recall = recall_score(y_labels, (binary_prob > 0.5).astype(int))        # Recall: TP / (TP + FN)
        f1 = f1_score(y_labels, (binary_prob > 0.5).astype(int))                # F1-score: Harmonic mean of precision and recall 2*(precision*recall)/(precision+recall)
        return auc_mean, ci_lower, ci_upper, auc_std, auc_values, precision, recall, f1, fpr, tpr
    else:
        # If not returning detailed metrics, just return the AUC and CI
        return auc_mean, ci_lower, ci_upper, auc_std, auc_values

# todo 有待完善
def evaluate_multiclass(y_labels, y_probs):
    """
    评估三分类任务，计算宏平均 macro auc_mean和95% CI、auc_std
    Parameters:
        y_labels (array-like): True labels. Can be a numpy array, torch tensor, or list.
        y_probs (array-like): Predicted probabilities. shape：[n_sample, n_classes]. Can be a numpy array, torch tensor, or list.
    
    Returns:
        tuple: A tuple containing:
            - auc_mean (float): Mean Macro-AUC from bootstrap samples.
            - ci_lower (float): Lower bound of the 95% confidence interval.
            - ci_upper (float): Upper bound of the 95% confidence interval.
            - auc_std (float): Standard deviation of AUC from bootstrap samples.
            - auc_values (np.ndarray): Array of AUC values from bootstrap samples.
    """


    # num_classes = y_probs.shape[1]
    # n_samples = y_probs.shape[0]
    macro_auc_mean, ci_lower, ci_upper, macro_auc_std, auc_values = bootstrap_auc(y_labels, y_probs) # compute 95% CI and std_auc

    # todo 是否在多分类里也计算precision recall f1 是否也需要进行bootstrap, 有待完善

    # print(f"多分类任务: F1-score={f1score_mean:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
    return macro_auc_mean, ci_lower, ci_upper, macro_auc_std, auc_values


# todo 可以将计算K-M曲线的P值和HR(95%CI)的函数合并成一个函数
# 计算K-M曲线的logrank test的P-value
def compute_P_value(hazard_scores: np.ndarray | torch.Tensor | list, 
                    labels: np.ndarray | torch.Tensor | list, 
                    status: np.ndarray | torch.Tensor | list) -> np.ndarray:
    """
    Compute the log-rank test p-value between high-risk and low-risk groups based on hazard scores.

    This function is typically used when plotting Kaplan-Meier (K-M) survival curves, to evaluate 
    whether there is a statistically significant difference in survival between two groups divided 
    by risk level. The input `hazard_scores` are divided by their median value into high-risk and 
    low-risk groups. Then, a log-rank test is applied to compare the survival distributions of the two groups.

    Parameters:
        hazard_scores (array-like): Predicted continuous risk scores from a survival model (e.g., Cox model or neural network).
            - Shape: (n_samples,)
            - Type: NumPy array, PyTorch tensor, or list
        labels (array-like): Observed survival times (time to event or censoring).
            - Shape: (n_samples,)
            - Type: NumPy array, PyTorch tensor, or list
        status (array-like): Binary event indicators.
            - 1 indicates the event occurred (e.g., death, relapse)
            - 0 indicates censored data
            - Shape: (n_samples,)
            - Type: NumPy array, PyTorch tensor, or list

    Notes:
        - The high-risk and low-risk groups are determined by the median value of `hazard_scores`.
        - This function uses `lifelines.statistics.logrank_test` internally.
        - A significant result supports the discriminative ability of the risk prediction model.

    Returns:
        float: The p-value from the log-rank test. A small p-value (typically < 0.05) indicates a 
        statistically significant difference in survival between the high-risk and low-risk groups.


    """
    
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy() # 将tensor先切换到cpu上然后转成np.ndarray
        return np.array(x) if isinstance(x, list) else x

    all_hazard_scores = to_numpy(hazard_scores)
    all_labels = to_numpy(labels)
    all_status = to_numpy(status)
    assert len(all_hazard_scores) == len(all_labels) == len(all_status)

    # 按 all_hazard_ratios的中位数 划分高低风险组
    median_risk = np.median(all_hazard_scores)
    high_risk_mask = all_hazard_scores > median_risk
    low_risk_mask = ~high_risk_mask  # 取反，表示低风险

    # 获取高、低风险组的 HR
    high_risk_HR = all_hazard_scores[high_risk_mask]
    low_risk_HR = all_hazard_scores[low_risk_mask]

    # 获取高风险组的索引
    high_risk_indices = np.where(high_risk_mask)[0]
    low_risk_indices = np.where(low_risk_mask)[0]

    # 按索引获取对应的数据
    high_risk_labels = all_labels[high_risk_indices]
    high_risk_status = all_status[high_risk_indices]

    low_risk_labels = all_labels[low_risk_indices]
    low_risk_status = all_status[low_risk_indices]


    # 4. 计算 p_value  
    results = logrank_test(
        durations_A=high_risk_labels,
        durations_B=low_risk_labels,
        event_observed_A=high_risk_status,
        event_observed_B=low_risk_status,
    )
    p_value = results.p_value
    return p_value




