"""
稳定的损失函数模块

提供多种数值稳定的损失函数，避免NaN问题
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class StableLossFunctions:
    """
    稳定的损失函数类
    
    提供多种损失函数实现，避免NaN问题：
    1. MSE Loss: 基本的L2距离
    2. Cosine Loss: 余弦相似度（带保护）
    3. KL Divergence: KL散度（用于分布对齐）
    4. Huber Loss: 结合MSE和MAE的优点
    5. Combined Loss: 多种损失的组合
    """
    
    def __init__(self, loss_type: str = "stable_similarity"):
        """
        初始化损失函数
        
        Args:
            loss_type: 损失函数类型
                - "stable_similarity": 组合损失（推荐）
                - "mse_only": 纯MSE损失
                - "cosine_only": 安全余弦相似度
                - "kl_only": KL散度损失
                - "huber": Huber损失
        """
        self.loss_type = loss_type
    
    def compute_vision_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 vision embedding 损失
        
        Args:
            predicted: [batch_size, num_tokens, hidden_dim] 预测的 embeddings
            target: [batch_size, num_tokens, hidden_dim] 目标 embeddings
            
        Returns:
            loss: 标量损失值
        """
        if self.loss_type == 'stable_similarity':
            return self._compute_stable_similarity_loss(predicted, target)
        elif self.loss_type == 'mse_only':
            return F.mse_loss(predicted, target)
        elif self.loss_type == 'cosine_only':
            return self._compute_safe_cosine_loss(predicted, target)
        elif self.loss_type == 'kl_only':
            return self._compute_kl_divergence_loss(predicted, target)
        elif self.loss_type == 'huber':
            return self._compute_huber_loss(predicted, target)
        else:
            return self._compute_stable_similarity_loss(predicted, target)
    
    def _compute_stable_similarity_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算稳定的向量相似度损失
        
        使用多种损失函数的组合，避免NaN问题：
        1. MSE Loss: 基本的L2距离
        2. Cosine Loss: 余弦相似度（带保护）
        3. KL Divergence: KL散度（用于分布对齐）
        
        Args:
            predicted: [batch_size, num_tokens, hidden_dim] 预测的 embeddings
            target: [batch_size, num_tokens, hidden_dim] 目标 embeddings
            
        Returns:
            loss: 标量损失值
        """
        # 1. MSE Loss (L2距离) - 最稳定的基础损失
        mse_loss = F.mse_loss(predicted, target)
        
        # 2. 安全的余弦相似度损失
        cosine_loss = self._compute_safe_cosine_loss(predicted, target)
        
        # 3. KL散度损失（将embedding视为概率分布）
        kl_loss = self._compute_kl_divergence_loss(predicted, target)
        
        # 4. 组合损失：加权平均
        # MSE提供稳定性，余弦相似度提供方向信息，KL散度提供分布对齐
        total_loss = 0.5 * mse_loss + 0.3 * cosine_loss + 0.2 * kl_loss
        
        return total_loss
    
    def _compute_safe_cosine_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        安全的余弦相似度损失计算
        
        使用以下策略避免NaN：
        1. 检查向量范数
        2. 使用安全的归一化
        3. 限制余弦相似度范围
        4. 提供回退机制
        """
        # 计算向量范数
        pred_norm = torch.norm(predicted, dim=-1, keepdim=True)
        target_norm = torch.norm(target, dim=-1, keepdim=True)
        
        # 检查是否有零向量或接近零的向量
        min_norm_threshold = 1e-6
        zero_pred_mask = pred_norm < min_norm_threshold
        zero_target_mask = target_norm < min_norm_threshold
        
        if zero_pred_mask.any() or zero_target_mask.any():
            # 如果有零向量，返回MSE损失
            return F.mse_loss(predicted, target)
        
        # 安全的归一化
        pred_normalized = predicted / (pred_norm + 1e-8)
        target_normalized = target / (target_norm + 1e-8)
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(pred_normalized, target_normalized, dim=-1)
        
        # 限制余弦相似度范围，避免数值不稳定
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # 计算损失：1 - cosine_similarity
        cosine_loss = (1 - cos_sim).mean()
        
        # 检查结果是否有效
        if torch.isnan(cosine_loss) or torch.isinf(cosine_loss):
            return F.mse_loss(predicted, target)
        
        return cosine_loss
    
    def _compute_kl_divergence_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算KL散度损失
        
        将embedding向量转换为概率分布，然后计算KL散度
        这种方法对向量的大小不敏感，只关注方向
        """
        # 将embedding转换为概率分布（使用softmax）
        pred_probs = F.softmax(predicted, dim=-1)
        target_probs = F.softmax(target, dim=-1)
        
        # 计算KL散度：KL(target || predicted)
        # 使用log_softmax和nll_loss的组合，数值更稳定
        log_pred_probs = F.log_softmax(predicted, dim=-1)
        
        # KL散度 = sum(target_probs * log(target_probs / pred_probs))
        # = sum(target_probs * log(target_probs)) - sum(target_probs * log(pred_probs))
        kl_loss = F.kl_div(log_pred_probs, target_probs, reduction='batchmean')
        
        # 检查结果是否有效
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            return F.mse_loss(predicted, target)
        
        return kl_loss
    
    def _compute_huber_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Huber损失
        
        Huber损失结合了MSE和MAE的优点：
        - 对小误差使用MSE（平滑梯度）
        - 对大误差使用MAE（对异常值鲁棒）
        - 数值稳定，不会出现NaN
        """
        return F.huber_loss(predicted, target, delta=1.0)
    
    def compute_weighted_loss(
        self, 
        vision_loss: torch.Tensor, 
        lm_loss: Optional[torch.Tensor],
        vision_loss_weight: float = 1.0,
        lm_loss_weight: float = 1.0,
        use_uncertainty_weighting: bool = False,
        log_var_vision: Optional[torch.Tensor] = None,
        log_var_lm: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算加权损失
        
        Args:
            vision_loss: vision 损失
            lm_loss: language modeling 损失（可选）
            vision_loss_weight: vision损失权重
            lm_loss_weight: lm损失权重
            use_uncertainty_weighting: 是否使用不确定性加权
            log_var_vision: vision损失的不确定性参数
            log_var_lm: lm损失的不确定性参数
            
        Returns:
            total_loss: 总损失
            loss_info: 损失详细信息字典
        """
        loss_info = {}
        
        if use_uncertainty_weighting and log_var_vision is not None and log_var_lm is not None:
            # 限制log_var的范围，避免数值不稳定
            log_var_vision_clamped = torch.clamp(log_var_vision, -10.0, 10.0)
            log_var_lm_clamped = torch.clamp(log_var_lm, -10.0, 10.0)
            
            # 不确定性加权（Kendall et al. 2018）
            # L = (1/2σ₁²) * L_vision + (1/2σ₂²) * L_lm + log(σ₁) + log(σ₂)
            precision_vision = torch.exp(-log_var_vision_clamped)  # 1/σ₁²
            precision_lm = torch.exp(-log_var_lm_clamped)          # 1/σ₂²
            
            weighted_vision_loss = 0.5 * precision_vision * vision_loss + 0.5 * log_var_vision_clamped
            
            if lm_loss is not None:
                weighted_lm_loss = 0.5 * precision_lm * lm_loss + 0.5 * log_var_lm_clamped
                total_loss = weighted_vision_loss + weighted_lm_loss
                
                loss_info = {
                    "sigma_vision": torch.exp(0.5 * log_var_vision_clamped).item(),
                    "sigma_lm": torch.exp(0.5 * log_var_lm_clamped).item(),
                    "precision_vision": precision_vision.item(),
                    "precision_lm": precision_lm.item(),
                }
            else:
                total_loss = weighted_vision_loss
                loss_info = {
                    "sigma_vision": torch.exp(0.5 * log_var_vision_clamped).item(),
                }
        else:
            if lm_loss is not None:
                total_loss = vision_loss_weight * vision_loss + lm_loss_weight * lm_loss
            else:
                total_loss = vision_loss_weight * vision_loss
        
        return total_loss


# 便捷函数
def compute_stable_vision_loss(predicted: torch.Tensor, target: torch.Tensor, loss_type: str = "stable_similarity") -> torch.Tensor:
    """
    便捷函数：计算稳定的vision损失
    
    Args:
        predicted: 预测的embeddings
        target: 目标embeddings
        loss_type: 损失函数类型
        
    Returns:
        损失值
    """
    loss_fn = StableLossFunctions(loss_type)
    return loss_fn.compute_vision_loss(predicted, target)


def compute_weighted_loss(
    vision_loss: torch.Tensor, 
    lm_loss: Optional[torch.Tensor],
    vision_loss_weight: float = 1.0,
    lm_loss_weight: float = 1.0,
    use_uncertainty_weighting: bool = False,
    log_var_vision: Optional[torch.Tensor] = None,
    log_var_lm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    便捷函数：计算加权损失
    
    Args:
        vision_loss: vision损失
        lm_loss: lm损失
        vision_loss_weight: vision损失权重
        lm_loss_weight: lm损失权重
        use_uncertainty_weighting: 是否使用不确定性加权
        log_var_vision: vision损失的不确定性参数
        log_var_lm: lm损失的不确定性参数
        
    Returns:
        总损失和损失信息
    """
    loss_fn = StableLossFunctions()
    return loss_fn.compute_weighted_loss(
        vision_loss, lm_loss, vision_loss_weight, lm_loss_weight,
        use_uncertainty_weighting, log_var_vision, log_var_lm
    )