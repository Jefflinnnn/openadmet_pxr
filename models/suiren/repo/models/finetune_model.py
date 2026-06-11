"""
Molecular Property Prediction Model using Suiren Pre-trained and Fine-tuned GNN

This module implements a transfer learning approach for molecular property prediction
by combining pre-trained GNN embeddings with fine-tuned layers. The architecture
supports both regression and classification tasks.

Author: JunyiAn
Date: 2026-02-28
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
from models.graph_NN import GNN
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter, softmax as pyg_softmax


class PredictModel2D(torch.nn.Module):
    """
    2D Molecular Property Predictor with Transfer Learning
    
    This model employs a two-stage approach:
    1. Pre-trained GNN: Extracts general molecular representations
    2. Fine-tuned GNN: Refines representations for specific property prediction
    
    The overall pipeline:
    - Extract node embeddings from pre-trained model (all layers)
    - Process through fine-tuned model with pre-trained embedding conditioning
    - Project node embeddings to latent space
    - Aggregate to molecular level (graph pooling)
    - Generate final prediction (regression or classification)
    
    Args:
        pretrain_num_layer (int): Number of layers in pre-trained model
        finetune_num_layer (int): Number of layers in fine-tuned model
        pretrain_embed_dim (int): Embedding dimension of pre-trained model
        finetune_embed_dim (int): Embedding dimension of fine-tuned model
        drop_ratio (float): Dropout ratio for regularization (default: 0.1)
        d_proj (int): Projection dimension for latent space (default: 256)
        class_num (int): Number of classes for classification (default: 2)
        class_flag (bool): Whether to perform classification (True) or regression (False)
                          (default: False)
    
    Attributes:
        pretrain_model (GNN): Pre-trained GNN model (returns all layer outputs)
        finetune_model (GNN): Fine-tune GNN model
        proj_2d (Sequential): Projection network for node embeddings to latent space
        proj_2d_glob (Sequential): Global predictor from molecular representation
    """
    
    def __init__(self,
                 pretrain_num_layer,
                 finetune_num_layer,
                 pretrain_embed_dim,
                 finetune_embed_dim,
                 drop_ratio=0.1,
                 d_proj=256,
                 class_num=2,
                 class_flag=False,
                 moe_head=False,
                 global_feat_dim=0,
                 heteroscedastic=False,
                 attn_pool=False):
        
        super().__init__()
        self.heteroscedastic = heteroscedastic

        # ========================================================================
        # Pre-trained and Fine-tuned Models
        # ========================================================================
        # Pre-trained model extracts embeddings from all layers
        self.pretrain_model = GNN(
            num_layer=pretrain_num_layer,
            emb_dim=pretrain_embed_dim,
            drop_ratio=0.0,
            output_type="layers"  # Return outputs from all layers
        )
        
        # Fine-tuned model refines representations with conditioning from pre-trained model
        self.finetune_model = GNN(
            num_layer=finetune_num_layer,
            emb_dim=finetune_embed_dim,
            drop_ratio=drop_ratio,
            model_mode="finetune",
            pretrain_emb_dim=pretrain_embed_dim,
            pretrain_num_layer=pretrain_num_layer
        )
        
        # ========================================================================
        # Projection Networks
        # ========================================================================
        # Node embedding projection to latent space
        self.proj_2d = nn.Sequential(
            nn.Linear(finetune_embed_dim, d_proj),
            nn.SiLU(),
            nn.Linear(d_proj, d_proj),
            nn.SiLU(),
            nn.Linear(d_proj, d_proj),
        )

        # Attention pooling: learned per-node importance weights
        self.attn_pool = attn_pool
        if attn_pool:
            self.attn_gate = nn.Sequential(
                nn.Linear(d_proj, d_proj // 4),
                nn.Tanh(),
                nn.Linear(d_proj // 4, 1),
            )

        # Global feature projection (e.g., Uni-Mol2 embeddings)
        self.global_feat_dim = global_feat_dim
        if global_feat_dim > 0:
            self.global_feat_proj = nn.Sequential(
                nn.Linear(global_feat_dim, d_proj),
                nn.SiLU(),
            )
            head_input_dim = d_proj * 2
        else:
            self.global_feat_proj = None
            head_input_dim = d_proj

        # Global prediction head (molecular level)
        self.moe_head = moe_head
        if class_flag:
            self.proj_2d_glob = nn.Sequential(
                nn.Linear(head_input_dim, d_proj),
                nn.SiLU(),
                nn.Linear(d_proj, class_num)
            )
        elif moe_head:
            self.moe_gate = nn.Sequential(
                nn.Linear(head_input_dim, d_proj // 2),
                nn.SiLU(),
                nn.Linear(d_proj // 2, 1),
                nn.Sigmoid()
            )
            self.head_inactive = nn.Sequential(
                nn.Linear(head_input_dim, d_proj),
                nn.SiLU(),
                nn.Linear(d_proj, 1)
            )
            self.head_active = nn.Sequential(
                nn.Linear(head_input_dim, d_proj),
                nn.SiLU(),
                nn.Linear(d_proj, 1)
            )
            self.proj_2d_glob = None
        else:
            out_dim = 2 if heteroscedastic else 1
            self.proj_2d_glob = nn.Sequential(
                nn.Linear(head_input_dim, d_proj),
                nn.SiLU(),
                nn.Linear(d_proj, out_dim)
            )

        # MMP delta head (auxiliary, training only)
        self.delta_head = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.SiLU(),
            nn.Linear(d_proj // 2, 1)
        )

        # Multi-task auxiliary heads (log2fc targets, training only)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, d_proj // 2),
                nn.SiLU(),
                nn.Linear(d_proj // 2, 1),
            )
            for _ in range(2)
        ])

    def _pretrain_forward(self, data):
        """Run pretrain model, skipping gradient tracking when backbone is frozen."""
        first_param = next(self.pretrain_model.parameters())
        ctx = nullcontext() if first_param.requires_grad else torch.no_grad()
        with ctx:
            return self.pretrain_model(
                node_atom=data.x,
                edge_index=data.edge_index,
                edge_index_all=data.edge_index_all,
                edge_attr=data.edge_attr,
                batch=data.batch
            )

    def _pool(self, data):
        """Compute the 256-dim molecular embedding from raw graph data."""
        reference_2d = self._pretrain_forward(data)
        outputs_2d = self.finetune_model(
            node_atom=data.x,
            edge_index=data.edge_index,
            edge_index_all=data.edge_index_all,
            edge_attr=data.edge_attr,
            batch=data.batch,
            extra_embedding=reference_2d
        )
        outputs_2d = self.proj_2d(outputs_2d)
        if self.attn_pool:
            attn_logits = self.attn_gate(outputs_2d).squeeze(-1)
            attn_weights = pyg_softmax(attn_logits, data.batch).unsqueeze(-1)
            weighted = outputs_2d * attn_weights
            return scatter(weighted, data.batch, dim=0, reduce='sum')
        return global_mean_pool(outputs_2d, data.batch)

    def _apply_head(self, pooled, data):
        """Apply global-feat projection + prediction head."""
        if self.global_feat_proj is not None and hasattr(data, 'global_feat'):
            gf = self.global_feat_proj(data.global_feat)
            combined = torch.cat([pooled, gf], dim=-1)
        else:
            combined = pooled
        if self.moe_head:
            gate = self.moe_gate(combined)
            return gate * self.head_active(combined) + (1 - gate) * self.head_inactive(combined)
        return self.proj_2d_glob(combined)

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data: PyTorch Geometric Data object with attributes:
                - x (Tensor): Node features
                - edge_index (LongTensor): Local graph edges
                - edge_index_all (LongTensor): Full-connect graph edges
                - edge_attr (Tensor): Edge attributes
                - batch (LongTensor): Batch assignment for nodes

        Returns:
            Tensor: Molecular level predictions
                - Shape [batch_size, class_num] for classification
                - Shape [batch_size, 1] for regression
        """
        pooled = self._pool(data)
        return self._apply_head(pooled, data)

    def forward_embedding(self, data):
        """Return the 256-dim molecular embedding (before the prediction head)."""
        return self._pool(data)

    def forward_with_embedding(self, data):
        """Return both the prediction and the molecular embedding."""
        pooled = self._pool(data)
        pred = self._apply_head(pooled, data)
        return pred, pooled

    def forward_delta(self, emb_a, emb_b):
        """Predict ΔpEC50 from the difference of two molecular embeddings."""
        return self.delta_head(emb_a - emb_b)

    def forward_aux(self, data):
        """Return pEC50 prediction and list of auxiliary head predictions."""
        pooled = self._pool(data)
        if self.global_feat_proj is not None and hasattr(data, 'global_feat'):
            gf = self.global_feat_proj(data.global_feat)
            combined = torch.cat([pooled, gf], dim=-1)
        else:
            combined = pooled
        primary = self.proj_2d_glob(combined)
        aux = [head(combined) for head in self.aux_heads]
        return primary, aux


def standard_finetune(class_num=2, class_flag=False, moe_head=False, global_feat_dim=0, heteroscedastic=False, attn_pool=False):
    return PredictModel2D(
        pretrain_num_layer=12,
        finetune_num_layer=16,
        pretrain_embed_dim=256,
        finetune_embed_dim=256,
        drop_ratio=0.1,
        d_proj=256,
        class_num=class_num,
        class_flag=class_flag,
        moe_head=moe_head,
        global_feat_dim=global_feat_dim,
        heteroscedastic=heteroscedastic,
        attn_pool=attn_pool,
    )
