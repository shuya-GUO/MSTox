import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import warnings
import json
import glob
from pathlib import Path
import seaborn as sns
warnings.filterwarnings('ignore')

class ChemicalEncoder(nn.Module):
    def __init__(self, in_dim=5000, hidden=512, out_dim=128, dropout1=0.4, dropout2=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden, out_dim),
            nn.Dropout(dropout2)
        )
    def forward(self, x):
        return self.net(x)


class ConditionEncoder(nn.Module):
    def __init__(self, species_num=6, route_num=2, emb_dim=32, out_dim=128, dropout=0.2):
        super().__init__()
        self.species_emb = nn.Embedding(species_num, emb_dim)
        self.route_emb = nn.Embedding(route_num, 8)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + 8 + emb_dim * 8, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )
    def forward(self, species_id, route_id):
        s = self.species_emb(species_id)
        r = self.route_emb(route_id)
        interaction = torch.einsum('bi,bj->bij', s, r).view(s.size(0), -1)
        cond = torch.cat([s, r, interaction], dim=1)
        return self.mlp(cond)


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim=128, num_heads=2, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    def forward(self, z_chem, z_cond):
        batch_size = z_chem.size(0)
        residual = z_chem
        z_chem = z_chem.unsqueeze(1)
        z_cond = z_cond.unsqueeze(1)
        Q = self.q_proj(z_cond)
        K = self.k_proj(z_chem)
        V = self.v_proj(z_chem)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1)
        out = self.out_proj(out)
        out = self.norm(residual + out)
        return out

class Expert(nn.Module):
    def __init__(self, dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class MMoE(nn.Module):
    def __init__(self, dim=128, num_experts=2, num_tasks=12):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim, dropout=0.2) for _ in range(num_experts)])
        self.gates = nn.ModuleList([nn.Linear(dim, num_experts) for _ in range(num_tasks)])
    def forward(self, x, task_id):
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        outputs = []
        for i in range(len(self.gates)):
            mask = (task_id == i)
            if mask.sum() == 0:
                continue
            gate_w = F.softmax(self.gates[i](x[mask]), dim=1)
            mixed = torch.sum(gate_w.unsqueeze(2) * expert_outs[mask], dim=1)
            outputs.append((mask, mixed))
        if outputs:
            final = torch.zeros_like(x)
            for mask, out in outputs:
                final[mask] = out
            return final
        else:
            return x

class TaskEmbeddingLayer(nn.Module):
    def __init__(self, num_tasks=12, emb_dim=32):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim)
        )
    def forward(self, task_id):
        return self.mlp(self.task_emb(task_id))

class AdaptiveTaskHeads(nn.Module):
    def __init__(self, dim=128, num_tasks=12, dropout=0.2,
                 use_shrinkage=True, use_ordinal=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_shrinkage = use_shrinkage
        self.use_ordinal = use_ordinal
        self.register_buffer('low_performance_tasks', torch.tensor([], dtype=torch.long))
        self.register_buffer('tiny_sample_tasks', torch.tensor([], dtype=torch.long))
        self.normal_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
            for _ in range(num_tasks)
        ])
        if use_shrinkage:
            self.shrinkage_head = nn.Linear(dim, 1)
            self.shrinkage_factor = nn.Parameter(torch.tensor(0.5))
        if use_ordinal:
            self.ordinal_head = nn.Linear(dim, 3)
            self.ordinal_bins = nn.Parameter(torch.tensor([2.0, 3.0]))
    
    def forward(self, x, task_id, return_all=False):
        reg_pred = torch.zeros(x.size(0), 1, device=x.device)
        for i in range(len(self.normal_heads)):
            mask = (task_id == i)
            if mask.sum() == 0:
                continue
            tp = self.normal_heads[i](x[mask])
            reg_pred[mask] = tp
        if self.use_ordinal and return_all:
            return reg_pred, self.ordinal_head(x), self.ordinal_bins.detach()
        return reg_pred


class ImprovedToxicityModel(nn.Module):
    def __init__(self, device='cuda', dropout_chem1=0.4, dropout_chem2=0.3,
                 dropout_cond=0.2, dropout_attn=0.3, dropout_expert=0.2, dropout_head=0.2,
                 num_heads=2, num_experts=2, use_shrinkage=True, use_ordinal=True):
        super().__init__()
        self.chem_encoder = ChemicalEncoder(dropout1=dropout_chem1, dropout2=dropout_chem2)
        self.cond_encoder = ConditionEncoder(dropout=dropout_cond)
        self.cross_attn = CrossAttentionLayer(dim=128, num_heads=num_heads, dropout=dropout_attn)
        self.mmoe = MMoE(num_experts=num_experts)
        self.task_embedding = TaskEmbeddingLayer()
        self.heads = AdaptiveTaskHeads(dropout=dropout_head, use_shrinkage=use_shrinkage,
                                       use_ordinal=use_ordinal)
    
    def forward(self, fp, species_id, route_id, task_id):
        z_chem = self.chem_encoder(fp)
        z_cond = self.cond_encoder(species_id, route_id)
        z = self.cross_attn(z_chem, z_cond)
        z = self.mmoe(z, task_id)
        return self.heads(z, task_id, return_all=True)


INPUT_FP_FILE = os.path.join("data", "ECRFS_fps.csv")
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models"

OUTPUT_DIR = SCRIPT_DIR / "ECRFS_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TASK_NAMES = [
    'mouse_oral', 'mouse_iv',
    'rat_oral', 'rat_iv',
    'rabbit_oral', 'rabbit_iv',
    'dog_oral', 'dog_iv',
    'cat_oral', 'cat_iv',
    'guineapig_oral', 'guineapig_iv'
]
SPECIES_NAMES = ['mouse', 'rat', 'rabbit', 'dog', 'cat', 'guineapig']
ROUTE_NAMES = ['oral', 'iv']

def classify_toxicity(log_ld50):
    if log_ld50 < 2:
        return '高毒'
    elif log_ld50 < 3:
        return '中毒'
    else:
        return '低毒'

fp_df = pd.read_csv(INPUT_FP_FILE)
fp_columns = list(fp_df.columns[1:5001])
fp_data = fp_df[fp_columns].values.astype("float32")
n_samples = fp_data.shape[0]

extra_cols = list(fp_df.columns[0:1])
meta_df = fp_df[extra_cols].copy()

def load_ensemble_models(model_paths, device='cpu'):
    models = []
    for i, path in enumerate(model_paths):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        bp = ckpt.get('best_params', {})
        fr = ckpt.get('fold_results', {})
        model = ImprovedToxicityModel(
            device=device,
            dropout_chem1=bp.get('dropout_chem1', 0.4),
            dropout_chem2=bp.get('dropout_chem2', 0.3),
            dropout_cond=bp.get('dropout_cond', 0.2),
            dropout_attn=bp.get('dropout_attn', 0.3),
            dropout_expert=bp.get('dropout_expert', 0.2),
            dropout_head=bp.get('dropout_head', 0.2),
            num_heads=bp.get('num_heads', 2),
            num_experts=bp.get('num_experts', 2),
            use_shrinkage=True,
            use_ordinal=True
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"  Fold {i+1} | Val R\u00b2: {fr.get('val_r2', 'N/A')}")
    return models

model_paths = sorted(MODEL_DIR.glob("MSTox_optimized_fold*.pth"))
if not model_paths:
    raise FileNotFoundError(
        f"No model files found in {MODEL_DIR}. "
        "Please download the trained model weights into the models/ directory."
    )
model_paths = [str(p) for p in model_paths]

models = load_ensemble_models(model_paths, device=DEVICE)

def predict_all_tasks(fp_data, models, device='cuda', batch_size=128):
    n = fp_data.shape[0]
    results_all = {}
    
    for task_id in range(12):
        species_id = task_id // 2
        route_id = task_id % 2
        task_name = TASK_NAMES[task_id]
        
        species_ids = np.full(n, species_id, dtype=np.int64)
        route_ids = np.full(n, route_id, dtype=np.int64)
        task_ids = np.full(n, task_id, dtype=np.int64)
        
        all_fold_preds = []
        with torch.no_grad():
            for model in models:
                fold_preds = []
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    batch = {
                        'fp': torch.tensor(fp_data[start:end], dtype=torch.float32).to(device),
                        'sp': torch.tensor(species_ids[start:end], dtype=torch.int64).to(device),
                        'rt': torch.tensor(route_ids[start:end], dtype=torch.int64).to(device),
                        'tk': torch.tensor(task_ids[start:end], dtype=torch.int64).to(device),
                    }
                    pred, _, _ = model(fp=batch['fp'], species_id=batch['sp'], route_id=batch['rt'], task_id=batch['tk'])
                    fold_preds.extend(pred.cpu().numpy().flatten())
                all_fold_preds.append(fold_preds)
        
        all_fold_preds = np.array(all_fold_preds)  # (n_folds, n_samples)
        ensemble_log10 = np.mean(all_fold_preds, axis=0)
        ensemble_std = np.std(all_fold_preds, axis=0)
        ld50_mg_kg = 10 ** ensemble_log10
        
        results_all[task_name] = {
            'log10_LD50': ensemble_log10,
            'LD50_mg_kg': ld50_mg_kg,
            'prediction_std': ensemble_std,
            'fold_preds': all_fold_preds,
            'species': SPECIES_NAMES[species_id],
            'route': ROUTE_NAMES[route_id],
            'species_id': species_id,
            'route_id': route_id,
            'task_id': task_id,
        }
        
        print(f"  [{task_id+1:2d}/12] {task_name:15s} | log10: {ensemble_log10.mean():.4f} | LD50: {ld50_mg_kg.mean():.2f} mg/kg")
    
    return results_all

results_all = predict_all_tasks(fp_data, models, device=DEVICE, batch_size=128)

columns = ['compound_index']
for task_name in TASK_NAMES:
    columns += [
        f'{task_name}_log10',
        f'{task_name}_LD50_mg_kg',
        f'{task_name}_std',
        f'{task_name}_toxicity'
    ]

data = {'compound_index': list(range(n_samples))}
for task_name in TASK_NAMES:
    r = results_all[task_name]
    data[f'{task_name}_log10'] = r['log10_LD50']
    data[f'{task_name}_LD50_mg_kg'] = r['LD50_mg_kg']
    data[f'{task_name}_std'] = r['prediction_std']
    data[f'{task_name}_toxicity'] = [classify_toxicity(v) for v in r['log10_LD50']]

main_df = pd.DataFrame(data, columns=columns)
main_csv_path = os.path.join(OUTPUT_DIR, "predictions_all_12_tasks.csv")
main_df.to_csv(main_csv_path, index=False, encoding='utf-8-sig')
