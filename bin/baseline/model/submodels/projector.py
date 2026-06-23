import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path):
    folder = model_path
    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights

def load_mm_a_projector(model_path, cache_dir=None, token=None):
    folder = model_path
    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector_a.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights

def load_qformer_projector(model_path, cache_dir=None, token=None):
    folder = model_path
    qformer_projector_weights = torch.load(os.path.join(folder, 'qformer.bin'), map_location='cpu')
    qformer_projector_weights = {k: v.to(torch.float16) for k, v in qformer_projector_weights.items()}
    return qformer_projector_weights

class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_audio_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_a_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size_a, config.mm_hidden_size_a)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.mm_hidden_size_a, config.mm_hidden_size_a))
        return nn.Sequential(*modules)
    if projector_type == "linear":
        # note that for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size_a, config.mm_hidden_size_a)
    elif projector_type == 'identity':
        return IdentityMap()
    elif projector_type == 'conv':
        return nn.Conv1d(in_channels=config.mm_hidden_size_a, out_channels=config.mm_hidden_size_a, kernel_size=2, stride=2, padding=0) # 50Hz -> 25Hz
    else:
        raise ValueError(f"Unsupported projector type: {projector_type}")
        
def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


