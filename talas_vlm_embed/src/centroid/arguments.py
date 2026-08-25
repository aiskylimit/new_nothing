from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CentroidArguments:
    num_centroids: int = field(default=64, metadata={"help": "Number of non-dustbin centroids."})
    centroid_hidden_dim: int = field(default=256, metadata={"help": "Internal token dimension used for centroid matching."})
    centroid_dim: int = field(default=128, metadata={"help": "Downscaled dimension for each centroid slot."})
    centroid_layer_idx: int = field(default=-1, metadata={"help": "Teacher hidden layer used as token features."})
    centroid_temperature: float = field(default=0.02, metadata={"help": "InfoNCE temperature."})
    sinkhorn_epsilon: float = field(default=0.05, metadata={"help": "Entropy regularization for Sinkhorn."})
    sinkhorn_iters: int = field(default=5, metadata={"help": "Number of Sinkhorn normalization iterations."})
    dustbin_mass: Optional[float] = field(
        default=None,
        metadata={"help": "Fixed dustbin mass. Leave unset for SALAD-style token-count adaptive mass."},
    )
    drop_special_tokens: bool = field(default=False, metadata={"help": "Deprecated. Token mask is attention-based and keeps attended EOS tokens."})
    centroid_checkpoint: str = field(default=None, metadata={"help": "Checkpoint path for eval/resume."})
