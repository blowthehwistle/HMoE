from torchtitan.protocols.train_spec import TrainSpec, register_train_spec
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.models.llama3.infra.pipeline import pipeline_llama
from torchtitan.components.optimizer import build_optimizers
from ttp.datasets.dataloader import build_dataloader
from ttp.components.loss import build_loss
from ttp.components.metrics import build_ttp_metrics_processor
from ttp.components.tokenizer import build_tokenizer

from ttp.experiments.hmoe.model.model import Hybrid
from ttp.experiments.hmoe.model.args import HybridModelArgs
from ttp.experiments.hmoe.infra.parallelize import parallelize_hybrid


# 配置函数
hybrid_hmoe_configs = {
    "debugmodel": HybridModelArgs(
        dim=2048,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        multiple_of=8,
        norm_eps=1e-6,
        use_flex_attn=False,
        attn_mask_type="causal",
        eos_id=0,
        qk_norm=True,
        qk_norm_after_rope=False,
        # hybrid config
        hybrid_config="ae" * 24,
        # MoE configs
        expert_hidden_dim=768,
        num_experts=128,
        num_shared_experts=0,
        top_k=8,
        use_grouped_mm=True,
        max_seq_len=4096,
        aux_load_balance_loss_coeff=1e-3,
        p_penalty_coeff=5e1,
        initializer_range=0.01275775908,
        use_heterogeneous_moe=True,
        num_expert_groups=4,
        num_experts_per_group=32,
        expert_hidden_dims=[512, 672, 896, 1024]
    ),
}

register_train_spec(
    TrainSpec(
        name="hmoe",
        model_cls=Hybrid,
        model_args=hybrid_hmoe_configs,
        parallelize_fn=parallelize_hybrid,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_loss,
        build_metrics_processor_fn=build_ttp_metrics_processor
    )
)
