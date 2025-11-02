import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module, distribute_tensor, Replicate, Shard, Partial
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torchtitan.distributed import ParallelDims
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.tools.logging import logger
from torchtitan.models.llama3.infra.parallelize import apply_ac, apply_ddp
from torchtitan.distributed.expert_parallel import (
    TensorParallel,
    NoParallel,
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel
)
from torchtitan.experiments.llama4.infra.parallelize import (
    apply_compile,
    apply_fsdp
)
from ttp.config.job_config import TTPJobConfig
from ttp.experiments.hmoe.model.model import (
    Hybrid,
    AttentionBlock,
    FeedForwardBlock,
    MoEBlock
)


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
    etp_enabled: bool,
):
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(),
                # NOTE: 与llama4不同，hybrid里的moe block有norm
                "moe_norm": SequenceParallel(),
            }
            if ep_mesh is not None and not etp_enabled:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that
                # the reorderer, the all-to-all comms, and routed experts computation
                # are effectively running Sequence Parallel (split along the folded bs*slen dim)
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            if transformer_block.moe.shared_experts is not None:
                # input Replicate, output Partial
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial()
                        ),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None:
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
        elif etp_enabled:
            experts_mesh = ep_tp_mesh
            experts_plan = ExpertTensorParallel(tp_mesh=tp_mesh, ep_mesh=ep_mesh)
        else:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()

        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def apply_moe_tp_minus_sp(module: nn.Module, tp_mesh: DeviceMesh):
    moe_layer_plan = {
        # input / output sharding on the seqlen dim
        # all-gather for input, reduce-scatter for output
        "moe": PrepareModuleInputOutput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
            use_local_input=True,
            output_layouts=(Partial(),),
            desired_output_layouts=(Replicate(),),
        ),
        # replicate computation for the router
        "moe.router.gate": NoParallel(),
        # input Replicate, output Partial
        "moe.experts": TensorParallel(),
    }
    if module.moe.shared_experts is not None:
        moe_layer_plan.update(
            {
                "moe.shared_experts.w1": ColwiseParallel(),
                "moe.shared_experts.w2": RowwiseParallel(
                    output_layouts=Partial()
                ),
                "moe.shared_experts.w3": ColwiseParallel(),
            }
        )
    parallelize_module(
        module=module,
        device_mesh=tp_mesh,
        parallelize_plan=moe_layer_plan,
    )


class QKNormParallel(PrepareModuleInputOutput):

    def __init__(self, sharding_dim: int):
        super().__init__(
            input_layouts=(Shard(sharding_dim),),
            desired_input_layouts=(Shard(sharding_dim),),
            use_local_input=False,
            output_layouts=(Shard(sharding_dim),),
            desired_output_layouts=(Shard(sharding_dim),),
            use_local_output=True
        )

    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
            # us to simply just use from_local
            replicated_param = torch.nn.Parameter(distribute_tensor(param, device_mesh, [Replicate()]))
            module.register_parameter(p_name, replicated_param)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh):
        super()._apply(module, device_mesh)
        distribute_module(module, device_mesh, self._replicate_module_fn)


def apply_tp_minus_sp(
    model: Hybrid,
    tp_mesh: DeviceMesh,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool
):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        raise ValueError("Hybrid dosen't support enable_float8_tensorwise_tp.")
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    for block in model.layers.values():
        if isinstance(block, AttentionBlock):
            parallelize_module(
                block,
                tp_mesh,
                {
                    "attention": prepare_module_input(
                        input_layouts=(Replicate(), None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    "attention.q_norm": QKNormParallel(sharding_dim=2),
                    "attention.k_norm": QKNormParallel(sharding_dim=2),
                    "attention.wq": colwise_parallel(),
                    "attention.wk": colwise_parallel(),
                    "attention.wv": colwise_parallel(),
                    "attention.wo": rowwise_parallel(),
                }
            )
        if isinstance(block, FeedForwardBlock):
            parallelize_module(
                block,
                tp_mesh,
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Replicate(),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(),
                    "feed_forward.w3": colwise_parallel(),
                }
            )
        if isinstance(block, MoEBlock):
            apply_moe_tp_minus_sp(block, tp_mesh)

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)


def apply_non_moe_tp(
    model: Hybrid,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        raise ValueError("Hybrid dosen't support enable_float8_tensorwise_tp.")
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    for block in model.layers.values():
        if isinstance(block, AttentionBlock):
            parallelize_module(
                block,
                tp_mesh,
                {
                    "attention_norm": SequenceParallel(),
                    "attention": prepare_module_input(
                        input_layouts=(Shard(1), None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    "attention.q_norm": QKNormParallel(sharding_dim=2),
                    "attention.k_norm": QKNormParallel(sharding_dim=2),
                    "attention.wq": colwise_parallel(),
                    "attention.wk": colwise_parallel(),
                    "attention.wv": colwise_parallel(),
                    "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
                }
            )
        if isinstance(block, FeedForwardBlock):
            parallelize_module(
                block,
                tp_mesh,
                {
                    "ffn_norm": SequenceParallel(),
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


def parallelize_hybrid(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: TTPJobConfig,
):

    world_mesh = parallel_dims.world_mesh
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if (
        job_config.parallelism.context_parallel_degree > 1
        and model.model_args.use_flex_attn
    ):
        raise NotImplementedError("CP support for FlexAttention is still in progress.")

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )
    if parallel_dims.tp_enabled:
        if (
            job_config.parallelism.enable_async_tensor_parallel
            and not job_config.training.compile
        ):
            raise RuntimeError("Async TP requires --training.compile")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=job_config.parallelism.enable_async_tensor_parallel,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled
                and parallel_dims.ep_enabled
                and parallel_dims.etp_enabled
                else None
            ),
            etp_enabled=parallel_dims.etp_enabled,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        # NOTE: needed for torch.compile to work with dynamic shapes in token-choice MoE
        torch._dynamo.config.capture_scalar_outputs = True
        apply_compile(model)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            dp_mod_ep_mesh=(
                world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
                if parallel_dims.ep_enabled
                else None
            ),
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = world_mesh
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )

    return model
