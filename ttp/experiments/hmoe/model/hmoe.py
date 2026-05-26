import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from torch.profiler import record_function

from torchtitan.models.moe import TokenReorderer, FeedForward
from ttp.experiments.hmoe.model.moe import TokenChoiceTopKRouter
from .args import HybridModelArgs


class HeterogeneousGroupedExperts(nn.Module):

    def __init__(
        self,
        dim: int,
        num_expert_groups: int,
        num_experts_per_group: int,
        expert_hidden_dims: List[int],
        use_grouped_mm: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_expert_groups = num_expert_groups
        self.num_experts_per_group = num_experts_per_group
        self.num_experts = num_expert_groups * num_experts_per_group
        self.expert_hidden_dims = expert_hidden_dims
        self.use_grouped_mm = use_grouped_mm

        assert len(expert_hidden_dims) == num_expert_groups, \
            f"expert_hidden_dims length ({len(expert_hidden_dims)}) must equal num_expert_groups ({num_expert_groups})"

        self.group_params = nn.ModuleList()
        for group_idx in range(num_expert_groups):
            hidden_dim = expert_hidden_dims[group_idx]

            group_module = nn.Module()
            group_module.w1 = nn.Parameter(torch.empty(num_experts_per_group, hidden_dim, dim))
            group_module.w2 = nn.Parameter(torch.empty(num_experts_per_group, dim, hidden_dim))
            group_module.w3 = nn.Parameter(torch.empty(num_experts_per_group, hidden_dim, dim))

            self.group_params.append(group_module)

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.use_grouped_mm:
            return self._run_grouped_mm(x, num_tokens_per_expert)
        else:
            return self._run_for_loop(x, num_tokens_per_expert)

    def _run_grouped_mm(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        num_tokens_per_expert_list = num_tokens_per_expert.tolist()

        x_splits = torch.split(
            x[:sum(num_tokens_per_expert_list)],
            split_size_or_sections=num_tokens_per_expert_list,
            dim=0,
        )

        all_outputs = []

        with record_function("hmoe_grouped_experts"):
            for group_idx in range(self.num_expert_groups):
                group_params = self.group_params[group_idx]
                hidden_dim = self.expert_hidden_dims[group_idx]

                start_expert = group_idx * self.num_experts_per_group
                end_expert = start_expert + self.num_experts_per_group

                group_x_splits = []
                group_num_tokens_per_expert = []

                for expert_idx in range(start_expert, end_expert):
                    if expert_idx < len(x_splits):
                        group_x_splits.append(x_splits[expert_idx])
                        group_num_tokens_per_expert.append(num_tokens_per_expert[expert_idx].item())
                    else:
                        group_x_splits.append(torch.empty(0, x.shape[1], dtype=x.dtype, device=x.device))
                        group_num_tokens_per_expert.append(0)

                group_x = torch.cat(group_x_splits, dim=0)
                total_group_tokens = int(sum(group_num_tokens_per_expert))

                group_num_tokens_tensor = torch.tensor(group_num_tokens_per_expert, dtype=torch.int32, device=x.device)
                offsets = torch.cumsum(group_num_tokens_tensor, dim=0, dtype=torch.int32)

                label = f"hmoe_grouped_mm/group_{group_idx}/hidden_{hidden_dim}/tokens_{total_group_tokens}"
                with record_function(label):
                    h = F.silu(
                        torch._grouped_mm(group_x.bfloat16(), group_params.w1.bfloat16().transpose(-2, -1), offs=offsets)
                    )
                    h = h * torch._grouped_mm(
                        group_x.bfloat16(), group_params.w3.bfloat16().transpose(-2, -1), offs=offsets
                    )
                    group_out = torch._grouped_mm(h, group_params.w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(group_x)

                all_outputs.append(group_out)

        if all_outputs:
            return torch.cat(all_outputs, dim=0)
        else:
            return torch.zeros_like(x)

    def _run_for_loop(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        num_tokens_per_expert_list = num_tokens_per_expert.tolist()

        x_splits = torch.split(
            x[:sum(num_tokens_per_expert_list)],
            split_size_or_sections=num_tokens_per_expert_list,
            dim=0,
        )

        out_experts_splits = []
        for expert_idx, x_expert in enumerate(x_splits):

            print(f"{expert_idx} x_expert.shape: {x_expert.shape}")

            group_idx = expert_idx // self.num_experts_per_group
            group_expert_idx = expert_idx % self.num_experts_per_group
            group_params = self.group_params[group_idx]

            w1_expert = group_params.w1[group_expert_idx]
            w2_expert = group_params.w2[group_expert_idx]
            w3_expert = group_params.w3[group_expert_idx]

            h = F.silu(torch.matmul(x_expert, w1_expert.transpose(-2, -1)))
            h = h * torch.matmul(x_expert, w3_expert.transpose(-2, -1))
            h = torch.matmul(h, w2_expert.transpose(-2, -1))

            out_experts_splits.append(h)

        if out_experts_splits:
            return torch.cat(out_experts_splits, dim=0)
        else:
            return torch.zeros_like(x)

    def init_weights(self, init_std: float):

        for group_idx in range(self.num_expert_groups):
            group_params = self.group_params[group_idx]

            nn.init.trunc_normal_(group_params.w1, mean=0.0, std=0.02)
            nn.init.trunc_normal_(group_params.w2, mean=0.0, std=init_std)
            nn.init.trunc_normal_(group_params.w3, mean=0.0, std=init_std)

    def extra_repr(self) -> str:
        expert_dims_str = ",".join(map(str, self.expert_hidden_dims))
        return f"HeterogeneousGroupedExperts(dim={self.dim}, groups={self.num_expert_groups}, experts_per_group={self.num_experts_per_group}, hidden_dims=[{expert_dims_str}], use_grouped_mm={self.use_grouped_mm})"


class HeterogeneousMoE(nn.Module):

    def __init__(self, model_args: HybridModelArgs):
        super().__init__()

        self.dim = model_args.dim
        self.model_args = model_args

        if not model_args.use_heterogeneous_moe:
            raise ValueError("use_heterogeneous_moe must be True to use HeterogeneousMoE")

        if model_args.expert_hidden_dims is None:
            raise ValueError("expert_hidden_dims must be specified for heterogeneous MoE")

        self.num_experts = model_args.num_expert_groups * model_args.num_experts_per_group

        self.experts = HeterogeneousGroupedExperts(
            dim=self.dim,
            num_expert_groups=model_args.num_expert_groups,
            num_experts_per_group=model_args.num_experts_per_group,
            expert_hidden_dims=model_args.expert_hidden_dims,
            use_grouped_mm=model_args.use_grouped_mm
        )

        self.router = TokenChoiceTopKRouter(
            dim=self.dim,
            num_experts=self.num_experts,
            top_k=model_args.top_k,
            use_sigmoid=model_args.moe_router_use_sigmoid,
            route_norm=model_args.route_norm,
        )

        self.reorderer = TokenReorderer(
            num_experts=self.num_experts,
            top_k=model_args.top_k
        )

        if model_args.num_shared_experts > 0:
            if model_args.shared_expert_hidden_dim is None:
                avg_hidden_dim = sum(model_args.expert_hidden_dims) // model_args.num_expert_groups
                shared_hidden_dim = avg_hidden_dim * model_args.num_shared_experts
            else:
                shared_hidden_dim = model_args.shared_expert_hidden_dim * model_args.num_shared_experts

            self.shared_experts = FeedForward(
                dim=self.dim,
                hidden_dim=shared_hidden_dim
            )
        else:
            self.shared_experts = None

        self.load_balance_coeff = model_args.load_balance_coeff
        self.aux_load_balance_loss_coeff = model_args.aux_load_balance_loss_coeff
        self.p_penalty_coeff = model_args.p_penalty_coeff

        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff >= 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            importance_per_expert,
        ) = self.router(x, self.expert_bias)

        if self.p_penalty_coeff is not None and self.p_penalty_coeff > 0.0:
            self.last_p_penalty_loss = self._compute_p_penalty_loss(
                top_scores, selected_experts_indices, bs * slen
            )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        token_indices_experts_sorted = token_indices_experts_sorted.reshape(
            -1, 1
        ).expand(-1, dim)

        routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        routed_output = (
            routed_output.to(torch.float32)
            * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        out = out.scatter_add(
            dim=0, index=token_indices_experts_sorted, src=routed_output
        )
        out = out.reshape(bs, slen, dim)

        num_tokens = bs * slen
        if self.aux_load_balance_loss_coeff is not None:
            importance_per_expert = importance_per_expert.to(torch.float32)
            load_per_expert = num_tokens_per_expert.to(torch.float32).to(importance_per_expert.device)
            self.last_load_balancing_loss = (
                self.experts.num_experts
                * (importance_per_expert * load_per_expert).sum()
                / (num_tokens ** 2)
            )

        return out

    def _compute_p_penalty_loss(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        expert_sizes = self.model_args.expert_hidden_dims

        all_expert_dims = []
        for group_idx in range(self.model_args.num_expert_groups):
            group_size = expert_sizes[group_idx]
            for _ in range(self.model_args.num_experts_per_group):
                all_expert_dims.append(group_size)

        all_expert_dims_tensor = torch.tensor(all_expert_dims, dtype=torch.float32, device=top_scores.device)

        selected_expert_dims = all_expert_dims_tensor[selected_experts_indices]

        numerator = (top_scores * selected_expert_dims).sum()

        denominator = all_expert_dims_tensor.sum()

        top_k = top_scores.shape[1]
        total_activations = num_tokens * top_k

        if denominator > 0 and total_activations > 0:
            p_penalty_loss = numerator / total_activations / denominator
        else:
            p_penalty_loss = torch.tensor(0.0, device=top_scores.device)

        return p_penalty_loss

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        if self.load_balance_coeff is not None:
            with torch.device(buffer_device):
                self.expert_bias = torch.zeros(
                    self.num_experts, dtype=torch.float32
                )
                self.tokens_per_expert = torch.zeros(
                    self.num_experts, dtype=torch.float32
                )

    def extra_repr(self) -> str:
        expert_sizes = ",".join(map(str, self.experts.expert_hidden_dims))
        shared_info = f"+shared({self.shared_experts.w1.out_features})" if self.shared_experts else ""
        return f"HeterogeneousMoE({self.dim}->{expert_sizes}{shared_info}->{self.dim})"
