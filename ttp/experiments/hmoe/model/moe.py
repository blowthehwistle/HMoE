import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.moe import GroupedExperts, TokenReorderer, FeedForward
from ttp.utils.nvtx import nvtx_range
from .args import HybridModelArgs


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        use_sigmoid (bool): Whether to use sigmoid or softmax for router scores. Default is False.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        use_sigmoid: bool = False,
        route_norm: bool = True,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_sigmoid = use_sigmoid
        self.route_norm = route_norm

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """

        scores = self.gate(x)

        if self.use_sigmoid:
            scores = torch.sigmoid(scores.to(torch.float32))
        else:
            scores = F.softmax(scores.to(torch.float32), dim=1)

        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        if self.route_norm:
            top_scores_sum = top_scores.sum(dim=1, keepdim=True)
            top_scores_sum = torch.clamp(top_scores_sum, min=torch.finfo(top_scores_sum.dtype).eps)
            top_scores = top_scores / top_scores_sum

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        importance_per_expert = scores.sum(dim=0)

        return top_scores, selected_experts_indices, num_tokens_per_expert, importance_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    def __init__(self, model_args: HybridModelArgs):
        super().__init__()

        dim = model_args.dim
        num_experts = model_args.num_experts

        assert model_args.expert_hidden_dim % model_args.multiple_of == 0
        hidden_dim = model_args.expert_hidden_dim

        self.use_grouped_mm = model_args.use_grouped_mm
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=self.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=model_args.top_k,
            use_sigmoid=model_args.moe_router_use_sigmoid,
            route_norm=model_args.route_norm,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=model_args.top_k)

        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * model_args.num_shared_experts)
            if model_args.num_shared_experts > 0
            else None
        )

        self.aux_load_balance_loss_coeff = model_args.aux_load_balance_loss_coeff

        self.load_balance_coeff = model_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff >= 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        with nvtx_range("moe/router"):
            (
                top_scores,
                selected_experts_indices,
                num_tokens_per_expert,
                importance_per_expert,
            ) = self.router(x, self.expert_bias)

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        with nvtx_range("moe/reorder"):
            (
                top_scores_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
            ) = self.reorderer(top_scores, selected_experts_indices)

        with nvtx_range("moe/dispatch_gather"):
            token_indices_experts_sorted = token_indices_experts_sorted.reshape(
                -1, 1
            ).expand(-1, dim)

            routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

        with nvtx_range("moe/experts"):
            routed_output = self.experts(routed_input, num_tokens_per_expert)

        with nvtx_range("moe/combine_scatter"):
            routed_output = (
                routed_output.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

            if self.shared_experts is not None:
                out = self.shared_experts(x)
            else:
                out = torch.zeros_like(x.reshape(bs * slen, dim))

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
                    self.experts.num_experts, dtype=torch.float32
                )
                self.tokens_per_expert = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
