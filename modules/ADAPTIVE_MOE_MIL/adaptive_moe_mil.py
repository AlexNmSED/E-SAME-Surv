import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.AB_MIL.ab_mil import AB_MIL
from modules.WIKG_MIL.wikg_mil import WIKG_MIL


class ADAPTIVE_MOE_MIL(nn.Module):
    """Adaptive feature-level mixture of AB-MIL and WIKG experts.

    The router follows the research plan's dual-front-token design:
    T_wsi = [T_pheno || phi(T_mean)], where T_pheno is the average soft
    phenotype composition and T_mean is the raw mean patch-feature anchor.
    T_wsi is combined with differentiable spatial entropy from mandatory H5
    physical patch coordinates to produce soft expert-routing weights. The
    mean-feature token and DSE entropy can be disabled independently for
    ablations while still being computed for logging.
    """

    def __init__(
        self,
        in_dim=768,
        num_classes=4,
        ab_L=512,
        ab_D=128,
        wikg_dim_hidden=512,
        wikg_topk=6,
        wikg_chunk_size=1024,
        wikg_agg_type="bi-interaction",
        wikg_pool="attn",
        wikg_coord_neighbor_weight=0.0,
        wikg_coord_rbf_sigma=0.0,
        fusion_dim=512,
        phenotype_dim=8,
        context_dim=128,
        router_hidden_dim=128,
        router_use_mean_token=True,
        router_use_dse_entropy=True,
        dse_neighbors=8,
        dse_sigma=0.0,
        gate_temperature=1.0,
        gate_warmup_epochs=5,
        lambda_orth=0.01,
        use_evidential_fusion=False,
        evidential_as_prediction=True,
        dropout=0.1,
        ab_act=None,
        wikg_act=None,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.phenotype_dim = int(phenotype_dim)
        self.context_dim = int(context_dim)
        self.router_use_mean_token = bool(router_use_mean_token)
        self.router_use_dse_entropy = bool(router_use_dse_entropy)
        self.dse_neighbors = int(dse_neighbors)
        self.dse_sigma = float(dse_sigma)
        self.gate_temperature = float(gate_temperature)
        self.gate_warmup_epochs = int(gate_warmup_epochs)
        self.lambda_orth = float(lambda_orth)
        self.use_evidential_fusion = bool(use_evidential_fusion)
        self.evidential_as_prediction = bool(evidential_as_prediction)
        self.eps = 1e-8
        self.num_classes = int(num_classes)
        self.register_buffer("_spatial_offsets", self._build_spatial_offsets(self.dse_neighbors))

        self.local_expert = AB_MIL(
            L=ab_L,
            D=ab_D,
            num_classes=num_classes,
            dropout=dropout,
            act=ab_act if ab_act is not None else nn.ReLU(),
            in_dim=in_dim,
        )
        self.wikg_expert = WIKG_MIL(
            in_dim=in_dim,
            act=wikg_act if wikg_act is not None else nn.LeakyReLU(),
            dim_hidden=wikg_dim_hidden,
            topk=wikg_topk,
            num_classes=num_classes,
            agg_type=wikg_agg_type,
            dropout=dropout,
            pool=wikg_pool,
            neighbor_chunk_size=wikg_chunk_size,
            coord_neighbor_weight=wikg_coord_neighbor_weight,
            coord_rbf_sigma=wikg_coord_rbf_sigma,
        )

        # Expert classifiers are bypassed because fusion happens in feature space.
        for parameter in self.local_expert.classifier.parameters():
            parameter.requires_grad = False
        for parameter in self.wikg_expert.fc.parameters():
            parameter.requires_grad = False

        self.phenotype_head = nn.Linear(in_dim, phenotype_dim)
        self.context_head = nn.Sequential(
            nn.Linear(in_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )
        router_in_dim = (
            phenotype_dim
            + (context_dim if self.router_use_mean_token else 0)
            + (1 if self.router_use_dse_entropy else 0)
        )
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 2),
        )
        self.local_projection = nn.Sequential(nn.Linear(ab_L, fusion_dim), nn.LayerNorm(fusion_dim))
        self.wikg_projection = nn.Sequential(nn.Linear(wikg_dim_hidden, fusion_dim), nn.LayerNorm(fusion_dim))
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.local_evidence_head = nn.Linear(fusion_dim, num_classes) if self.use_evidential_fusion else None
        self.wikg_evidence_head = nn.Linear(fusion_dim, num_classes) if self.use_evidential_fusion else None

        self.register_buffer("_current_epoch", torch.tensor(0, dtype=torch.long))

    @property
    def current_epoch(self):
        return int(self._current_epoch.item())

    @property
    def t_mean_projection(self):
        return self.context_head

    def set_epoch(self, epoch):
        self._current_epoch.fill_(int(epoch))

    def _split_features_and_coords(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected a [B, N, D] feature tensor, got shape {tuple(x.shape)}.")
        if x.shape[0] != 1:
            raise ValueError("ADAPTIVE_MOE_MIL currently supports one bag per forward pass.")
        if x.shape[-1] != self.in_dim + 2:
            raise ValueError(
                f"ADAPTIVE_MOE_MIL requires H5 features with appended physical coordinates; "
                f"expected dimension {self.in_dim + 2}, got {x.shape[-1]}."
            )
        coords = x[..., -2:]
        if not torch.isfinite(coords).all():
            raise ValueError("ADAPTIVE_MOE_MIL received missing or invalid H5 coordinates.")
        return x[..., : self.in_dim], coords

    @staticmethod
    def _build_spatial_offsets(num_neighbors):
        target_candidates = max(int(num_neighbors) * 4, 8)
        radius = 1
        offsets = []
        while len(offsets) < target_candidates:
            offsets = [
                (x, y)
                for x in range(-radius, radius + 1)
                for y in range(-radius, radius + 1)
                if x != 0 or y != 0
            ]
            radius += 1
        offsets.sort(key=lambda value: (value[0] ** 2 + value[1] ** 2, value[0], value[1]))
        return torch.tensor(offsets, dtype=torch.long)

    def _neighbor_weights(self, coords):
        """Build sparse physical neighbors from regular H5 patch-grid coordinates."""
        coords = coords[0].float()
        num_patches = coords.shape[0]
        if num_patches == 1:
            index = torch.zeros((1, 1, 1), dtype=torch.long, device=coords.device)
            weight = torch.ones((1, 1, 1), dtype=coords.dtype, device=coords.device)
            return index, weight

        steps = []
        for axis in range(2):
            unique_values = torch.unique(coords[:, axis], sorted=True)
            differences = torch.diff(unique_values)
            differences = differences[differences > 0]
            steps.append(differences.min() if differences.numel() else coords.new_tensor(1.0))
        grid_step = torch.stack(steps).clamp_min(self.eps)
        grid_coords = torch.round((coords - coords.min(dim=0).values) / grid_step).long()

        y_width = int(grid_coords[:, 1].max().item()) + 1
        keys = grid_coords[:, 0] * y_width + grid_coords[:, 1]
        sorted_keys, sorted_order = torch.sort(keys)
        candidate_grid = grid_coords.unsqueeze(1) + self._spatial_offsets.unsqueeze(0)
        candidate_keys = candidate_grid[..., 0] * y_width + candidate_grid[..., 1]
        positions = torch.searchsorted(sorted_keys, candidate_keys)
        clamped_positions = positions.clamp(max=num_patches - 1)
        in_bounds = (
            (candidate_grid[..., 0] >= 0)
            & (candidate_grid[..., 0] <= grid_coords[:, 0].max())
            & (candidate_grid[..., 1] >= 0)
            & (candidate_grid[..., 1] < y_width)
        )
        valid = in_bounds & (positions < num_patches) & (sorted_keys[clamped_positions] == candidate_keys)
        candidate_index = sorted_order[clamped_positions]

        candidate_coords = coords[candidate_index]
        candidate_distances = torch.linalg.vector_norm(candidate_coords - coords.unsqueeze(1), dim=-1)
        candidate_distances = candidate_distances.masked_fill(~valid, float("inf"))
        num_neighbors = min(max(self.dse_neighbors, 1), candidate_distances.shape[1])
        neighbor_distances, selected = torch.topk(candidate_distances, k=num_neighbors, dim=-1, largest=False)
        neighbor_index = torch.gather(candidate_index, 1, selected)

        valid_neighbors = torch.isfinite(neighbor_distances)
        row_index = torch.arange(num_patches, device=coords.device).unsqueeze(1).expand_as(neighbor_index)
        neighbor_index = torch.where(valid_neighbors, neighbor_index, row_index)
        if self.dse_sigma > 0:
            sigma = neighbor_distances.new_tensor(self.dse_sigma)
        else:
            finite_distances = neighbor_distances[valid_neighbors]
            sigma = finite_distances.detach().median() if finite_distances.numel() else grid_step.mean()
            sigma = sigma.clamp_min(self.eps)
        neighbor_weight = torch.exp(-neighbor_distances.square() / (2.0 * sigma.square()))
        neighbor_weight = torch.where(valid_neighbors, neighbor_weight, torch.zeros_like(neighbor_weight))
        isolated = neighbor_weight.sum(dim=-1, keepdim=True) <= self.eps
        neighbor_weight[:, :1] = torch.where(isolated, torch.ones_like(neighbor_weight[:, :1]), neighbor_weight[:, :1])
        neighbor_weight = neighbor_weight / neighbor_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return neighbor_index.unsqueeze(0), neighbor_weight.unsqueeze(0)

    def _differentiable_entropy(self, features, coords, phenotype_prob):
        neighbor_index, neighbor_weight = self._neighbor_weights(coords)
        batch_index = torch.arange(features.shape[0], device=features.device).view(-1, 1, 1)
        neighbor_prob = phenotype_prob[batch_index, neighbor_index]
        co_occurrence = torch.einsum(
            "bnk,bnq,bnql->bkl",
            phenotype_prob,
            neighbor_weight,
            neighbor_prob,
        )
        joint_prob = co_occurrence / co_occurrence.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        entropy = -(joint_prob * torch.log(joint_prob.clamp_min(self.eps))).sum(dim=(-2, -1))
        return entropy / math.log(max(self.phenotype_dim * self.phenotype_dim, 2))

    def _build_wsi_token(self, features, phenotype_prob):
        t_pheno = phenotype_prob.mean(dim=1)
        t_mean = features.mean(dim=1)
        t_mean_token = self.context_head(t_mean)
        t_wsi = torch.cat([t_pheno, t_mean_token], dim=-1)
        return t_wsi, t_pheno, t_mean, t_mean_token

    def _route(self, features, coords):
        phenotype_prob = F.softmax(self.phenotype_head(features), dim=-1)
        t_wsi, t_pheno, t_mean, t_mean_token = self._build_wsi_token(features, phenotype_prob)
        dse_entropy = self._differentiable_entropy(features, coords, phenotype_prob)
        router_token = t_wsi if self.router_use_mean_token else t_pheno
        if self.router_use_dse_entropy:
            router_input = torch.cat([router_token, dse_entropy.unsqueeze(-1)], dim=-1)
        else:
            router_input = router_token
        gate_logits = self.router(router_input)
        learned_weights = F.softmax(gate_logits / max(self.gate_temperature, self.eps), dim=-1)
        if self.current_epoch < self.gate_warmup_epochs:
            gate_weights = torch.full_like(learned_weights, 0.5)
        else:
            gate_weights = learned_weights
        route_info = {
            "t_wsi": t_wsi,
            "t_pheno": t_pheno,
            "t_mean": t_mean,
            "t_mean_token": t_mean_token,
        }
        return gate_weights, gate_logits, dse_entropy, route_info

    @staticmethod
    def _combined_attention(local_attention, wikg_attention, gate_weights):
        local_attention = F.softmax(local_attention.reshape(1, -1), dim=-1)
        wikg_attention = F.softmax(wikg_attention.reshape(1, -1), dim=-1)
        return gate_weights[:, :1] * local_attention + gate_weights[:, 1:] * wikg_attention

    def _evidence_to_opinion(self, evidence):
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        belief = evidence / strength
        uncertainty = evidence.new_tensor(float(self.num_classes)) / strength
        probability = alpha / strength
        return {
            "evidence": evidence,
            "alpha": alpha,
            "belief": belief,
            "uncertainty": uncertainty,
            "strength": strength,
            "probability": probability,
        }

    def _cautious_fusion(self, local_opinion, wikg_opinion):
        local_q = local_opinion["belief"] + local_opinion["uncertainty"]
        wikg_q = wikg_opinion["belief"] + wikg_opinion["uncertainty"]
        fused_q = torch.minimum(local_q, wikg_q)
        fused_uncertainty = torch.minimum(local_opinion["uncertainty"], wikg_opinion["uncertainty"])
        fused_belief_raw = torch.relu(fused_q - fused_uncertainty)
        normalizer = (fused_belief_raw.sum(dim=-1, keepdim=True) + fused_uncertainty).clamp_min(self.eps)
        fused_belief = fused_belief_raw / normalizer
        fused_uncertainty = fused_uncertainty / normalizer
        fused_strength = fused_belief.new_tensor(float(self.num_classes)) / fused_uncertainty.clamp_min(self.eps)
        fused_alpha = fused_belief * fused_strength + 1.0
        fused_alpha_sum = fused_alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        fused_probability = fused_alpha / fused_alpha_sum
        return {
            "alpha": fused_alpha,
            "belief": fused_belief,
            "uncertainty": fused_uncertainty,
            "strength": fused_alpha_sum,
            "probability": fused_probability,
        }

    def _probabilities_to_hazard_logits(self, probabilities):
        hazard_eps = max(self.eps, 1e-6)
        probabilities = probabilities.clamp_min(hazard_eps)
        previous_mass = torch.cumsum(probabilities, dim=-1) - probabilities
        survival_before = (1.0 - previous_mass).clamp_min(hazard_eps)
        hazards = (probabilities / survival_before).clamp(min=hazard_eps, max=1.0 - hazard_eps)
        return torch.logit(hazards)

    def _evidential_fusion(self, local_feature, wikg_feature, gate_weights):
        local_evidence = F.softplus(self.local_evidence_head(local_feature))
        wikg_evidence = F.softplus(self.wikg_evidence_head(wikg_feature))
        discounted_local_evidence = gate_weights[:, :1] * local_evidence
        discounted_wikg_evidence = gate_weights[:, 1:] * wikg_evidence
        raw_local_opinion = self._evidence_to_opinion(local_evidence)
        raw_wikg_opinion = self._evidence_to_opinion(wikg_evidence)
        discounted_local_opinion = self._evidence_to_opinion(discounted_local_evidence)
        discounted_wikg_opinion = self._evidence_to_opinion(discounted_wikg_evidence)
        fused_opinion = self._cautious_fusion(discounted_local_opinion, discounted_wikg_opinion)

        local_prob = raw_local_opinion["probability"]
        wikg_prob = raw_wikg_opinion["probability"]
        discounted_local_prob = discounted_local_opinion["probability"]
        discounted_wikg_prob = discounted_wikg_opinion["probability"]
        fused_prob = fused_opinion["probability"]
        expert_disagreement = torch.abs(local_prob - wikg_prob).mean(dim=-1)
        fused_prob_entropy = -(fused_prob * torch.log(fused_prob.clamp_min(self.eps))).sum(dim=-1)
        fused_belief_entropy = -(
            fused_opinion["belief"] * torch.log(fused_opinion["belief"].clamp_min(self.eps))
        ).sum(dim=-1)

        return {
            "local_evidence": local_evidence,
            "wikg_evidence": wikg_evidence,
            "discounted_local_evidence": discounted_local_evidence,
            "discounted_wikg_evidence": discounted_wikg_evidence,
            "local_alpha": raw_local_opinion["alpha"],
            "wikg_alpha": raw_wikg_opinion["alpha"],
            "discounted_local_alpha": discounted_local_opinion["alpha"],
            "discounted_wikg_alpha": discounted_wikg_opinion["alpha"],
            "fused_alpha": fused_opinion["alpha"],
            "local_belief": raw_local_opinion["belief"],
            "wikg_belief": raw_wikg_opinion["belief"],
            "discounted_local_belief": discounted_local_opinion["belief"],
            "discounted_wikg_belief": discounted_wikg_opinion["belief"],
            "fused_belief": fused_opinion["belief"],
            "local_probability": local_prob,
            "wikg_probability": wikg_prob,
            "discounted_local_probability": discounted_local_prob,
            "discounted_wikg_probability": discounted_wikg_prob,
            "fused_probability": fused_prob,
            "local_uncertainty": raw_local_opinion["uncertainty"],
            "wikg_uncertainty": raw_wikg_opinion["uncertainty"],
            "discounted_local_uncertainty": discounted_local_opinion["uncertainty"],
            "discounted_wikg_uncertainty": discounted_wikg_opinion["uncertainty"],
            "fused_uncertainty": fused_opinion["uncertainty"],
            "local_evidence_strength": raw_local_opinion["strength"],
            "wikg_evidence_strength": raw_wikg_opinion["strength"],
            "discounted_local_evidence_strength": discounted_local_opinion["strength"],
            "discounted_wikg_evidence_strength": discounted_wikg_opinion["strength"],
            "fused_evidence_strength": fused_opinion["strength"],
            "expert_disagreement": expert_disagreement,
            "fused_probability_entropy": fused_prob_entropy,
            "fused_belief_entropy": fused_belief_entropy,
        }

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        features, coords = self._split_features_and_coords(x)
        gate_weights, gate_logits, dse_entropy, route_info = self._route(features, coords)

        local_output = self.local_expert(
            features,
            return_WSI_attn=return_WSI_attn,
            return_WSI_feature=True,
        )
        wikg_output = self.wikg_expert(
            features,
            coords=coords,
            return_WSI_attn=return_WSI_attn,
            return_WSI_feature=True,
        )
        local_feature = self.local_projection(local_output["WSI_feature"])
        wikg_feature = self.wikg_projection(wikg_output["WSI_feature"])
        fused_feature = gate_weights[:, :1] * local_feature + gate_weights[:, 1:] * wikg_feature
        fused_feature = self.fusion_norm(fused_feature)
        logits = self.classifier(fused_feature)
        evidential_output = None
        if self.use_evidential_fusion:
            evidential_output = self._evidential_fusion(local_feature, wikg_feature, gate_weights)
            if self.evidential_as_prediction:
                logits = self._probabilities_to_hazard_logits(evidential_output["fused_probability"])

        expert_cosine = F.cosine_similarity(local_feature, wikg_feature, dim=-1)
        orth_loss = expert_cosine.square().mean()
        gate_entropy = -(gate_weights * torch.log(gate_weights.clamp_min(self.eps))).sum(dim=-1).mean()
        t_pheno = route_info["t_pheno"]
        t_pheno_entropy = -(t_pheno * torch.log(t_pheno.clamp_min(self.eps))).sum(dim=-1).mean()
        local_norm = torch.linalg.vector_norm(local_feature, dim=-1)
        wikg_norm = torch.linalg.vector_norm(wikg_feature, dim=-1)
        alignment_norm_gap = torch.abs(local_norm - wikg_norm).mean()
        aux_loss = self.lambda_orth * orth_loss

        output = {
            "logits": logits,
            "WSI_feature": fused_feature,
            "aux_loss": aux_loss,
            "orth_loss": orth_loss,
            "expert_cosine": expert_cosine.mean(),
            "alignment_norm_gap": alignment_norm_gap,
            "gate_entropy": gate_entropy,
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "gate_is_warmup": features.new_tensor(float(self.current_epoch < self.gate_warmup_epochs)),
            "dse_entropy": dse_entropy.mean(),
            "t_wsi_norm": torch.linalg.vector_norm(route_info["t_wsi"], dim=-1).mean(),
            "t_mean_norm": torch.linalg.vector_norm(route_info["t_mean"], dim=-1).mean(),
            "t_mean_token_norm": torch.linalg.vector_norm(route_info["t_mean_token"], dim=-1).mean(),
            "t_pheno_entropy": t_pheno_entropy,
            "fusion_feature_norm": torch.linalg.vector_norm(fused_feature, dim=-1).mean(),
            "uses_physical_coords": features.new_tensor(float(coords is not None)),
        }
        if evidential_output is not None:
            output.update(evidential_output)
            output["evidential_as_prediction"] = features.new_tensor(float(self.evidential_as_prediction))
        if return_WSI_attn:
            output["WSI_attn"] = self._combined_attention(
                local_output["WSI_attn"],
                wikg_output["WSI_attn"],
                gate_weights,
            )
            output["expert_attn"] = {
                "local": local_output["WSI_attn"],
                "wikg": wikg_output["WSI_attn"],
            }
            if "graph_index" in wikg_output:
                output["expert_graph"] = {
                    "wikg_index": wikg_output["graph_index"],
                    "wikg_weight": wikg_output.get("graph_weight"),
                }
        return output
