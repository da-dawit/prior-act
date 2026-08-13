"""
Model 2: ACT + Scene-Conditioned CVAE Prior
Extends vanilla ACT by replacing the fixed N(0,I) inference prior
with a DINOv2-conditioned prior p(z|scene).

Drop-in alongside modeling_act.py. Register as policy type 'act_prior'.
"""
from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act_prior.configuration_act_prior import ACTPriorConfig
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
    get_activation_fn,
)
from lerobot.policies.pretrained import PreTrainedPolicy


class ACTPriorPolicy(PreTrainedPolicy):
    """
    ACT + Scene-Conditioned CVAE Prior (Model 2).

    Only difference from vanilla ACT:
      - DINOv2 ViT-S encodes the current image → CLS token
      - Small MLP maps CLS token → (mu_prior, sigma_prior)
      - KL loss uses N(mu_prior, sigma_prior) as target instead of N(0,I)
      - At inference: z ~ N(mu_prior, sigma_prior) instead of z = 0
    """
    config_class = ACTPriorConfig
    name = "act_prior"

    def __init__(self, config: ACTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTWithPrior(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )
        self.reset()

    def get_optim_params(self):
        return [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("model.backbone")
                    and not n.startswith("model.dino")   # DINOv2 always frozen
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor],
                             latent_override: Tensor | None = None) -> Tensor:
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions = self.model(batch, latent_override=latent_override)[0]
        return actions

    def forward(self, batch: dict[str, Tensor], return_actions: bool = False,
                latent_override: Tensor | None = None) -> tuple[Tensor, dict]:
        # `return_actions=True` additionally returns the grad-enabled predicted chunk from
        # this SAME forward, so a caller can add a trajectory-consistency term without a second
        # (wasteful, detached) forward. Default path is byte-for-byte the original (loss, dict).
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions_hat, (mu_hat, log_sigma_x2_hat, mu_prior, sigma_prior) = self.model(
            batch, latent_override=latent_override)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}

        # `mu_hat is not None` GUARDS THE EVAL PATH, and without it eval crashes.
        # The VAE encoder runs only when self.training, so in eval mode mu_hat and
        # log_sigma_x2_hat are both None -- and so is mu_prior, which sends the code into the
        # vanilla-KL branch below where `1 + log_sigma_x2_hat` raises
        # `TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'`.
        # This is only reachable with eval_steps > 0, which requires dataset.eval_split > 0, which
        # no act_prior run had set, so this path was first reached by the first run with a
        # held-out split, where it would have failed at the first evaluation.
        # Dropping the KL in eval is also the RIGHT answer, not just a patch: without the posterior
        # there is no variational bound to report, and the eval number want is the L1 the policy
        # will actually be judged on.
        if self.config.use_vae and mu_hat is not None:
            if mu_prior is not None:
                # ── Scene-conditioned KL (Model 2) ──────────────────────────────
                # KL( N(mu_q, sigma_q²) || N(mu_p, sigma_p²) )
                # = log(sigma_p/sigma_q) + (sigma_q² + (mu_q-mu_p)²)/(2*sigma_p²) - 0.5
                sigma_q = torch.exp(0.5 * log_sigma_x2_hat).clamp(min=1e-4)
                kld_per_dim = (
                    torch.log(sigma_prior / sigma_q)
                    + (sigma_q ** 2 + (mu_hat - mu_prior) ** 2) / (2 * sigma_prior ** 2)
                    - 0.5
                )                                              # (B, latent_dim)
            else:
                # Fallback to vanilla KL if no image available
                kld_per_dim = -0.5 * (
                    1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()
                )

            mean_kld = kld_per_dim.sum(-1).mean()              # TRUE KL, always reported
            free_bits = getattr(self.config, "kl_free_bits", 0.0) or 0.0
            if free_bits > 0:
                # Penalise only the dims carrying MORE than `free_bits` nats. Dims at or below
                # the floor contribute a constant and therefore NO gradient, so optimisation
                # stops squeezing the latent to zero while the reported kld stays honest.
                kld_opt = torch.clamp(kld_per_dim.mean(0), min=free_bits).sum()
            else:
                kld_opt = mean_kld

            loss_dict["kld_loss"] = mean_kld.item()

            # ── KL( sg(q) || p ): the term that trains the PRIOR ───────────────
            # Gradient flows into prior_mlp only. The posterior is detached, so the prior cannot
            # make its own job easier by dragging q toward a constant -- the collapse that left
            # latent_authority at 1e-4 in both earlier runs. And it carries NO free-bits floor,
            # because the floor is what silenced the prior in the run this replaces: every dim sat
            # under it, so the clamped KL was a constant and its gradient was exactly zero.
            _pw = float(getattr(self.config, "prior_fit_weight", 0.0) or 0.0)
            if _pw > 0.0 and mu_prior is not None:
                mu_q_d = mu_hat.detach()
                sigma_q_d = torch.exp(0.5 * log_sigma_x2_hat).clamp(min=1e-4).detach()
                prior_fit = (
                    torch.log(sigma_prior / sigma_q_d)
                    + (sigma_q_d ** 2 + (mu_q_d - mu_prior) ** 2) / (2 * sigma_prior ** 2)
                    - 0.5
                ).sum(-1).mean()
                loss_dict["prior_fit"] = prior_fit.item()
            else:
                prior_fit = None
            if latent_override is not None:
                # The latent came from outside (a discrete skill code), so the sampled z is
                # discarded and there is no variational bound left to optimise. Charging the KL
                # would keep squeezing a posterior nothing reads, at weight `kl_weight`,
                # competing with the L1 that is doing the actual work. It is reported but not optimised.
                loss = l1_loss
            else:
                loss = l1_loss + kld_opt * self.config.kl_weight
                if prior_fit is not None:
                    loss = loss + _pw * prior_fit
        else:
            loss = l1_loss

        # ── Latent health: logged, never optimised ──────────────────────────────
        # See `latent_diag_every` in the config for why these three exist. Under no_grad and off
        # the main path, so they cannot influence training -- they only make it observable.
        every = int(getattr(self.config, "latent_diag_every", 0) or 0)
        if every > 0 and self.training and latent_override is None and mu_hat is not None:
            self._diag_step = getattr(self, "_diag_step", 0) + 1
            if self._diag_step % every == 0:
                # CACHE the result. lerobot_train logs only the output_dict of whichever step
                # happens to coincide with log_freq, so diagnostics computed on a different
                # cadence would land in wandb as an almost-always-empty panel. Recomputed every
                # `every` steps, reported on every step.
                self._latent_diag = {}
                # EVAL MODE FOR THE PROBES, . `nn.MultiheadAttention`
                # keeps its dropout as a FLOAT attribute, not an `nn.Dropout` submodule, so it
                # stays active in train mode no matter what you do to the module tree. Two train
                # -mode forwards therefore differ by ~0.007 in normalised action units from
                # dropout alone -- which is larger than the effect being measured, An earlier version reported authority, prior gap and sampling jitter as three
                # near-identical numbers because all three were measuring dropout.
                was_training = self.model.training
                self.model.eval()
                try:
                    with torch.no_grad():
                        # Reference is a(z = mu_q), NOT actions_hat: actions_hat came from a
                        # SAMPLE z = mu_q + sigma_q*eps with sigma_q ~ 1.4, and it was computed
                        # in train mode. Anchor deterministically, in one mode.
                        a_mu = self.model(batch, latent_override=mu_hat)[0]
                        a_zero = self.model(batch, latent_override=torch.zeros_like(mu_hat))[0]
                        self._latent_diag["latent_authority"] = (
                            (a_mu - a_zero).abs().mean().item())
                        sig = torch.exp(0.5 * log_sigma_x2_hat)
                        a_samp = self.model(
                            batch, latent_override=mu_hat + sig * torch.randn_like(mu_hat))[0]
                        self._latent_diag["sample_jitter"] = (
                            (a_mu - a_samp).abs().mean().item())
                        if mu_prior is not None:
                            a_prior = self.model(batch, latent_override=mu_prior)[0]
                            self._latent_diag["prior_post_gap"] = (
                                (a_mu - a_prior).abs().mean().item())
                            # Scene-dependence of the prior itself. A prior that emits the same
                            # vector for every image is a learned CONSTANT wearing a learned constant, and it will still drive the KL to zero.
                            if mu_prior.shape[0] > 1:
                                self._latent_diag["mu_prior_std"] = mu_prior.std(0).mean().item()
                        # std over the BATCH dim is undefined for a single sample; a NaN in wandb
                        # is worse than an absent key.
                        if mu_hat.shape[0] > 1:
                            self._latent_diag["mu_q_std"] = mu_hat.std(0).mean().item()
                        self._latent_diag["mu_q_norm"] = mu_hat.norm(dim=-1).mean().item()
                finally:
                    if was_training:
                        self.model.train()
            loss_dict.update(getattr(self, "_latent_diag", {}))

        if return_actions:
            return loss, loss_dict, actions_hat
        return loss, loss_dict


class ACTWithPrior(nn.Module):
    """
    ACT model with DINOv2 scene prior.
    Identical to ACT except:
      - Adds DINOv2 ViT-S (frozen) + prior_mlp
      - At inference: samples z from N(mu_prior, sigma_prior)
      - Returns (mu_prior, sigma_prior) during training for KL computation
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # ── VAE encoder (same as vanilla ACT) ───────────────────────────────────
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(
                config.dim_model, config.latent_dim * 2
            )
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(
                    num_input_token_encoder, config.dim_model
                ).unsqueeze(0),
            )

        # ── Visual backbone (same as vanilla ACT) ───────────────────────────────
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False, False, config.replace_final_stride_with_dilation
                ],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )

        # ── Transformer encoder / decoder (same as vanilla ACT) ─────────────────
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Optional second entry point for the latent -- see `latent_injection` in the config.
        # As one token among ~900, z cannot steer the output; seeding the decoder's action
        # queries with it gives it authority over every predicted timestep directly. Only
        # built when asked for, so "token" checkpoints keep an identical state_dict.
        self._latent_injection = getattr(config, "latent_injection", "token")
        if self._latent_injection in ("decoder_seed", "both"):
            self.decoder_latent_seed_proj = nn.Linear(config.latent_dim, config.dim_model)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        n_1d_tokens = 1
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
                config.dim_model // 2
            )

        _n_cam = len(config.image_features) if config.image_features else 0
        if getattr(config, "camera_identity_embedding", True) and _n_cam > 1:
            self._cam_embed = nn.Embedding(_n_cam, config.dim_model)
            nn.init.zeros_(self._cam_embed.weight)
        else:
            self._cam_embed = None
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(
            config.dim_model, self.config.action_feature.shape[0]
        )

        # ── Frozen ViT scene prior ───────────────────────────────────────────────
        # DINOv2 ViT-S/14 or DINOv3 ViT-S/16 -- both ~21M params, both a 384-dim CLS, both
        # frozen throughout. The matching width is what makes this a drop-in: `prior_mlp` and
        # every existing checkpoint are unaffected by the choice. Only the patch grid differs
        # (14 vs 16), which stays internal to the ViT since read the CLS token alone.
        # DINOv3 weights are LICENSE-GATED: accept the terms at
        # huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m, otherwise the download 403s.
        backbone = getattr(config, "scene_backbone", "dinov2")
        local_path = getattr(config, "scene_backbone_path", None)
        if backbone == "dinov3":
            # Via HuggingFace, NOT torch.hub. The hub entry downloads straight from
            # dl.fbaipublicfiles.com and never sees the HF token, so it 403s even once the
            # licence is accepted; the HF repo honours it. Wrapped so `self.dino(x)` returns the
            # CLS token, matching the DINOv2 hub module's call signature exactly.
            from transformers import AutoModel

            _feat_mode = str(getattr(config, "scene_prior_features", "cls") or "cls")

            class _HFDino(nn.Module):
                #SPATIAL POOLING IS OPTIONAL AND OFF BY DEFAULT so every existing checkpoint keeps
                #loading: "cls" reproduces the original single-token output exactly.
                GRID = (2, 3)                    #rows x cols: near/far x left/centre/right

                def __init__(self, repo, mode="cls"):
                    super().__init__()
                    self.m = AutoModel.from_pretrained(repo)
                    self.mode = mode

                def forward(self, x):
                    h = self.m(pixel_values=x).last_hidden_state
                    cls = h[:, 0]
                    if self.mode != "cls_grid":
                        return cls
                    #TAKE THE PATCH TOKENS FROM THE END, not from index 1. DINOv3 puts register
                    #tokens between CLS and the patches, and their count varies by checkpoint;
                    #slicing from the tail is correct for any number of them.
                    n = h.shape[1] - 1
                    side = int(round(n ** 0.5))
                    while side * side > n:
                        side -= 1
                    patches = h[:, h.shape[1] - side * side:]
                    b, _, d = patches.shape
                    g = patches.transpose(1, 2).reshape(b, d, side, side)
                    g = F.adaptive_avg_pool2d(g, self.GRID)        #(B, D, 2, 3)
                    return torch.cat([cls, g.flatten(1)], dim=-1)  #(B, D*7)

            self.dino = _HFDino(local_path or "facebook/dinov3-vits16-pretrain-lvd1689m",
                                mode=_feat_mode)
        else:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
            )
        for p in self.dino.parameters():
            p.requires_grad = False

        #READ THE WIDTH, do not assume it. ViT-S is 384 for both v2 and v3, but a local path may
        #point at ViT-B (768) or ViT-L (1024) and a hardcoded 384 would fail deep inside prior_mlp
        #with a shape error rather than here.
        dino_dim = 384
        m = getattr(self.dino, "m", None)
        if m is not None and hasattr(m, "config"):
            dino_dim = int(getattr(m.config, "hidden_size", 384))

        # ── Scene cameras for the prior ─────────────────────────────────────────
        # The prior can fuse several STATIC scene cams (overview + external) to infer
        # which subtask is active without language. Resolve the configured keys to their
        # positions in image_features; default to camera 0 (original single-cam behavior).
        img_keys = list(self.config.image_features) if self.config.image_features else []
        scene_cams = getattr(config, "scene_prior_cameras", None)
        if scene_cams:
            self._scene_cam_idxs = [img_keys.index(k) for k in scene_cams if k in img_keys]
        if not scene_cams or not self._scene_cam_idxs:
            self._scene_cam_idxs = [0] if img_keys else []
        self._scene_fusion = getattr(config, "scene_fusion", "mean")
        # concat grows the prior input; mean keeps it at dino_dim (so old ckpts still load)
        #cls_grid emits CLS plus a 2x3 pooled grid, i.e. 7 vectors of width dino_dim.
        _per_cam = dino_dim * (7 if str(getattr(config, "scene_prior_features", "cls")) == "cls_grid" else 1)
        prior_in = (_per_cam * len(self._scene_cam_idxs)
                    if self._scene_fusion == "concat" else _per_cam)

        # Maps the (fused) DINOv2 CLS → (mu_prior, log_sigma_prior)
        self.prior_mlp = nn.Sequential(
            nn.Linear(prior_in, 256),
            nn.ReLU(),
            nn.Linear(256, config.latent_dim * 2),
        )

        # DINOv2 expects 224×224 with ImageNet normalization.
        # Images from lerobot are normalized with dataset stats.
        # We resize only here; normalization mismatch is absorbed by prior_mlp's
        # first linear layer which learns appropriate re-scaling.
        self.dino_resize = T.Resize((224, 224), antialias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _scene_images(self, batch) -> list:
        """The scene-camera image tensors the prior reads, in configured order."""
        imgs = batch["observation.images"]
        return [imgs[i] for i in self._scene_cam_idxs]

    def _encode_scene_prior(self, images):
        """
        images: list of (B, C, H, W) lerobot-normalized scene-camera tensors.
        DINOv2 CLS per camera, fused across cameras (mean by default; concat as ablation),
        then mapped to the prior. Single camera => identical to the original behavior.
        Returns: mu_prior (B, latent_dim), sigma_prior (B, latent_dim)
        """
        if torch.is_tensor(images):              # tolerate a single tensor
            images = [images]
        cls_list = []
        for image in images:
            img = self.dino_resize(image)
            with torch.no_grad():
                cls_list.append(self.dino(img))  # (B, 384)
        if self._scene_fusion == "concat":
            cls_token = torch.cat(cls_list, dim=-1)
        else:                                    # mean-pool (few tokens -> avg wins, no params)
            cls_token = torch.stack(cls_list, dim=0).mean(dim=0)
        out = self.prior_mlp(cls_token)          # (B, latent_dim * 2)
        mu_p, log_s_p = out.chunk(2, dim=-1)
        sigma_p = torch.exp(0.5 * log_s_p).clamp(min=1e-4)
        return mu_p, sigma_p

    def forward(self, batch, latent_override=None):
        # `latent_override` (B, latent_dim) replaces the CVAE sample. It is how a DISCRETE
        # skill code can drive the decoder instead of an opaque continuous latent: the
        # scene/affordance/instruction pick a code, the code becomes z. None keeps the
        # original behaviour exactly, so existing checkpoints are unaffected.
        if self.config.use_vae and self.training:
            assert "action" in batch

        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # ── VAE encoder (training only) ─────────────────────────────────────────
        if self.config.use_vae and "action" in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(
                    batch["observation.state"]
                ).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])

            if self.config.robot_state_feature:
                vae_encoder_input = torch.cat(
                    [cls_embed, robot_state_embed, action_embed], axis=1
                )
            else:
                vae_encoder_input = torch.cat([cls_embed, action_embed], axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)

        else:
            mu = log_sigma_x2 = None
            if self.config.use_vae and self.config.image_features and "observation.images" in batch:
                # ── Scene-conditioned sampling at inference (NEW) ────────────────
                mu_p, sigma_p = self._encode_scene_prior(self._scene_images(batch))
                # Resampling z every control step is only safe while z has no authority. Once it
                # does (latent_injection="decoder_seed"), a fresh draw at sigma~1.1 per step is
                # noise injected straight into the arm. `deterministic_latent` takes the mean.
                if getattr(self.config, "deterministic_latent", False):
                    latent_sample = mu_p
                else:
                    latent_sample = mu_p + sigma_p * torch.randn_like(mu_p)
            else:
                latent_sample = torch.zeros(
                    [batch_size, self.config.latent_dim], dtype=torch.float32
                ).to(batch["observation.state"].device)

        # ── Compute scene prior for KL during training (NEW) ────────────────────
        if self.training and self.config.use_vae and self.config.image_features:
            mu_prior, sigma_prior = self._encode_scene_prior(self._scene_images(batch))
        else:
            mu_prior, sigma_prior = None, None

        # ── Transformer encoder ─────────────────────────────────────────────────
        if latent_override is not None:
            latent_sample = latent_override.to(latent_sample.dtype)
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        )

        if self.config.robot_state_feature:
            encoder_in_tokens.append(
                self.encoder_robot_state_input_proj(batch["observation.state"])
            )
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(
                    batch["observation.environment_state"]
                )
            )
        if self.config.image_features:
            for _cam_i, img in enumerate(batch["observation.images"]):
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                #CAMERA IDENTITY -- see the note in policies/act/modeling_act.py. act_prior
                #re-implements this loop instead of calling ACT's, so the fix must be applied in
                #BOTH files or it silently lands in only one.
                if getattr(self, "_cam_embed", None) is not None:
                    cam_features = cam_features + self._cam_embed.weight[_cam_i].view(1, -1, 1, 1)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(
                    cam_pos_embed, "b c h w -> (h w) b c"
                )
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # Stash the encoder token sequence so an external module can read the policy's internal
        # representation without a second forward pass. This is what the RL-token bottleneck
        # compresses: everything the policy perceives, vision included -- which a residual actor
        # conditioned only on joint angles cannot see. (M, B, dim_model) -> (B, M, dim_model).
        self._last_encoder_out = encoder_out.transpose(0, 1)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        if self._latent_injection in ("decoder_seed", "both"):
            # Start every action query FROM the latent instead of from zero. This is the only
            # path by which z reaches the output without competing for attention against the
            # image patches, which is what makes a discrete skill code able to steer at all.
            decoder_in = decoder_in + self.decoder_latent_seed_proj(latent_sample).unsqueeze(0)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        # Return mu_prior and sigma_prior for KL computation in ACTPriorPolicy.forward()
        return actions, (mu, log_sigma_x2, mu_prior, sigma_prior)
