from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_prior")
@dataclass
class ACTPriorConfig(ACTConfig):
    """ACTConfig + a scene-conditioned CVAE prior (Model 2).

    The DINOv2 scene prior can read MORE than one static scene camera (e.g. an overview
    + an external view) and fuse their CLS tokens, which helps a language-free policy
    infer WHICH subtask is active by reducing occlusion. Default = mean-pool (fuses only
    a couple of tokens: the few-items regime where average pooling wins, and it adds no
    parameters). Leave `scene_prior_cameras=None` to keep the original single-camera
    (image_features[0]) behavior exactly — existing checkpoints load unchanged.
    """

    #Frozen ViT for the scene prior. "dinov2" (ViT-S/14) or "dinov3" (ViT-S/16). Both are ~21M
    #params with a 384-dim CLS, so switching does not touch `prior_mlp` or invalidate a
    #checkpoint -- only the patch grid differs, and that stays inside the ViT since read the
    #CLS alone. DINOv3 is trained on LVD-1689M with denser supervision and gives sharper dense
    #features, which is what a scene prior for a precision task wants.
    #NOTE: DINOv3 weights are license-gated -- accept at
    #huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m or the download returns 403.
    #SEPARATE FIELD, deliberately. `vision_backbone` is ACT's own ResNet trunk and its validator
    #rejects anything not starting with "resnet", so overloading it to name the scene ViT made
    #`--policy.vision_backbone=dinov3` fail with "must be one of the ResNet variants". The scene
    #prior gets its own knob; the trunk keeps resnet18.
    scene_backbone: str = "dinov2"            #"dinov2" (ViT-S/14) | "dinov3" (ViT-S/16)
    #Local weights directory, to bypass the DINOv3 licence gate entirely. When set, this is loaded
    #with AutoModel.from_pretrained instead of the hub repo, and the CLS width is read from the
    #model rather than assumed -- a ViT-L/16 has a 1024-dim CLS, not 384.
    scene_backbone_path: str | None = None

    #Static scene cameras the prior fuses (keys must be in input_features). None => the
    #first image feature only (back-compat). Do NOT include the wrist cam (local view).
    scene_prior_cameras: list[str] | None = None
    #"mean" (default; parameter-free, robust) or "concat" (ablation; larger prior input).
    scene_fusion: str = "mean"

    #Free bits (Kingma et al. 2016): a floor, in nats PER LATENT DIM, below which the KL is not
    #penalised. With a LEARNED prior the KL has a trivial minimum -- the prior chases the
    #posterior until z carries no information (measured on plates100: kld -> 5e-4 and
    #sigma_prior -> 1.87, i.e. the scene latent went dead, which for a language-free policy
    #kills the very thing that selects the subtask). Free bits zeroes the KL gradient only for
    #dims already below the floor, so it stops the collapse without a schedule to tune.
    #0.0 = disabled = the exact original objective, so existing checkpoints are unaffected.
    kl_free_bits: float = 0.0

    #See ACTConfig. act_prior re-implements the encoder loop, so the embedding is added in both.
    camera_identity_embedding: bool = True

    #HOW OFTEN to measure whether the latent is doing anything, in optimiser steps. 0 disables.
    #Costs two extra forwards on the step it fires, so 500 is ~0.4% overhead.
    #
    #Rationale. An earlier run trained for 33k steps and finished with a latent that
    #moved the action by 1.2e-5 rad -- a dead branch carrying 85% of the parameters -- and NOTHING
    #in the loss said so. l1_loss fell, kld_loss fell, both looked healthy. kld falling to 2e-4 is
    #not convergence, it is the prior catching a posterior that has collapsed, and the two are
    #indistinguishable from the loss curve alone. These three numbers separate them:
    #
    #  latent_authority  how far the predicted chunk moves when z goes from the posterior mean to
    #                    ZERO. ~0 means the decoder ignores z: the branch is decoration.
    #  prior_post_gap    how far it moves when z goes from the posterior mean to the PRIOR mean.
    #                    This is the train/test mismatch IN ACTION UNITS -- at training z comes
    #                    from q(z|a,s), which has seen the future actions; at inference it comes
    #                    from p(z|scene), which has not.
    #  mu_q_std          spread of the posterior mean across the batch. ~0 means the posterior has
    #                    collapsed and there is nothing left for the prior to predict.
    #
    #Read them together. authority~0 is a wasted branch but harmless. authority HIGH with
    #prior_post_gap also high is worse than useless: the decoder has come to depend on information
    #it will not have at inference. That combination is the specific risk of latent_injection=
    #"decoder_seed", which is why this defaults on whenever a run turns the latent up.
    latent_diag_every: int = 500

    #WEIGHT ON KL( sg(q) || p ) -- the term that actually TRAINS THE PRIOR. 0 disables it.
    #
    #WHY IT HAD TO EXIST. Measured on prior_sd_aug at step 1000: total kld 8.77 nats over 32 dims
    #= 0.274 nats/dim, against kl_free_bits=0.5. Free bits clamps EVERY dim at the floor, so
    #kld_opt is the constant 32*0.5 = 16.0 and its gradient is exactly zero -- the headline loss
    #sat at 160.000 + l1 for the whole run, which is what that constant looks like. The prior MLP
    #therefore never left initialisation (mu_prior_std 0.0315 against a posterior spread of 0.457,
    #i.e. it emits nearly the same vector for every scene) while `decoder_seed` gave the latent
    #real authority over the action: latent_authority 0.1642 with prior_post_gap 0.1620, meaning
    #99% of the latent's influence is information the decoder gets at TRAINING time from q(z|a,s)
    #-- which has seen the future actions -- and will not have at inference. That is strictly worse
    #than the dead latent of the earlier runs: it is a train/test shortcut.
    #
    #WHY sg(q) AND NOT MORE FREE-BITS TUNING. The symmetric KL has to serve two masters: keep the
    #posterior from collapsing (wants a floor) and teach the prior to predict it (wants gradient).
    #A floor high enough for the first kills the second, which is precisely the failure above, and
    #a floor low enough for the second re-opens the collapse that made the latent dead at 1e-4 in
    #the two runs before it. Splitting them ends the tug of war. The floored KL(q||p) keeps doing
    #what it did -- it is the variational bound and it regularises the posterior -- and this second
    #term adds a path to the prior that the floor cannot switch off. q is detached in it, so the
    #prior can never make its own job easier by dragging the posterior toward a constant.
    prior_fit_weight: float = 0.0

    #WHAT THE PRIOR READS from the frozen backbone.
    #  "cls"      the CLS token alone -- the original behaviour, and the default so that every
    #             existing checkpoint still loads.
    #  "cls_grid" CLS concatenated with the patch grid average-pooled to 2x3 cells.
    #
    #WHY THE GRID. The latent is supposed to say WHERE the bottles are; CLS is a global semantic
    #summary and discards most of that. A 2x3 pooling keeps left/centre/right and near/far while
    #adding only 6 pooled vectors, and it costs nothing extra in the backbone -- the patch tokens
    #are already computed and were being thrown away.
    scene_prior_features: str = "cls"

    #WHERE the latent enters the network. "token" is ACT's original design: z becomes ONE token
    #in the encoder sequence. Measured problem -- with 3 cameras that sequence is ~900 tokens, so
    #the latent is 1/900 of the evidence the decoder attends over, and its authority is diluted
    #to nothing: perturbing z moves the action chunk 0.0008 rad at RANDOM INIT versus 0.336 rad
    #for the cameras. That ceiling is structural, not an optimisation failure -- an A/B from
    #scratch showed kl_free_bits=2.0 holds the KL up (12.5 vs 0.28 nats) yet changes authority
    #by only 1.2x. Any scheme that steers the policy through z (skill codes, task tokens, style
    #vectors) is therefore dead on arrival with "token".
    #
    #"decoder_seed" instead initialises the ACT decoder's action queries with z, replacing the
    #zeros they start from today. Every one of the chunk_size queries then begins from the
    #latent, so it has direct authority over every predicted timestep and never has to win an
    #attention competition against 900 image patches. "both" does both.
    #"token" is the default, so existing checkpoints load and behave identically.
    latent_injection: str = "token"          #"token" | "decoder_seed" | "both"

    #At inference act_prior draws z ~ N(mu_scene, sigma_scene) FRESH on every forward pass, with
    #a measured sigma ~1.1. That is harmless only while z has no authority: give it authority
    #(see `latent_injection`) and it becomes full-scale random noise entering the arm at every
    #control step -- the fix would present as jitter. True uses mu_scene instead, which is the
    #right default for control; keep the sampling for deliberate multimodality (the anti-stall
    #reflex toggles this off to draw a genuinely different mode when the policy is stuck).
    #False = original behaviour, so existing checkpoints are unaffected.
    #SET THIS TRUE whenever latent_injection != "token".
    deterministic_latent: bool = False
