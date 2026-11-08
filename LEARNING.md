# Learning plan: `vision_transformer`

A study plan for Google's original ViT release, written for someone who has already worked through [vit-pytorch](https://github.com/tbvecc-forks/vit-pytorch) (the [study plan in ANALYSIS.md](https://github.com/tbvecc-forks/vit-pytorch/blob/main/ANALYSIS.md#12-study-plan), the CIFAR-10 runs and the ablations).

**What `vision_transformer` is:**
- the **reference code released with the ViT paper**: about 3,500 lines of JAX/Flax;
- MLP-Mixer, AugReg ("How to train your ViT"), SAM/GSAM, and LiT for evaluation;
- **fine-tuning and evaluation only.** The README says the models were pre-trained in [`big_vision`](https://github.com/tbvecc-forks/big_vision/blob/main/LEARNING.md);
- more than 50,000 public AugReg checkpoints.

**Why read it:** it's the smallest, most literal implementation of the paper, a good place to see exactly what "ViT" meant in 2020. Use it as the **ground truth** for comparing vit-pytorch, timm and `big_vision`.

**Suggested order across repos:** this repo, then `big_vision`, then timm. If you've already done timm, read this one as "where timm's `_load_weights` and `augreg` tags come from".

Each stage has **Read** (with line numbers at the time of writing), **Do** and a **Checkpoint**. Papers are listed at the end.

---

## Stage 0: setup (½–1 session)

JAX is not installed on this machine. `vit_jax/requirements.txt` pins `jax[cuda11_cudnn86]`, which is too old for a current A40 driver stack, so install a CUDA 12 build instead.

```bash
cd ~/vision_transformer
python3 -m venv --without-pip .venv-jax && curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv-jax/bin/python
grep -v "jax\[cuda11" vit_jax/requirements.txt | grep -v "find-links" > /tmp/requirements-cuda12.txt
.venv-jax/bin/pip install -r /tmp/requirements-cuda12.txt "jax[cuda12]"
.venv-jax/bin/python -m pytest vit_jax/models_test.py -q        # instantiates every registered model config
```

Then read [`README.md`](README.md): the "available ViT models", "expected ViT results" and AugReg sections.

> ✅ **Checkpoint:** `models_test.py` passes, and you can name the six papers this repo covers.

## Stage 1: the model, against the paper (1 session)

**Read**: [`vit_jax/models_vit.py`](vit_jax/models_vit.py), with the ViT paper's equations 1–4 side by side.
- `AddPositionEmbs` at `:37`: the shape comes from the input at initialization, and the initialization is std 0.02 "from BERT".
- `MlpBlock` at `:66`, `Encoder1DBlock` at `:105` (pre-norm: Eqs. 2–3), `Encoder` at `:159` (final LN: Eq. 4).
- `VisionTransformer` at `:211`:
  - the patch embedding is a **strided `nn.Conv`**, commented "merge s2d+emb into a single conv; it's the same";
  - the cls token is **zero-initialized**;
  - `classifier`: `'token'`, `'gap'`, `'unpooled'` or `'token_unpooled'`;
  - `representation_size`: a tanh pre-logits layer;
  - the head is **zero-initialized**.
- [`vit_jax/models_resnet.py`](vit_jax/models_resnet.py) (`StdConv` at `:30`): the R50+ViT hybrid stem. Paper §3.1, "Hybrid Architecture".

**Do**
1. Write a three-column table: **paper equation → this repo → vit-pytorch's [`vit.py`](https://github.com/tbvecc-forks/vit-pytorch/blob/main/vit_pytorch/vit.py)**. Note every difference:
   - initializations: zeros and 0.02 here vs `randn` in vit-pytorch;
   - q/k/v biases: present here, absent in vit-pytorch;
   - vit-pytorch's two extra patch LayerNorms;
   - the head initialization.
2. Show on paper that a conv with `kernel = stride = patch` equals vit-pytorch's `Rearrange` + `Linear`, and write down the weight reshape between them (`(ph, pw, c, D)` ↔ `(D, ph·pw·c)`).

> ✅ **Checkpoint:** explain why the head is zero-initialized for fine-tuning (paper §3.2) and what `representation_size` is for during pre-training.

## Stage 2: configs, model sizes and the "/16" naming (½ session)

**Read**
- [`vit_jax/configs/models.py`](vit_jax/configs/models.py): the `_register` decorator at `:29`, `get_b16_config` at `:113` and the other sizes, and `get_r50_b16_config` at `:270`. For hybrids, "/16" is the *effective* downsampling: ResNet stride × patch.
- [`vit_jax/models_test.py`](vit_jax/models_test.py): **exact parameter counts** for every config, e.g. `ViT-B_16: 86_567_656` at `:40`.
- [`vit_jax/configs/common.py`](vit_jax/configs/common.py): `get_config` at `:20` and `DATASET_PRESETS` at `:85` (CIFAR-10 is fine-tuned at **384 px**).
- [`vit_jax/configs/vit.py`](vit_jax/configs/vit.py), [`vit_jax/configs/augreg.py`](vit_jax/configs/augreg.py).

**Do**
1. Build vit-pytorch's `ViT(image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, dim_head=64, num_classes=1000)` and compare its parameter count with **86,567,656**. Account for every difference in parameters (patch LayerNorms, q/k/v biases, head).
2. Do the same with timm's `vit_base_patch16_224`, which should match exactly.

> ✅ **Checkpoint:** explain what "ViT-B/16", "R50+ViT-B/16" and "ViT-B/16-gap-norep" each specify.

## Stage 3: fine-tuning and changing resolution (1–2 sessions)

**Read**
- [`vit_jax/train.py`](vit_jax/train.py): `make_update_fn` at `:37` and `train_and_evaluate` at `:73`. The optimizer is **SGD with momentum 0.9 and no weight decay**, plus gradient clipping (`:138-145`), with gradient accumulation.
- [`vit_jax/input_pipeline.py`](vit_jax/input_pipeline.py): `get_datasets` at `:82` and `get_data_from_tfds` at `:138`. Inception crop, flip, resize to 384, scale to [-1, 1].
- [`vit_jax/checkpoint.py`](vit_jax/checkpoint.py): `load` at `:130`, `load_pretrained` at `:159`, and **`interpolate_posembed` at `:209`**. It splits off the cls position, reshapes the rest to a √N×√N grid, zooms bilinearly, then flattens. This is the correct version of what vit-pytorch's `pos_embedding[:seq]` gets wrong.

**Do**
1. **The official CIFAR-10 fine-tune**, shortened (README.md:123-125, "Expected ViT results"):
   ```bash
   .venv-jax/bin/python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
       --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 \
       --config.pretrained_dir='gs://vit_models/imagenet21k' \
       --config.total_steps=500 --config.warmup_steps=50
   ```
   The README reports **98.59% in 17 minutes** for this setting. Compare with your from-scratch 80–82% and think about what pre-training is worth. If memory runs short, lower `batch` or raise `accum_steps`.
2. **Position interpolation by hand:** load `ViT-B_16.npz`, take `Transformer/posembed_input/pos_embedding`, and call `interpolate_posembed(posemb, 577, True)` (14×14 → 24×24). Plot the cosine-similarity maps before and after (ViT paper, Fig. 7), and compare with your sin-cos analysis.
3. **Ablate the fine-tuning resolution:** `--config.pp.crop=224` vs 384 on CIFAR-10. How much does upsampling 32×32 images to 384 help?

> ✅ **Checkpoint:** explain why fine-tuning at a *higher* resolution than pre-training helps (paper §3.2), and what goes wrong without position interpolation.

## Stage 4: the AugReg checkpoint zoo (1 session)

**Read**
- [`vit_jax_augreg.ipynb`](vit_jax_augreg.ipynb): exploring `index.csv`, loading checkpoints in JAX **and in timm**, and fine-tuning.
- [`vit_jax/checkpoint.py:241`](vit_jax/checkpoint.py) `get_augreg_df`, and the filename scheme in the README:
  `{model}-{ds}-{epochs}-lr_{lr}-aug_{aug}-wd_{wd}-do_{do}-sd_{sd}--{adapt_ds}-steps_{N}k-lr_{lr}-res_{res}`.

**Do**
1. With `pandas`, query `gs://vit_models/augreg/index.csv`. For each model size, plot how upstream augmentation (`aug`) and weight decay (`wd`) affect the downstream score: the paper's main finding, from the raw data.
2. Load the same AugReg checkpoint in JAX and in timm (timm plan, Stage 3) and check the logits match.

> ✅ **Checkpoint:** summarize the AugReg paper's advice for training a ViT on a small or medium dataset. How does it relate to the gap between your CIFAR-10 from-scratch and fine-tuned results?

## Stage 5: Mixer and LiT (optional, 1 session)

**Read**
- [`vit_jax/models_mixer.py`](vit_jax/models_mixer.py): `MixerBlock` at `:32` and `MlpMixer` at `:48`. Token mixing by transposing, then an MLP; no attention at all.
- [`vit_jax/models_lit.py:109`](vit_jax/models_lit.py) `LitModel`, [`lit.ipynb`](lit.ipynb) and [`model_cards/lit.md`](model_cards/lit.md).

**Do**
1. Fine-tune Mixer-B/16 on CIFAR-10 (README command) and compare with ViT-B/16 at the same step budget.
2. Run zero-shot classification on CIFAR-10 with `lit.ipynb`. Compare with SigLIP in the `big_vision` plan (Stage 5).

> ✅ **Checkpoint:** what does the Mixer result say about how much self-attention matters for ViT's performance?

---

## Reading list

★ = core reading for the plan. Papers come from the README. Every link was checked against arXiv.

| | Paper | Link | In `vision_transformer` |
|---|---|---|---|
| ★ | *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT) | [arXiv 2010.11929](https://arxiv.org/abs/2010.11929) | [`models_vit.py`](vit_jax/models_vit.py), [`vit_jax.ipynb`](vit_jax.ipynb) |
| ★ | *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers* (AugReg) | [arXiv 2106.10270](https://arxiv.org/abs/2106.10270) | [`configs/augreg.py`](vit_jax/configs/augreg.py), [`vit_jax_augreg.ipynb`](vit_jax_augreg.ipynb) |
| ★ | *Attention Is All You Need* (the Transformer encoder ViT reuses) **(related)** | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | [`models_vit.py`](vit_jax/models_vit.py) |
| | *Big Transfer (BiT): General Visual Representation Learning* (StdConv/GroupNorm ResNets for hybrids; the repo was forked from `big_transfer`) **(related)** | [arXiv 1912.11370](https://arxiv.org/abs/1912.11370) | [`models_resnet.py`](vit_jax/models_resnet.py) |
| | *MLP-Mixer: An all-MLP Architecture for Vision* | [arXiv 2105.01601](https://arxiv.org/abs/2105.01601) | [`models_mixer.py`](vit_jax/models_mixer.py) |
| | *When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations* (SAM checkpoints) | [arXiv 2106.01548](https://arxiv.org/abs/2106.01548) | `gs://vit_models/sam` |
| | *Sharpness-Aware Minimization for Efficiently Improving Generalization* (SAM) | [arXiv 2010.01412](https://arxiv.org/abs/2010.01412) | `gs://vit_models/sam` |
| | *Surrogate Gap Minimization Improves Sharpness-Aware Training* (GSAM) | [arXiv 2203.08065](https://arxiv.org/abs/2203.08065) | `gs://vit_models/gsam` |
| | *LiT: Zero-Shot Transfer with Locked-image text Tuning* | [arXiv 2111.07991](https://arxiv.org/abs/2111.07991) | [`models_lit.py`](vit_jax/models_lit.py), [`lit.ipynb`](lit.ipynb) |
| | *Better plain ViT baselines for ImageNet-1k* (what changed after this repo; see `big_vision`) **(related)** | [arXiv 2205.01580](https://arxiv.org/abs/2205.01580) | `big_vision` |

---

## Gotchas collected while analyzing

- **`requirements.txt` pins CUDA 11 JAX.** Install `jax[cuda12]` instead (Stage 0).
- **142 MB of the clone isn't ViT code.** It's the `gh-pages` branch holding the browser LiT demo's model shards. `main` itself is small.
- **`train.py:242` uses `eval_every` where it means `checkpoint_every`,** so the checkpoint interval setting is ignored, apart from 0 disabling it.
- **`augreg.py` sets `pp.resize=448`,** but `input_pipeline.py` only reads `pp.crop`.
- **Fine-tuning uses SGD with no weight decay,** unlike the AdamW recipes in timm and `big_vision`.
