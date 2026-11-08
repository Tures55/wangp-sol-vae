# Sol VAE for WanGP (MiniMax H3)

A patch that adds Comfy-Org's distilled **Sol VAE** decoder to [WanGP](https://github.com/deepbeepmeep/WanGP) as an extra "Video VAE" choice for MiniMax H3. It runs on WanGP's own fused INT8 kernels, with SageAttention when installed (PyTorch SDPA otherwise).

Upstream issue: https://github.com/deepbeepmeep/WanGP/issues/2625

## Why

On an RTX 3060 12 GB, Sol decodes faster than LightVAE and keeps almost all of the full VAE's detail.

| Decode, 73 frames | LightVAE Lite INT8 | Sol INT8 (this patch) |
|---|---|---|
| 544p | 4.5 s, 1.66 GB peak | **3.5 s**, 2.66 GB peak |
| 1088p | 17.7 s, 3.31 GB peak | **14.1 s**, 4.64 GB peak |

Quality against the full H3 VAE (same latents, 1920x1088):

| | PSNR / SSIM vs full VAE | Sharpness (Laplacian variance) |
|---|---|---|
| full H3 VAE | — | 19.5 |
| Sol | 45.5 dB / 0.988 | 19.4 |
| LightVAE | 41.8 dB / 0.982 | 15.3 |

A full generation (124 frames, 832x480, FastH3 distilled) ran end to end in 164 s with no errors.

## Install

1. Download `vae/minimax_h3_sol_vae_int8_convrot.safetensors` (about 2 GB) from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) and put it in `ckpts/minimax_h3/` inside your WanGP folder. The patch does not download it.
2. In the WanGP folder, apply the patch that matches your version:
   ```bash
   git apply patches/wangp-19.25-sol-vae.diff
   ```
   (`wangp-19.21-sol-vae.diff` for WanGP 19.21.)
3. Restart WanGP. In the MiniMax H3 settings choose **Video VAE → Sol VAE INT8 ConvRot (Comfy-Org)**.

`SOL_SAGE=0` turns off SageAttention and `SOL_FUSED=0` turns off the fused INT8 path, for comparison.

## Status

- Tested on WanGP 19.21, RTX 3060 (sm_86), Windows.
- The 19.25 patch applies cleanly and parses, but has not been run on 19.25.
- Spatial X2 upsampling still swaps in the X2 decoder, as it does for LightVAE.

## Licenses

- `sol_decoder.py` (inside the patches) follows the structure of ComfyUI's `comfy/ldm/minimax/vae.py::NonCausalViT3DDecoder`. ComfyUI is GPL-3.0, so this repository is GPL-3.0 (see `LICENSE`).
- WanGP itself is under the WanGP Community License. The patches only change a few lines of it.
- No model weights are included. The Sol VAE and MiniMax H3 are under the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Read its territory and use terms before using them.
