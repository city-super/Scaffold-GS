"""
Scaffold-GS -> Standard 3DGS PLY (pruned, fp16 internals, bg/sky boosted)

This script converts a Scaffold-GS PLY model plus three TorchScript MLPs
(color, covariance, opacity) into a standard 3DGS PLY file with a reduced
set of Gaussians. Key points:

- Sparse voxelization and min/max scale options are removed for simplicity.
- Model inference and intermediate results use FP16 for memory/bandwidth
    efficiency; final file is written in float32 for compatibility.
- Background/sky boosting: anchors detected as sky or far-away have their
    opacity-selection threshold increased so distant/sky anchors are pruned
    more aggressively or boosted as configured.
- Optional `--anchors_only` mode keeps only original anchors (no offsets
    expanded), but still fits and saves spherical harmonic (SH) coefficients
    up to specified order (0-3).

Usage example:
    python convert_3dgs_ply.py --ply input_scaffold.ply --mlp /path/to/mlp_dir --output /out/dir
"""

import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os
from tqdm import tqdm
import json
from datetime import datetime


def load_scaffold_model(ply_path, mlp_path):
    """Load a Scaffold-GS model (PLY + three TorchScript MLPs).

    Returns a dict containing anchors, per-anchor features, offsets,
    scales, rotations, number of offsets, and the three MLP modules loaded
    and converted to half precision.
    """
    plydata = PlyData.read(ply_path)

    # anchor
    anchor = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1).astype(np.float32)

    # anchor features
    anchor_feat_names = [p.name for p in plydata.elements[0].properties
                         if p.name.startswith("f_anchor_feat")]
    anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split('_')[-1]))
    anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)), dtype=np.float32)
    for idx, attr_name in enumerate(anchor_feat_names):
        anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

    # offsets
    offset_names = [p.name for p in plydata.elements[0].properties
                    if p.name.startswith("f_offset")]
    offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
    offsets = np.zeros((anchor.shape[0], len(offset_names)), dtype=np.float32)
    for idx, attr_name in enumerate(offset_names):
        offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
    n_offsets = len(offset_names) // 3
    offsets = offsets.reshape((offsets.shape[0], 3, n_offsets))

    # scales and rotations
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((anchor.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((anchor.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

    # load TorchScript MLPs
    mlp_color = torch.jit.load(os.path.join(mlp_path, 'color_mlp.pt')).cuda().eval()
    mlp_cov = torch.jit.load(os.path.join(mlp_path, 'cov_mlp.pt')).cuda().eval()
    mlp_opacity = torch.jit.load(os.path.join(mlp_path, 'opacity_mlp.pt')).cuda().eval()

    # force MLPs to use fp16 for inference
    mlp_color.half()
    mlp_cov.half()
    mlp_opacity.half()

    return {
        'anchor': anchor,
        'anchor_feat': anchor_feats,
        'offset': offsets,
        'scale': scales,
        'rotation': rots,
        'n_offsets': n_offsets,
        'mlp_color': mlp_color,
        'mlp_cov': mlp_cov,
        'mlp_opacity': mlp_opacity
    }


def sample_view_directions(n_samples=50, seed=0):
    """Fibonacci sphere sampling of view directions. Returns float16 array.

    Args:
        n_samples: number of directions to sample
        seed: RNG seed for reproducibility
    Returns:
        numpy array of shape [n_samples, 3] dtype float16
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(0, n_samples, dtype=np.float32) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    dirs = np.stack([x, y, z], axis=1).astype(np.float32)
    return dirs.astype(np.float16)


def eval_sh_bases_gpu(deg, dirs_fp16):
    """Evaluate spherical harmonic (SH) basis functions on GPU.

    The input directions are provided in fp16 for bandwidth savings; the
    internal computation is promoted to fp32 for numerical stability and the
    result is returned in fp16.
    """
    dirs = dirs_fp16.to(torch.float32)
    result = []
    result.append(torch.ones(dirs.shape[0], device=dirs.device, dtype=torch.float32) * 0.28209479177387814)
    if deg > 0:
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        result.append(-0.48860251190291992 * y)
        result.append(0.48860251190291992 * z)
        result.append(-0.48860251190291992 * x)
        if deg > 1:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            result.append(1.0925484305920792 * xy)
            result.append(-1.0925484305920792 * yz)
            result.append(0.94617469575755997 * zz - 0.31539156525251999)
            result.append(-1.0925484305920792 * xz)
            result.append(0.54627421529603959 * (xx - yy))
            if deg > 2:
                result.append(0.59004358992664352 * y * (3*xx - yy))
                result.append(2.8906114426405538 * xy * z)
                result.append(0.45704579946446572 * y * (4*zz - xx - yy))
                result.append(0.3731763325901154 * z * (2*zz - 3*xx - 3*yy))
                result.append(0.45704579946446572 * x * (4*zz - xx - yy))
                result.append(1.4453057213202769 * z * (xx - yy))
                result.append(0.59004358992664352 * x * (xx - 3*yy))
    A = torch.stack(result, dim=1)  # fp32
    return A.half()  # convert back to fp16 to save bandwidth


def fit_sh_coefficients_gpu_batch(colors_fp16, view_dirs_fp16, sh_degree=3):
    """
    colors: [M, n_samples, 3] (fp16)
    view_dirs: [n_samples, 3] (fp16)
    """
    # Use fp32 for solving the normal equations (pseudo-inverse) for
    # better numerical stability, even though inputs/outputs are fp16.
    A_fp16 = eval_sh_bases_gpu(sh_degree, view_dirs_fp16)             # [n, C] fp16
    A = A_fp16.to(torch.float32)                                      # fp32
    AtA = A.t() @ A
    AtA_inv = torch.inverse(AtA + 1e-6 * torch.eye(AtA.shape[0], device=AtA.device, dtype=A.dtype))
    At = A.t()

    colors = colors_fp16.to(torch.float32)  # [M, n, 3] -> fp32
    M = colors.shape[0]
    n_coeffs = A.shape[1]
    out = torch.empty((M, 3, n_coeffs), device=colors.device, dtype=torch.float32)
    for c in range(3):
        # [M, n] -> [n, M]
        rhs = colors[:, :, c].t()
        sh_c = AtA_inv @ (At @ rhs)  # [C, M]
        sh_c = sh_c.t()              # [M, C]
        out[:, c, :] = torch.clamp(sh_c, 0.0, 1.0)

    return out.half()  # return fp16


def detect_sky_and_far_anchors(anchor_b, feat_b, n_offsets, mlp_color, sky_blue_delta=0.15,
                               far_radius=None):
    """
    Simple sky / far anchor detection.

    Heuristics:
    - Sky: evaluate color using an upward view [0,0,1]. If the blue channel
      is significantly higher than red/green (B - max(R,G) > sky_blue_delta),
      mark as sky.
    - Far: if the anchor radius exceeds the provided far_radius threshold.

    Returns a boolean mask [B] indicating anchors that should receive
    boosted opacity thresholds (or be treated specially).
    """
    B = anchor_b.shape[0]
    # evaluate color when viewing upward to detect sky-like anchors
    v_up = torch.tensor([[0.0, 0.0, 1.0]], device=anchor_b.device, dtype=torch.float16).repeat(B, 1)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        mlp_in = torch.cat([feat_b.half(), v_up], dim=1)                 # [B, C+3] fp16
        color_up = mlp_color(mlp_in)                                     # [B, 3*k] fp16
    color_up = color_up.reshape(B, n_offsets, 3).float()                 # promote to fp32 for robust stats
    color_up_mean = color_up.mean(dim=1)                                 # [B, 3]
    R, G, Bc = color_up_mean[:, 0], color_up_mean[:, 1], color_up_mean[:, 2]
    sky_mask = (Bc - torch.maximum(R, G)) > sky_blue_delta               # [B]

    if far_radius is not None:
        r = torch.linalg.norm(anchor_b.float(), dim=1)                   # [B]
        far_mask = r > far_radius
        boost_mask = torch.logical_or(sky_mask, far_mask)
    else:
        boost_mask = sky_mask

    return boost_mask


def convert_to_3dgs_pruned(model,
                           output_path,
                           n_view_samples=50,
                           sh_degree=3,
                           anchor_batch_size=128,
                           opacity_thresh=0.20,     # base threshold (0-1, after tanh->prob)
                           per_anchor_topk=4,       # max offsets to keep per anchor (ignored when anchors_only=True)
                           bg_boost=0.15,           # additional threshold boost for sky/far anchors
                           sky_blue_delta=0.15,     # blue - max(R,G) threshold for sky detection
                           far_radius_pct=99.0,     # percentile for far-distance radius threshold (global)
                           anchors_only=False,
                           seed=0):

    anchor = torch.from_numpy(model['anchor']).cuda()
    feat = torch.from_numpy(model['anchor_feat']).cuda()
    offset = torch.from_numpy(model['offset']).cuda()  # [N, 3, k]
    scale = torch.from_numpy(model['scale']).cuda()    # [N, 6]
    n_offsets = model['n_offsets']

    mlp_color = model['mlp_color']     # already half precision
    mlp_cov = model['mlp_cov']         # already half precision
    mlp_opacity = model['mlp_opacity'] # already half precision

    # sample view directions (fp16)
    view_dirs_np = sample_view_directions(n_view_samples, seed=seed)
    view_dirs = torch.from_numpy(view_dirs_np).cuda()  # fp16

    n_anchors = anchor.shape[0]

    # precompute far-distance threshold (percentile over all anchor radii)
    radii = np.linalg.norm(model['anchor'], axis=1)
    far_radius = float(np.percentile(radii, far_radius_pct)) if far_radius_pct is not None else None

    # output buffers (accumulate in-memory, then write once). Internals use
    # fp16 where possible; final arrays are converted to float32 for file output.
    out_xyz = []
    out_dc = []
    out_rest = []
    out_opacity = []
    out_scale = []
    out_rot = []

    with torch.no_grad():
        n_batches = (n_anchors + anchor_batch_size - 1) // anchor_batch_size

        for b in tqdm(range(n_batches), desc="Batches"):
            s = b * anchor_batch_size
            e = min(s + anchor_batch_size, n_anchors)
            B = e - s

            anchor_b = anchor[s:e]
            feat_b = feat[s:e]
            offset_b = offset[s:e]
            scale_b = scale[s:e]

            # compute real positions for each offset (only required if not anchors_only)
            if not anchors_only:
                xyz_bk = (anchor_b.unsqueeze(-1).float() +
                          offset_b.float() * torch.exp(scale_b[:, :3].float()).unsqueeze(-1))  # [B, 3, k]
                xyz_bk = xyz_bk.permute(0, 2, 1).reshape(B * n_offsets, 3)  # [B*k, 3]

            # compute background/sky boost mask ([B])
            boost_mask = detect_sky_and_far_anchors(anchor_b, feat_b, n_offsets, mlp_color,
                                                    sky_blue_delta=sky_blue_delta,
                                                    far_radius=far_radius)

            # coarse pruning using opacity to avoid doing SH fitting for invalid offsets
            feat_rep = feat_b.half().unsqueeze(1).repeat(1, n_view_samples, 1).reshape(B * n_view_samples, -1)  # [B*n, C]
            views_rep = view_dirs.unsqueeze(0).repeat(B, 1, 1).reshape(B * n_view_samples, 3)                   # [B*n, 3]
            with torch.cuda.amp.autocast(dtype=torch.float16):
                mlp_in = torch.cat([feat_rep, views_rep], dim=1)                       # [B*n, C+3] fp16
                opacity_all = torch.tanh(mlp_opacity(mlp_in))                          # [B*n, k] fp16
            opacity_prob = (opacity_all + 1.0) * 0.5                                   # [B*n, k] -> [0,1]
            opacity_prob = opacity_prob.reshape(B, n_view_samples, n_offsets)          # [B, n, k]

            # compute per-anchor / per-offset visibility score
            scores = opacity_prob.max(dim=1).values                                    # [B, k]

            # per-anchor dynamic threshold (increase threshold for sky/far anchors)
            thresh_vec = torch.full((B, 1), opacity_thresh, device=scores.device, dtype=scores.dtype)
            thresh_vec[boost_mask] = opacity_thresh + bg_boost
            keep_by_thresh = scores > thresh_vec                                       # [B, k] bool

            if anchors_only:
                # anchors-only mode: choose one representative offset per anchor
                mean_op = opacity_prob.mean(dim=1)                                     # [B, k]
                best_k = torch.argmax(mean_op, dim=1)                                   # [B]
                # keep only anchors that pass the threshold
                keep_anchor = keep_by_thresh[torch.arange(B, device=keep_by_thresh.device), best_k]  # [B]
                if keep_anchor.sum().item() == 0:
                    continue

                # SH color fitting: predict colors for all offsets and select representative
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    color_all = mlp_color(mlp_in).reshape(B, n_view_samples, n_offsets, 3)  # [B, n, k, 3] fp16
                # sample the best_k color: [B, n, 3]
                batch_idx = torch.arange(B, device=best_k.device)
                colors_selected = color_all[batch_idx, :, best_k, :]                         # [B, n, 3] fp16

                # covariance (scale + rotation): single upward view is sufficient
                v_up = torch.tensor([[0.0, 0.0, 1.0]], device=anchor_b.device, dtype=torch.float16).repeat(B, 1)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    cov_in = torch.cat([feat_b.half(), v_up], dim=1)                         # [B, C+3]
                    scale_rot = mlp_cov(cov_in).reshape(B, n_offsets, 7)                     # [B, k, 7] fp16
                scale_rot_sel = scale_rot[batch_idx, best_k, :].float()                      # [B, 7] fp32

                # position: use anchor itself (do not expand offsets)
                xyz_sel = anchor_b.float()                                                   # [B, 3]
                
                # opacity: mean for the representative offset
                op_sel = mean_op[batch_idx, best_k].float()                                   # [B]

                # fit SH coefficients only for kept anchors
                keep_idx = keep_anchor.nonzero(as_tuple=True)[0]
                colors_kept = colors_selected[keep_idx]                                      # [M, n, 3] fp16
                sh = fit_sh_coefficients_gpu_batch(colors_kept, view_dirs, sh_degree)        # [M, 3, C] fp16
                sh_dc = sh[:, :, 0].float()
                sh_rest = (sh[:, :, 1:].reshape(sh.shape[0], -1).float()
                           if sh_degree > 0 else torch.zeros((sh.shape[0], 0), device=sh.device))

                # final scale: base_scale * predicted scale factor (sigmoid-MLP)
                base_scale = torch.exp(scale_b[:, 3:].float())[keep_idx]                     # [M, 3]
                pred_scale = torch.sigmoid(scale_rot_sel[keep_idx, :3])                      # [M, 3]
                scale_final = base_scale * pred_scale                                        # [M, 3]
                # rot
                rot = scale_rot_sel[keep_idx, 3:7]
                rot = rot / (rot.norm(dim=1, keepdim=True) + 1e-8)
                # opacity
                op_kept = op_sel[keep_idx]                                                    # [M]

                # accumulate results (convert to numpy float32 before writing)
                out_xyz.append(xyz_sel[keep_idx].cpu().numpy().astype(np.float32))
                out_dc.append(sh_dc.cpu().numpy().astype(np.float32))
                out_rest.append(sh_rest.cpu().numpy().astype(np.float32))
                out_opacity.append(op_kept.cpu().numpy().astype(np.float32))
                out_scale.append(scale_final.cpu().numpy().astype(np.float32))
                out_rot.append(rot.cpu().numpy().astype(np.float32))

            else:
                # regular mode: select offsets by threshold and top-K per anchor
                if per_anchor_topk > 0:
                    topk_vals, topk_idx = torch.topk(scores, k=min(per_anchor_topk, n_offsets), dim=1)
                    topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                    topk_mask.scatter_(1, topk_idx, True)
                    keep_mask = keep_by_thresh & topk_mask
                else:
                    keep_mask = keep_by_thresh

                keep_mask_flat = keep_mask.reshape(-1)  # [B*k]
                if keep_mask_flat.sum().item() == 0:
                    continue

                # colors
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    color_all = mlp_color(mlp_in).reshape(B, n_view_samples, n_offsets, 3)   # [B, n, k, 3] fp16
                colors_flat = color_all.permute(0, 2, 1, 3).reshape(B * n_offsets, n_view_samples, 3)  # [B*k, n, 3] fp16
                colors_kept = colors_flat[keep_mask_flat]                                                # [M, n, 3]

                # covariance (scale + rotation): single upward view
                v_up = torch.tensor([[0.0, 0.0, 1.0]], device=anchor_b.device, dtype=torch.float16).repeat(B, 1)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    cov_in = torch.cat([feat_b.half(), v_up], dim=1)                        # [B, C+3]
                    scale_rot_all = mlp_cov(cov_in).reshape(B, n_offsets, 7)                # [B, k, 7] fp16
                scale_rot_flat = scale_rot_all.reshape(B * n_offsets, 7)[keep_mask_flat].float()  # [M, 7]

                # fit SH for the kept offsets
                sh = fit_sh_coefficients_gpu_batch(colors_kept, view_dirs, sh_degree)        # [M, 3, C] fp16
                sh_dc = sh[:, :, 0].float()
                sh_rest = (sh[:, :, 1:].reshape(sh.shape[0], -1).float()
                           if sh_degree > 0 else torch.zeros((sh.shape[0], 0), device=sh.device))

                # position / opacity / scale / rotation
                xyz_kept = xyz_bk[keep_mask_flat].float()                                    # [M, 3]
                op_kept = opacity_prob.mean(dim=1).reshape(B * n_offsets)[keep_mask_flat].float()
                base_scale = torch.exp(scale_b[:, 3:].float()).unsqueeze(1).repeat(1, n_offsets, 1).reshape(B * n_offsets, 3)
                base_scale = base_scale[keep_mask_flat]
                pred_scale = torch.sigmoid(scale_rot_flat[:, :3])
                scale_final = base_scale * pred_scale
                rot = scale_rot_flat[:, 3:7]
                rot = rot / (rot.norm(dim=1, keepdim=True) + 1e-8)

                # accumulate (convert to numpy float32 for output)
                out_xyz.append(xyz_kept.cpu().numpy().astype(np.float32))
                out_dc.append(sh_dc.cpu().numpy().astype(np.float32))
                out_rest.append(sh_rest.cpu().numpy().astype(np.float32))
                out_opacity.append(op_kept.cpu().numpy().astype(np.float32))
                out_scale.append(scale_final.cpu().numpy().astype(np.float32))
                out_rot.append(rot.cpu().numpy().astype(np.float32))

            if b % 20 == 0:
                # periodically clear CUDA cache to avoid fragmentation
                torch.cuda.empty_cache()

    # concatenate accumulated outputs
    if len(out_xyz) == 0:
        raise RuntimeError("All points were pruned. Check thresholds/parameters; they may be too strict.")
    xyz = np.concatenate(out_xyz, axis=0).astype(np.float32)
    dc = np.concatenate(out_dc, axis=0).astype(np.float32)
    rest = (np.concatenate(out_rest, axis=0).astype(np.float32)
            if out_rest[0].shape[1] > 0 else np.zeros((xyz.shape[0], 0), dtype=np.float32))
    opacity = np.concatenate(out_opacity, axis=0).astype(np.float32)
    scale_final = np.concatenate(out_scale, axis=0).astype(np.float32)
    rot = np.concatenate(out_rot, axis=0).astype(np.float32)

    print(f"After pruning: {xyz.shape[0]:,} gaussians")
    save_3dgs_ply(xyz, dc, rest, opacity, scale_final, rot, output_path, sh_degree)


def save_metadata(metadata, path):
    """Save metadata to a JSON file."""
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)


def save_3dgs_ply(xyz, sh_dc, sh_rest, opacity, scale, rotation, path, sh_degree):
    n = xyz.shape[0]
    attrs = ['x', 'y', 'z']
    for i in range(3):
        attrs.append(f'f_dc_{i}')
    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    for i in range(n_rest):
        attrs.append(f'f_rest_{i}')
    attrs.append('opacity')
    for i in range(3):
        attrs.append(f'scale_{i}')
    for i in range(4):
        attrs.append(f'rot_{i}')

    dtype_full = [(a, 'f4') for a in attrs]
    elements = np.empty(n, dtype=dtype_full)

    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['f_dc_0'] = sh_dc[:, 0]
    elements['f_dc_1'] = sh_dc[:, 1]
    elements['f_dc_2'] = sh_dc[:, 2]

    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    for i in range(n_rest):
        elements[f'f_rest_{i}'] = sh_rest[:, i] if sh_rest.shape[1] > 0 else 0.0

    elements['opacity'] = opacity
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]
    elements['rot_0'] = rotation[:, 0]
    elements['rot_1'] = rotation[:, 1]
    elements['rot_2'] = rotation[:, 2]
    elements['rot_3'] = rotation[:, 3]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Scaffold-GS model (PLY + TorchScript MLPs) into a standard 3DGS PLY. "
            "The script prunes Gaussians, fits spherical harmonics for color, computes "
            "covariance/rotation/opacity per Gaussian, and writes a PLY and metadata.json."
        )
    )

    parser.add_argument("--ply", required=True, help="Path to the input Scaffold-GS PLY file.")
    parser.add_argument("--mlp", required=True, help="Directory containing TorchScript MLPs: color_mlp.pt, cov_mlp.pt, opacity_mlp.pt")
    parser.add_argument("--output", required=True, help="Output directory. A timestamped subdirectory will be created inside this directory to store outputs.")

    parser.add_argument("--n_samples", type=int, default=50, help="Number of view samples for SH fitting (default: 50)")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonic degree (0-3)")
    parser.add_argument("--anchor_batch_size", type=int, default=128, help="Anchor batch size used for GPU processing")

    # pruning parameters
    parser.add_argument("--opacity_thresh", type=float, default=0.20, help="Base opacity threshold (0-1) applied after tanh->prob")
    parser.add_argument("--per_anchor_topk", type=int, default=4, help="Per-anchor top-K offsets to keep (ignored in anchors_only mode)")

    # background/sky boosting
    parser.add_argument("--bg_boost", type=float, default=0.15, help="Additional threshold added for anchors detected as background/sky")
    parser.add_argument("--sky_blue_delta", type=float, default=0.15, help="Blue minus max(R,G) delta for sky detection")
    parser.add_argument("--far_radius_pct", type=float, default=99.0, help="Percentile used to compute global far-distance radius")

    # anchors-only mode
    parser.add_argument("--anchors_only", action="store_true", help="Only keep original anchors (do not expand offsets); SH still fitted per anchor")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for view sampling")

    args = parser.parse_args()

    # Prepare output directory: args.output is expected to be a directory. Create
    # a timestamped subdirectory inside it to hold the converted files.
    out_base = args.output
    if os.path.exists(out_base) and not os.path.isdir(out_base):
        raise ValueError(f"--output must be a directory path, not a file: {out_base}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.abspath(out_base), f"3DGS_PLY_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # construct output PLY path using the basename of the input PLY
    ply_output_path = os.path.join(output_dir, os.path.basename(args.ply))
    json_output_path = os.path.join(output_dir, "metadata.json")

    model = load_scaffold_model(args.ply, args.mlp)
    convert_to_3dgs_pruned(
        model,
        output_path=ply_output_path,
        n_view_samples=args.n_samples,
        sh_degree=args.sh_degree,
        anchor_batch_size=args.anchor_batch_size,
        opacity_thresh=args.opacity_thresh,
        per_anchor_topk=args.per_anchor_topk,
        bg_boost=args.bg_boost,
        sky_blue_delta=args.sky_blue_delta,
        far_radius_pct=args.far_radius_pct,
        anchors_only=args.anchors_only,
        seed=args.seed
    )

    # save metadata
    metadata = {
        "ply_path": args.ply,
        "mlp_path": args.mlp,
        "output_ply": ply_output_path,
        "n_samples": args.n_samples,
        "sh_degree": args.sh_degree,
        "anchor_batch_size": args.anchor_batch_size,
        "opacity_thresh": args.opacity_thresh,
        "per_anchor_topk": args.per_anchor_topk,
        "bg_boost": args.bg_boost,
        "sky_blue_delta": args.sky_blue_delta,
        "far_radius_pct": args.far_radius_pct,
        "anchors_only": args.anchors_only,
        "seed": args.seed,
        "timestamp": timestamp
    }
    save_metadata(metadata, json_output_path)