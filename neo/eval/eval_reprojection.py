import os
import json
import torch
import cv2
import tyro
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from kornia.feature import LoFTR
from kornia.color import rgb_to_grayscale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def robust_filter_errors(err: torch.Tensor, method="mad", thresh=3.0):
    """
    Remove outliers from reprojection error.

    method:
        - "mad": Median Absolute Deviation (best)
        - "percentile": clip top X%
    """
    if len(err) < 10:
        return err  # not enough data

    if method == "mad":
        median = torch.median(err)
        abs_dev = torch.abs(err - median)
        mad = torch.median(abs_dev) + 1e-6

        robust_mask = abs_dev < thresh * mad
        return err[robust_mask]

    elif method == "percentile":
        cutoff = torch.quantile(err, 0.95)  # keep best 95%
        return err[err < cutoff]

    else:
        return err

def unproject_points(points_2d, depth, fl_x, fl_y, cx, cy):
    H, W = depth.shape[-2], depth.shape[-1]

    u = points_2d[:, 0]
    v = points_2d[:, 1]

    # Bilinear sampling
    grid_x = (u / (W - 1)) * 2 - 1
    grid_y = (v / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)

    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)

    r = torch.nn.functional.grid_sample(
        depth, grid, align_corners=True, mode="bilinear"
    ).view(-1)

    # --- KEY PART: convert range -> z-depth ---
    x_norm = (u - cx) / fl_x
    y_norm = (v - cy) / fl_y

    ray_norm = torch.sqrt(x_norm**2 + y_norm**2 + 1.0)

    z = r / ray_norm

    x = x_norm * z
    y = y_norm * z

    pts_3d = torch.stack([x, y, z], dim=-1)
    return pts_3d, r > 1e-6

def calculate_geometric_reprojection_error(
    data_path: Optional[Path] = None, 
    config_path: Optional[Path] = None,
    eval_exp_name: str = "full",
    step: int = 20, 
    confidence_thresh: float = 0.8,
    matcher_type: str = "loftr",
    debug_limit: Optional[int] = None,
    debug_out_dir: Optional[Path] = None
):
    """
    Computes geometric reprojection error between frame i and frame i+step.
    Will use data_path if provided directly. If config_path is provided, it reads the 
    yaml config, infers the ground-truth "after" data_path for transforms.json, 
    and reads testing rendering images from the output folders.
    """
    is_rendered = False
    render_output_path = None
    
    if config_path is not None:
        import yaml
        from nerfstudio.engine.trainer import TrainerConfig
        
        # Load edited model config
        model_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
        
        base_path = str(model_config.get_base_dir())
        if model_config.method_name == "replace-nerf":
            base_path = str(model_config.get_base_dir().parent)

        if "lerf-lite" in base_path:
            gt_data_path = Path(base_path.replace('outputs', 'data')).parents[1]
        else:
            gt_data_path = Path(base_path.replace('outputs', 'data').replace('before', 'after')).parents[1]

        render_output_path = Path(base_path.replace('outputs', 'renders'))
        data_path = gt_data_path
        is_rendered = True
        print(f"Using config! GT transforms path: {data_path}")
        print(f"Renderings path: {render_output_path} (eval_exp_name={eval_exp_name})")
    
    if data_path is None:
        raise ValueError("Either --data-path or --config-path must be specified.")

    if config_path is not None:
        parts = config_path.parent.parts
        rel_path = Path(*parts[1:]) if parts and parts[0] == "outputs" else config_path.parent
        eval_reproj_dir = Path("eval_reproj") / rel_path
    else:
        parts = data_path.parts
        rel_path = Path(*parts[1:]) if parts and parts[0] == "data" else data_path
        eval_reproj_dir = Path("eval_reproj") / rel_path / "dataset"

    transforms_path = data_path / 'transforms.json'
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing {transforms_path}")

    with open(transforms_path, 'r') as f:
        meta = json.load(f)

    # Extract intrinsics
    frames = meta['frames']
    fl_x = frames[0].get('fl_x', meta.get('fl_x'))
    fl_y = frames[0].get('fl_y', meta.get('fl_y'))
    cx = frames[0].get('cx', meta.get('cx'))
    cy = frames[0].get('cy', meta.get('cy'))

    # --- NEW: camera matrix ---
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # --- NEW: distortion coefficients ---
    k1 = meta.get("k1", 0.0)
    k2 = meta.get("k2", 0.0)
    k3 = meta.get("k3", 0.0)
    p1 = meta.get("p1", 0.0)
    p2 = meta.get("p2", 0.0)

    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    has_distortion = np.any(np.abs(dist_coeffs) > 1e-8)

    matcher = LoFTR(pretrained="indoor").to(device)

    total_error = 0.0
    total_matches = 0

    print(f"Evaluating {len(frames) - step} pairs with step {step}...")
    pair_results = []

    for i in tqdm(range(len(frames) - step)):
        if debug_limit is not None and i >= debug_limit:
            break

        f1 = frames[i]
        f2 = frames[i + step]

        mask1_path = None
        if is_rendered and render_output_path is not None:
            file_name1 = Path(f1['file_path']).name
            file_name2 = Path(f2['file_path']).name
            
            img1_path = render_output_path / f"{eval_exp_name}_edited" / "rgb" / file_name1
            img2_path = render_output_path / f"{eval_exp_name}_edited" / "rgb" / file_name2
            
            # Depth maps map to the same name
            depth1_path = render_output_path / f"{eval_exp_name}_edited" / "depth" / file_name1
            
            # Masks
            mask1_path = render_output_path / f"{eval_exp_name}_masks" / "temp_masks" / file_name1
        else:
            img1_path = data_path / f1['file_path']
            img2_path = data_path / f2['file_path']
            # Replace images/ with depth/ extension or adapt based on dataset layout
            depth1_path = data_path / str(f1['file_path']).replace('images', 'depth').replace('rgb', 'depth')
            
        if not img1_path.exists() or not img2_path.exists() or not depth1_path.exists():
            continue

        # Load RGB
        img1_uint8 = cv2.imread(str(img1_path))
        img2_uint8 = cv2.imread(str(img2_path))
        
        img1 = torch.tensor(img1_uint8, device=device).float() / 255.0
        img2 = torch.tensor(img2_uint8, device=device).float() / 255.0
        
        img1 = img1.permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
        img2 = img2.permute(2, 0, 1).unsqueeze(0)
        
        # Load Depth
        # Realsense captures depth in millimeters usually. Scale by 1000.
        depth1 = torch.tensor(np.int64(cv2.imread(str(depth1_path), cv2.IMREAD_ANYDEPTH)), device=device).float() / 1000.0

        # Load Mask
        mask1 = None
        if mask1_path is not None and mask1_path.exists():
            mask1 = torch.tensor(cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE), device=device) / 255.0
        
        c2w_1 = torch.tensor(f1['transform_matrix'], device=device).float()
        c2w_2 = torch.tensor(f2['transform_matrix'], device=device).float()
        
        # Invert c2w_2 to get w2c_2
        w2c_2 = torch.linalg.inv(c2w_2)

        if matcher_type == "orb":
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1_uint8, None)
            kp2, des2 = orb.detectAndCompute(img2_uint8, None)
            
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                continue
                
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) == 0:
                continue
                
            kpts1 = torch.tensor(np.float32([kp1[m.queryIdx].pt for m in good_matches]), device=device)
            kpts2 = torch.tensor(np.float32([kp2[m.trainIdx].pt for m in good_matches]), device=device)

            if has_distortion:
                # Convert to numpy for OpenCV
                kpts1_np = kpts1.detach().cpu().numpy().reshape(-1, 1, 2)
                kpts2_np = kpts2.detach().cpu().numpy().reshape(-1, 1, 2)

                K_cv = K.copy()

                E, mask = cv2.findEssentialMat(
                    kpts1_np, kpts2_np,
                    K_cv,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )

                if mask is not None:
                    mask = torch.tensor(mask.squeeze().astype(bool), device=device)
                    kpts1 = kpts1[mask]
                    kpts2 = kpts2[mask]

                # Undistort into pixel coordinates (P=K keeps pixel space)
                kpts1_undist = cv2.undistortPoints(kpts1_np, K, dist_coeffs, P=K).reshape(-1, 2)
                kpts2_undist = cv2.undistortPoints(kpts2_np, K, dist_coeffs, P=K).reshape(-1, 2)

                # Back to torch
                kpts1 = torch.tensor(kpts1_undist, device=device, dtype=torch.float32)
                kpts2 = torch.tensor(kpts2_undist, device=device, dtype=torch.float32)
            
        elif matcher_type == "loftr":
            with torch.no_grad():
                img1_gray = rgb_to_grayscale(img1)
                img2_gray = rgb_to_grayscale(img2)
                
                input_dict = {"image0": img1_gray, "image1": img2_gray}
                correspondences = matcher(input_dict)
                
                valid_corrs = correspondences["confidence"] >= confidence_thresh
                kpts1 = correspondences["keypoints0"][valid_corrs] # (N, 2) in image 1
                kpts2 = correspondences["keypoints1"][valid_corrs] # (N, 2) in image 2

                if len(kpts1) >= 8:  # minimum for E
                    kpts1_np = kpts1.detach().cpu().numpy()
                    kpts2_np = kpts2.detach().cpu().numpy()

                    E, mask = cv2.findEssentialMat(
                        kpts1_np, kpts2_np,
                        K,
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=1.0
                    )

                    if mask is not None:
                        mask = torch.tensor(mask.squeeze().astype(bool), device=device)
                        kpts1 = kpts1[mask]
                        kpts2 = kpts2[mask]

        else:
            raise ValueError(f"Unknown matcher_type: {matcher_type}")
            
        if len(kpts1) == 0:
            continue
                
        # Unproject kpts1 using depth1
        pts_3d_cam1, valid_depth = unproject_points(kpts1, depth1, fl_x, fl_y, cx, cy)
        
        # Filter matches with missing depth
        kpts1 = kpts1[valid_depth]
        kpts2 = kpts2[valid_depth]
        pts_3d_cam1 = pts_3d_cam1[valid_depth]
        
        if len(pts_3d_cam1) == 0:
            continue

        # Convert local camera OpenCV unprojection (Y down, Z forward) to OpenGL (Y up, Z back) 
        # to match NerfStudio transforms.json poses!
        pts_3d_cam1[:, 1] *= -1.0
        pts_3d_cam1[:, 2] *= -1.0

        # Convert to homogeneous coords
        N = pts_3d_cam1.shape[0]
        pts_3d_cam1_h = torch.cat([pts_3d_cam1, torch.ones((N, 1), device=device)], dim=-1)
        
        # Transform from cam1 -> world -> cam2
        # Assume NeRF / OpenGL coords if loaded from transforms.json (c2w applies directly to cam points)
        pts_3d_world = (c2w_1 @ pts_3d_cam1_h.T).T
        pts_3d_cam2 = (w2c_2 @ pts_3d_world.T).T
        
        # Convert local cam2 back from OpenGL to OpenCV for image projection
        pts_3d_cam2[:, 1] *= -1.0
        pts_3d_cam2[:, 2] *= -1.0

        # Project onto cam2 image plane
        z2 = pts_3d_cam2[:, 2]
        
        # Extract masks from kpts1
        if mask1 is not None:
            # Gather mask values at kpts1 float positions
            u1_idx = torch.clamp(torch.round(kpts1[:, 0]).long(), 0, mask1.shape[1] - 1)
            v1_idx = torch.clamp(torch.round(kpts1[:, 1]).long(), 0, mask1.shape[0] - 1)
            kpts1_is_masked = mask1[v1_idx, u1_idx] > 0.5
        else:
            kpts1_is_masked = torch.zeros(kpts1.shape[0], dtype=torch.bool, device=device)

        # Filter points behind camera
        valid_z = z2 > 0.01
        pts_3d_cam2 = pts_3d_cam2[valid_z]
        kpts2 = kpts2[valid_z]
        kpts1_is_masked = kpts1_is_masked[valid_z]
        z2 = z2[valid_z]
        
        if len(pts_3d_cam2) == 0:
            continue
            
        u2_proj = (pts_3d_cam2[:, 0] * fl_x / z2) + cx
        v2_proj = (pts_3d_cam2[:, 1] * fl_y / z2) + cy
        kpts2_proj = torch.stack([u2_proj, v2_proj], dim=-1)
        
        # Calculate pixel error
        err = torch.linalg.norm(kpts2_proj - kpts2, dim=-1)
        err_filtered = robust_filter_errors(err, method="mad", thresh=3.0)

        # Skip completely broken pairs (checkerboard failure case)
        if len(err_filtered) < 20:
            continue

        # Store only filtered errors
        pair_errors = err_filtered.cpu().numpy().tolist()

        # Mask split ALSO on filtered set
        masked_err_vals = err[kpts1_is_masked]
        unmasked_err_vals = err[~kpts1_is_masked]

        masked_err_vals = robust_filter_errors(masked_err_vals)
        unmasked_err_vals = robust_filter_errors(unmasked_err_vals)

        pair_errors_masked = masked_err_vals.cpu().numpy().tolist() if len(masked_err_vals) > 0 else []
        pair_errors_unmasked = unmasked_err_vals.cpu().numpy().tolist() if len(unmasked_err_vals) > 0 else []
        
        pair_results.append({
            'mean_err': np.mean(pair_errors),
            'errors': pair_errors,
            'errors_masked': pair_errors_masked,
            'errors_unmasked': pair_errors_unmasked
        })

        # Debug Visualization
        if debug_out_dir is not None:
            os.makedirs(debug_out_dir, exist_ok=True)
            
            np_kpts2 = kpts2.cpu().numpy()
            np_kpts2_proj = kpts2_proj.cpu().numpy()
            mask_np = kpts1_is_masked.cpu().numpy()
            
            def save_debug_img(kpts_tgt, kpts_proj, suffix):
                dimg = img2_uint8.copy()
                for pt_tgt, pt_proj in zip(kpts_tgt, kpts_proj):
                    pt_tgt_tuple = (int(pt_tgt[0]), int(pt_tgt[1]))
                    pt_proj_tuple = (int(pt_proj[0]), int(pt_proj[1]))
                    
                    # Original match in Green
                    cv2.circle(dimg, pt_tgt_tuple, 3, (0, 255, 0), -1)
                    # Reprojected in Red
                    cv2.circle(dimg, pt_proj_tuple, 3, (0, 0, 255), -1)
                    # Line connecting them in Yellow
                    cv2.line(dimg, pt_tgt_tuple, pt_proj_tuple, (0, 255, 255), 1)
                    
                cv2.imwrite(str(debug_out_dir / f"debug_reproj_{suffix}_{i:04d}.jpg"), dimg)
                
            save_debug_img(np_kpts2, np_kpts2_proj, "all")
            if mask_np.sum() > 0:
                save_debug_img(np_kpts2[mask_np], np_kpts2_proj[mask_np], "masked")
            if (~mask_np).sum() > 0:
                save_debug_img(np_kpts2[~mask_np], np_kpts2_proj[~mask_np], "unmasked")

    if len(pair_results) > 0:
        # Select 20 pairs with the smallest mean reprojection score
        pair_results.sort(key=lambda x: x['mean_err'])
        best_pairs = pair_results[:20]
        
        errors = []
        errors_masked = []
        errors_unmasked = []
        for p in best_pairs:
            errors.extend(p['errors'])
            errors_masked.extend(p['errors_masked'])
            errors_unmasked.extend(p['errors_unmasked'])
            
        if len(errors) == 0:
            print("No valid matches found or missing depth maps.")
            return

        errors_np = np.array(errors)
        median_err = np.median(errors_np)

        # trimmed mean (ignore top 10%)
        p90 = np.percentile(errors_np, 90)
        trimmed = errors_np[errors_np < p90]
        mean_err = np.mean(trimmed)

        print(f"\nResults (Top {len(best_pairs)} pairs with smallest error):")
        print(f"Total valid matches evaluated: {len(errors)}")
        print(f"Mean Reprojection Error: {mean_err:.4f} pixels")
        print(f"Median Reprojection Error: {median_err:.4f} pixels")
        
        if len(errors_masked) > 0:
            mean_err_masked = np.mean(errors_masked)
            median_err_masked = np.median(errors_masked)
            print(f"\nMasked Region (Matches: {len(errors_masked)}):")
            print(f"Mean Reprojection Error: {mean_err_masked:.4f} pixels")
            print(f"Median Reprojection Error: {median_err_masked:.4f} pixels")
            
        if len(errors_unmasked) > 0:
            mean_err_unmasked = np.mean(errors_unmasked)
            median_err_unmasked = np.median(errors_unmasked)
            print(f"\nUnmasked Region (Matches: {len(errors_unmasked)}):")
            print(f"Mean Reprojection Error: {mean_err_unmasked:.4f} pixels")
            print(f"Median Reprojection Error: {median_err_unmasked:.4f} pixels")
            
        os.makedirs(eval_reproj_dir, exist_ok=True)
        results_dict = {
            "total_valid_pairs_evaluated": len(best_pairs),
            "total_valid_matches": len(errors),
            "step": step,
            "matcher_type": matcher_type,
            "eval_exp_name": eval_exp_name if is_rendered else "dataset",
            "all": {
                "mean_error": float(mean_err),
                "median_error": float(median_err)
            }
        }
        if len(errors_masked) > 0:
            results_dict["masked"] = {
                "matches": len(errors_masked),
                "mean_error": float(mean_err_masked),
                "median_error": float(median_err_masked)
            }
        if len(errors_unmasked) > 0:
            results_dict["unmasked"] = {
                "matches": len(errors_unmasked),
                "mean_error": float(mean_err_unmasked),
                "median_error": float(median_err_unmasked)
            }
            
        out_json_path = eval_reproj_dir / f"reprojection_results_{eval_exp_name}.json"
        with open(out_json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"\nSaved results to {out_json_path}")
            
    else:
        print("No valid matches found or missing depth maps.")

if __name__ == "__main__":
    tyro.cli(calculate_geometric_reprojection_error)
