import torch
from typing import Optional, Callable, Tuple, List, Any
from jaxtyping import Float

from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, PDFSampler, Sampler
from nerfstudio.cameras.rays import RayBundle, RaySamples

from neo.utils.bbox import BoundingBox


class DualProposalNetworkSampler(ProposalNetworkSampler):

    def __init__(
            self, 
            num_proposal_samples_per_ray: Tuple[int, ...] = (64, ), 
            num_nerf_samples_per_ray: int = 32, 
            num_proposal_network_iterations: int = 2, 
            single_jitter: bool = False, 
            update_sched: Callable[..., Any] = lambda x: 1, 
            initial_sampler: Optional[Sampler] = None, 
            pdf_sampler: Optional[PDFSampler] = None
        ) -> None:
        super().__init__(
            num_proposal_samples_per_ray, 
            num_nerf_samples_per_ray, 
            num_proposal_network_iterations, 
            single_jitter, 
            update_sched, 
            initial_sampler, 
            pdf_sampler
        )

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
        replace_density_fns: Optional[List[Callable]] = None,
        bboxes: Optional[List[BoundingBox]] = None,
        offset: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None
        assert replace_density_fns is not None
        assert bboxes is not None
        assert offset is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)

            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    ray_positions = ray_samples.frustums.get_positions()
                    density = density_fns[i_level](ray_positions)
                    replace_density = replace_density_fns[i_level](ray_positions)

                    # Remove object
                    masks_copy = None
                    for bbox in bboxes:
                        mask_copy = bbox.are_points_inside(ray_positions, offset=offset)

                        if masks_copy is None:
                            masks_copy = mask_copy
                        else:
                            masks_copy = torch.logical_or(masks_copy, mask_copy)

                    density = torch.where(masks_copy, replace_density, density)
                    
                else:
                    with torch.no_grad():
                        ray_positions = ray_samples.frustums.get_positions()
                        density = density_fns[i_level](ray_positions)
                        replace_density = replace_density_fns[i_level](ray_positions)

                        # Remove object
                        masks_copy = None
                        for bbox in bboxes:
                            mask_copy = bbox.are_points_inside(ray_positions, offset=offset)

                            if masks_copy is None:
                                masks_copy = mask_copy
                            else:
                                masks_copy = torch.logical_or(masks_copy, mask_copy)

                        density = torch.where(masks_copy, replace_density, density)

                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
    

class ReconfigurationalProposalNetworkSampler(ProposalNetworkSampler):

    def __init__(
            self, 
            num_proposal_samples_per_ray: Tuple[int, ...] = (64, ), 
            num_nerf_samples_per_ray: int = 32, 
            num_proposal_network_iterations: int = 2, 
            single_jitter: bool = False, 
            update_sched: Callable[..., Any] = lambda x: 1, 
            initial_sampler: Optional[Sampler] = None, 
            pdf_sampler: Optional[PDFSampler] = None
        ) -> None:
        super().__init__(
            num_proposal_samples_per_ray, 
            num_nerf_samples_per_ray, 
            num_proposal_network_iterations, 
            single_jitter, 
            update_sched, 
            initial_sampler, 
            pdf_sampler
        )

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
        replace_density_fns: Optional[List[Callable]] = None,
        bboxes: Optional[List[BoundingBox]] = None,
        offset: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None
        assert replace_density_fns is not None
        assert bboxes is not None
        assert offset is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)

            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    ray_positions = ray_samples.frustums.get_positions()
                    density = density_fns[i_level](ray_positions)
                    replace_density = replace_density_fns[i_level](ray_positions + torch.tensor([-0.3, 0.0, 0.0], device=density.device))

                    # Remove object
                    masks_copy = None
                    for bbox in bboxes:
                        mask_copy_old = bbox.are_points_inside(ray_positions, offset=offset)
                        mask_copy = bbox.are_points_inside_reconfigured(ray_positions, offset=offset)
                        mask_copy = torch.logical_or(mask_copy_old, mask_copy)

                        if masks_copy is None:
                            masks_copy = mask_copy
                        else:
                            masks_copy = torch.logical_or(masks_copy, mask_copy)

                    density = torch.where(masks_copy, replace_density, density)
                    
                else:
                    with torch.no_grad():
                        ray_positions = ray_samples.frustums.get_positions()
                        density = density_fns[i_level](ray_positions)
                        replace_density = replace_density_fns[i_level](ray_positions + torch.tensor([-0.3, 0.0, 0.0], device=density.device))

                        # Remove object
                        masks_copy = None
                        for bbox in bboxes:
                            mask_copy_old = bbox.are_points_inside(ray_positions, offset=offset)
                            mask_copy = bbox.are_points_inside_reconfigured(ray_positions, offset=offset)
                            masks_copy = torch.logical_or(mask_copy_old, mask_copy)

                            if masks_copy is None:
                                masks_copy = mask_copy
                            else:
                                masks_copy = torch.logical_or(masks_copy, mask_copy)

                        density = torch.where(masks_copy, replace_density, density)

                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
    

class RegionAwareProposalNetworkSampler(ProposalNetworkSampler):
    """Region aware proposal network sampler that allows for excluding regions along the ray from the sampling process.

    Args:
        ray_bundle: Rays to generate samples for
        density_fns: Sampler network
        ray_masks: Bounds for each ray specifying the regions excluded from the sampling
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
        ray_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray

            if ray_masks != None:
                ray_masks_copy = torch.clone(ray_masks)

            if i_level == 0:
                if ray_masks == None:
                    # Uniform sampling because we need to start with some samples
                    ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
           
                else:
                    if ray_masks.shape[1] < num_samples:
                        temp = torch.zeros((ray_masks.shape[0], num_samples, 1)).to(ray_masks.device)
                        temp[:, :ray_masks.shape[1], :] = ray_masks
                        ray_masks_copy = temp
                    else:
                        ray_masks_copy = ray_masks_copy[:, :num_samples ,:]

                    ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples, ray_masks=ray_masks_copy)

            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)

                if ray_masks == None:
                    ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)

                else:
                    if ray_masks.shape[1] < num_samples:
                        temp = torch.zeros((ray_masks.shape[0], num_samples, 1)).to(ray_masks.device)
                        temp[:, :ray_masks.shape[1], :] = ray_masks
                        ray_masks_copy = temp
                    else:
                        ray_masks_copy = ray_masks_copy[:, :num_samples ,:]
                        
                    ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples, ray_masks=ray_masks_copy)

            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.frustums.get_positions())

                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.frustums.get_positions())
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
    

class RegionAwareSpacedSampler(Sampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        ray_masks: Optional[torch.Tensor] = None,
    ) -> RaySamples:
        """Generates position samples according to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        assert ray_bundle is not None
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.origins.shape[0]

        bins = None
        starting_positions = None
        if ray_masks == None:
            bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

        else:
            ray_masks = self.spacing_fn(ray_masks[..., 0])

            zero_tensor = torch.zeros(ray_masks.shape[0])[..., None].to(ray_masks.device)
            ray_masks = torch.torch.cat((ray_masks, zero_tensor), dim=1)

            bin_mask_raise = torch.where(torch.diff(torch.where(ray_masks > 0, 1, 0), prepend=zero_tensor) > 0, 1, 0)
            bin_mask_fall = torch.where(torch.diff(torch.where(ray_masks > 0, 1, 0), append=zero_tensor) < 0, 1, 0)

            end = (1 - (torch.sum(ray_masks * bin_mask_fall, dim=1) - torch.sum(ray_masks * bin_mask_raise, dim=1)))[..., None]
            multidim_arange = torch.cumsum(torch.ones(ray_masks.shape).to(ray_masks.device), dim=1) - 1
            bins = multidim_arange * end / ray_masks.shape[1]

            raising_values = torch.roll(torch.sort(torch.where(bin_mask_raise * ray_masks > 0, bin_mask_raise * ray_masks, torch.inf)).values, -1)
            falling_values = torch.sort(torch.where(bin_mask_fall * ray_masks > 0, bin_mask_fall * ray_masks, torch.inf)).values

            increment_step = end / (ray_masks.shape[1] - 1)
            move_idxs = torch.floor(((bins - ray_masks) / (increment_step)) * torch.where(ray_masks > 0, 1, 0))

            closest_idxs = (multidim_arange - move_idxs).int()

            sort_start = torch.sort(closest_idxs * bin_mask_raise)[0]
            sort_end = torch.sort(closest_idxs * bin_mask_fall)[0]

            inserted_zero = torch.torch.cat((zero_tensor, sort_end - sort_start), dim=1)[:, :-1]
            starting_idxs = (sort_start - torch.cumsum(inserted_zero, dim=1)).int()
            starting_idxs = torch.where(starting_idxs >= ray_masks.shape[1], ray_masks.shape[1] - 1, starting_idxs)
            starting_positions = torch.zeros(ray_masks.shape).to(ray_masks.device)
            starting_positions[torch.arange(ray_masks.shape[0])[..., None], starting_idxs] = (sort_end - sort_start).float()
            bins = (multidim_arange + torch.cumsum(starting_positions, dim=1)) * increment_step

            # Comment this block of code for preview
            temp_bins = torch.sort(torch.concat([bins, raising_values, falling_values], dim=1))
            temp_starting_positions = torch.concat([starting_positions, torch.zeros(starting_positions.shape).to(ray_masks.device), torch.zeros(starting_positions.shape).to(ray_masks.device)], dim=1)[torch.arange(ray_masks.shape[0])[..., None], temp_bins.indices]
            bins = temp_bins.values[:, :ray_masks.shape[1]]
            starting_positions = temp_starting_positions[:, :ray_masks.shape[1]]
            bins[:, -1] = 1

        # TODO More complicated than it needs to be.
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)

            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears, ray_bundle.fars))

        def spacing_to_euclidean_fn(x):
            return self.spacing_fn_inv(x * s_far + (1 - x) * s_near)

        bin_starts = None
        bin_ends = None
        spacing_starts = None
        spacing_ends = None
        ray_samples = None

        if ray_masks == None:
            euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples+1]
            
            bin_starts = euclidean_bins[..., :-1, None]
            bin_ends = euclidean_bins[..., 1:, None]
            spacing_starts = bins[..., :-1, None]
            spacing_ends = bins[..., 1:, None]

            ray_samples = ray_bundle.get_ray_samples(
                bin_starts=bin_starts,
                bin_ends=bin_ends,
                spacing_starts=spacing_starts,
                spacing_ends=spacing_ends,
                spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            )

        else:
            euclidean_bins = spacing_to_euclidean_fn(bins)
            bin_mask_start = torch.roll(torch.where(starting_positions > 0, 1, 0), -2)[:, :-1, None]

            bin_starts = euclidean_bins[..., :-1, None]
            bin_ends = (euclidean_bins[..., 1:, None] * (1 - bin_mask_start)) + ((bin_starts + 1e-9) * bin_mask_start)
            spacing_starts = bins[..., :-1, None]
            spacing_ends = (bins[..., 1:, None] * (1 - bin_mask_start)) + ((spacing_starts + 1e-9) * bin_mask_start)

            ray_samples = ray_bundle.get_ray_samples(
                bin_starts=bin_starts,
                bin_ends=bin_ends,
                spacing_starts=spacing_starts,
                spacing_ends=spacing_ends,
                spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            )

        return ray_samples
    

class RegionAwareUniformLinDispPiecewiseSampler(RegionAwareSpacedSampler):
    """Piecewise Region Aware sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.


    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )
    

class RegionAwarePDFSampler(PDFSampler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        ray_samples: Optional[RaySamples] = None,
        weights: Optional[Float[torch.Tensor, "*batch num_samples 1"]] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
        ray_masks: Optional[torch.Tensor] = None,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")
        assert weights is not None, "weights must be provided"

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and self.training and ray_masks is None:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand

        else:
            # Uniform samples excluding selected region
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            u[:, -1] = 1

            if ray_masks != None:
                spacing_fn = lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x))
                masked_rays = spacing_fn(ray_masks[..., 0])

                zero_tensor = torch.zeros(masked_rays.shape[0])[..., None].to(masked_rays.device)
                masked_rays = torch.torch.cat((masked_rays, zero_tensor), dim=1)

                bin_mask_raise = torch.where(torch.diff(torch.where(masked_rays > 0, 1, 0), prepend=zero_tensor) > 0, 1, 0)
                bin_mask_fall = torch.where(torch.diff(torch.where(masked_rays > 0, 1, 0), append=zero_tensor) < 0, 1, 0)

                end = (1 - (torch.sum(masked_rays * bin_mask_fall, dim=1) - torch.sum(masked_rays * bin_mask_raise, dim=1)))[..., None]
                multidim_arange = torch.cumsum(torch.ones(masked_rays.shape).to(masked_rays.device), dim=1) - 1
                bins = (multidim_arange * end / masked_rays.shape[1])

                increment_step = end / (masked_rays.shape[1] - 1)
                move_idxs = torch.floor(((bins - masked_rays) / (increment_step)) * torch.where(masked_rays > 0, 1, 0))

                closest_idxs = (multidim_arange - move_idxs).int()

                sort_start = torch.sort(closest_idxs * bin_mask_raise)[0]
                sort_end = torch.sort(closest_idxs * bin_mask_fall)[0]

                inserted_zero = torch.torch.cat((zero_tensor, sort_end - sort_start), dim=1)[:, :-1]
                starting_idxs = (sort_start - torch.cumsum(inserted_zero, dim=1)).int()
                starting_idxs = torch.where(starting_idxs >= masked_rays.shape[1], masked_rays.shape[1] - 1, starting_idxs)
                starting_positions = torch.zeros(masked_rays.shape).to(masked_rays.device)
                starting_positions[torch.arange(masked_rays.shape[0])[..., None], starting_idxs] = (sort_end - sort_start).float()
                w = ((multidim_arange + torch.cumsum(starting_positions, dim=1)) * increment_step)
            
        u = u.contiguous()

        assert (
            ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert ray_samples.spacing_to_euclidean_fn is not None, "ray_samples.spacing_to_euclidean_fn must be provided"
        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts[..., 0],
                ray_samples.spacing_ends[..., -1:, 0],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        if ray_masks is not None:
            spacing_fn = lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x))
            masked_rays = spacing_fn(ray_masks[..., 0])

            zero_tensor = torch.zeros(masked_rays.shape[0])[..., None].to(masked_rays.device)

            bin_mask_raise = torch.where(torch.diff(torch.where(masked_rays > 0, 1, 0), prepend=zero_tensor) > 0, 1, 0)
            bin_mask_fall = torch.where(torch.diff(torch.where(masked_rays > 0, 1, 0), append=zero_tensor) < 0, 1, 0)

            above = torch.searchsorted(bins, masked_rays * bin_mask_raise)
            below = torch.searchsorted(bins, masked_rays * bin_mask_fall)

            temp = torch.zeros(masked_rays.shape).to(masked_rays.device)
            temp[torch.arange(masked_rays.shape[0])[..., None], torch.clip(above, 0, masked_rays.shape[1] - 1)] = 1
            temp[torch.arange(masked_rays.shape[0])[..., None], torch.clip(below, 0, masked_rays.shape[1] - 1)] += 1
            temp[:, 0] = 0

            values_to_remove = torch.cat([(torch.cumsum(temp, dim=-1) % 2), zero_tensor], dim=-1)
            bins, _ = torch.sort(((1 - values_to_remove) * bins) + (values_to_remove * w), dim=-1)
            
        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        bin_starts = None
        bin_ends = None
        spacing_starts = None
        spacing_ends = None
        
        if ray_masks == None:
            bin_starts = euclidean_bins[..., :-1, None]
            bin_ends = euclidean_bins[..., 1:, None]
            spacing_starts=bins[..., :-1, None]
            spacing_ends=bins[..., 1:, None]
        
        else:
            masked_rays = ray_masks[..., 0]

            zero_tensor = torch.zeros(masked_rays.shape[0])[..., None].to(ray_masks.device)

            bin_mask_raise = torch.where(torch.diff(torch.where(masked_rays > 0, 1, 0), prepend=zero_tensor) > 0, 1, -1)
            bin_mask_raise[:, 2] = bin_mask_raise[:, 1]
            bin_mask_raise = torch.cat([bin_mask_raise, -torch.ones((bin_mask_raise.shape[0], 1)).to(bin_mask_raise.device)], dim=1)
            bins_mean = (euclidean_bins[..., 1:, None] + euclidean_bins[..., :-1, None]) / 2
            bins_mean = torch.cat([bins_mean, torch.zeros((bins_mean.shape[0], 1, 1)).to(bins_mean.device)], dim=1)

            masked_rays = torch.cat([masked_rays, torch.zeros((masked_rays.shape[0], 1)).to(masked_rays.device)], dim=1)
            masked_inf_rays = torch.where(masked_rays == 0.0, torch.inf, masked_rays)

            insert_idx = torch.searchsorted(bins_mean[:, :, 0], masked_inf_rays) * bin_mask_raise
            insert_idx = torch.where(insert_idx < 0, -1, insert_idx)

            bin_mask_all = torch.zeros(euclidean_bins.shape, device=masked_rays.device)[..., None]
            insert_idx = torch.clip(insert_idx, torch.min(insert_idx), masked_rays.shape[1] - 1)
            bin_mask_all[torch.arange(masked_rays.shape[0])[..., None], insert_idx.int()] = 1

            bin_mask_start = torch.where(bin_mask_all > 0, 1, 0)[:, :-1, :]

            bin_starts = euclidean_bins[..., :-1, None]
            bin_ends = (euclidean_bins[..., 1:, None] * (1 - bin_mask_start)) + ((bin_starts + 1e-9) * bin_mask_start)

            spacing_starts = bins[..., :-1, None]
            spacing_ends = (bins[..., 1:, None] * (1 - bin_mask_start)) + ((spacing_starts + 1e-9) * bin_mask_start)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=bin_starts,
            bin_ends=bin_ends,
            spacing_starts=spacing_starts,
            spacing_ends=spacing_ends,
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )

        return ray_samples