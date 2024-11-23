import torch


def depth_to_distance(depth: torch.Tensor, fov_x: float = None, fov_y: float = None, fl_x: float = None, fl_y: float = None) -> torch.Tensor:
    device = depth.device
    batch_size, _, height, width = depth.shape

    if fov_x is not None and fov_y is not None:
        fov_x_rad = fov_x * torch.pi / 180.0
        fov_y_rad = fov_y * torch.pi / 180.0
        fx = 1 / torch.tan(fov_x_rad / 2)
        fy = 1 / torch.tan(fov_y_rad / 2)

    elif fl_x is not None and fl_y is not None:
        fx = fl_x
        fy = fl_y

    else:
        raise ValueError("Either both 'fov_x' and 'fov_y' or both 'fl_x' and 'fl_y' must be provided.")
    
    x = torch.linspace(-1, 1, width, device=device) / fx
    y = torch.linspace(-1, 1, height, device=device) / fy
    ys, xs = torch.meshgrid([y, x], indexing="ij")
    xy = torch.stack([xs, ys]).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    xy *= depth
    z = depth
    xyz = torch.cat([xy, z], dim=1)
    distance = torch.sqrt((xyz**2).sum(1, keepdim=True))

    return distance


def distance_to_depth(distance: torch.Tensor, fov_x: float, fov_y: float, fl_x: float = None, fl_y: float = None) -> torch.Tensor:
    device = distance.device
    batch_size, _, height, width = distance.shape

    if fov_x is not None and fov_y is not None:
        fov_x_rad = fov_x * torch.pi / 180.0
        fov_y_rad = fov_y * torch.pi / 180.0
        fx = 1 / torch.tan(fov_x_rad / 2)
        fy = 1 / torch.tan(fov_y_rad / 2)

    elif fl_x is not None and fl_y is not None:
        fx = fl_x
        fy = fl_y

    else:
        raise ValueError("Either both 'fov_x' and 'fov_y' or both 'fl_x' and 'fl_y' must be provided.")

    x = torch.linspace(-1, 1, width, device=device) / fx
    y = torch.linspace(-1, 1, height, device=device) / fy
    ys, xs = torch.meshgrid([y, x], indexing="ij")
    xy = torch.stack([xs, ys]).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    xyz = torch.cat([xy, torch.ones_like(xy[:, 0:1])], dim=1)
    xyz = xyz / torch.norm(xyz, dim=1, keepdim=True)
    xyz = xyz * distance
    depth = xyz[:, 2:3]

    return depth