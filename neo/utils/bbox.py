import torch

from typing import Tuple

from neo.utils.bbox_utils import BBoxData


class BoundingBox():

    def __init__(self, data: BBoxData, device="cuda:0") -> None:
        """Bounding box class for computing bounding distances

        Args:
            size: (x, y, z) size in meters
            data: BBoxData object containing:
                translation: translation of a cube in global coordinates
                rotation: rotation of a cube in RPY euler angles in radians
            device: device where to send data default "cuda:0"
        """
        self.device = device
        self.translation = torch.Tensor(data.position).to(self.device)
        self.rotation = torch.Tensor(data.euler).to(self.device)

        alpha, beta, gamma = self.rotation

        rot_roll = torch.Tensor([[1, 0, 0],
                                 [0, torch.cos(alpha), -torch.sin(alpha)],
                                 [0, torch.sin(alpha), torch.cos(alpha)]]).to(self.device)
        
        rot_pitch = torch.Tensor([[torch.cos(beta), 0, torch.sin(beta)],
                                  [0, 1, 0],
                                  [-torch.sin(beta), 0, torch.cos(beta)]]).to(self.device)
        
        rot_yaw = torch.Tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                                [torch.sin(gamma), torch.cos(gamma), 0],
                                [0, 0, 1]]).to(self.device)
        
        self.orientation = rot_roll @ rot_pitch @ rot_yaw

        x, y, z = data.dimensions
        x, y, z = x / 2, y / 2, z / 2
        
        cube = torch.Tensor([[-x, -y, -z],
                             [-x, y, -z],
                             [x, y, -z],
                             [x, -y, -z],
                             [-x, -y, z],
                             [-x, y, z],
                             [x, y, z],
                             [x, -y, z],]).to(self.device)

        EA = cube[4] - cube[0]
        DA = cube[3] - cube[0]
        BA = cube[1] - cube[0]
        FG = cube[5] - cube[6]
        HG = cube[7] - cube[6]
        CG = cube[2] - cube[6]

        self.N = torch.stack([torch.cross(DA, EA) / torch.linalg.norm(torch.cross(DA, EA)),
                              torch.cross(EA, BA) / torch.linalg.norm(torch.cross(EA, BA)),
                              torch.cross(BA, DA) / torch.linalg.norm(torch.cross(BA, DA)),
                              torch.cross(CG, FG) / torch.linalg.norm(torch.cross(CG, FG)),
                              torch.cross(HG, CG) / torch.linalg.norm(torch.cross(HG, CG)),
                              torch.cross(FG, HG) / torch.linalg.norm(torch.cross(FG, HG))])
        
        d = torch.stack([-(self.N[0][0] * cube[0][0] + self.N[0][1] * cube[0][1] + self.N[0][2] * cube[0][2]),
                         -(self.N[1][0] * cube[0][0] + self.N[1][1] * cube[0][1] + self.N[1][2] * cube[0][2]),
                         -(self.N[2][0] * cube[0][0] + self.N[2][1] * cube[0][1] + self.N[2][2] * cube[0][2]),
                         -(self.N[3][0] * cube[6][0] + self.N[3][1] * cube[6][1] + self.N[3][2] * cube[6][2]),
                         -(self.N[4][0] * cube[6][0] + self.N[4][1] * cube[6][1] + self.N[4][2] * cube[6][2]),
                         -(self.N[5][0] * cube[6][0] + self.N[5][1] * cube[6][1] + self.N[5][2] * cube[6][2])])
        
        self.p0 = self.N * -d[..., None]

        self.cube_center = torch.stack([torch.mean(cube[:, 0]), torch.mean(cube[:, 1]), torch.mean(cube[:, 2])])

        self.w_x = (torch.max(cube[:, 0]) - torch.min(cube[:, 0])) / 2
        self.w_y = (torch.max(cube[:, 1]) - torch.min(cube[:, 1])) / 2
        self.w_z = (torch.max(cube[:, 2]) - torch.min(cube[:, 2])) / 2

    def calculate_distances(self, ray_origins, ray_directions) -> torch.Tensor:
        """
        Perform inverse translations and rotations on rays. 
        This function returns bounds per ray.
        """

        ray_origins = ray_origins.detach().clone()
        ray_directions = ray_directions.detach().clone()

        ray_directions = (self.orientation @ ray_directions[..., None]).squeeze(-1)
        ray_origins -= self.translation
        ray_origins = (self.orientation @ (ray_origins + self.cube_center)[..., None]).squeeze(-1) - self.cube_center

        eps = 0.000001
        planes_intersections = []
        for i in range(6):
            den = (ray_directions * self.N[i].reshape(1, 3)).sum(dim=-1)[..., None]
            den_neg = (-ray_directions * self.N[i].reshape(1, 3)).sum(dim=-1)[..., None]
            direction = torch.where(den < eps, -1, 1)
            den = torch.where(den < eps, den_neg, den)

            t = ((self.p0 - ray_origins[0])[i] @ self.N[i].reshape(-1,1)) / den
            ip = torch.round(ray_origins + t * direction * ray_directions, decimals=4)

            arg_ip_x = torch.logical_or(ip[:, 0] > self.cube_center[0] + self.w_x, ip[:, 0] < self.cube_center[0] - self.w_x)
            arg_ip_y = torch.logical_or(ip[:, 1] > self.cube_center[1] + self.w_y, ip[:, 1] < self.cube_center[1] - self.w_y)
            arg_ip_z = torch.logical_or(ip[:, 2] > self.cube_center[2] + self.w_z, ip[:, 2] < self.cube_center[2] - self.w_z)
            arg_ip = torch.concatenate([arg_ip_x[..., None], arg_ip_y[..., None], arg_ip_z[..., None]], dim=-1)

            rows_to_replace = torch.any(arg_ip, dim=1)
            ip[rows_to_replace, :] = torch.Tensor([torch.inf, torch.inf, torch.inf]).to(self.device)

            planes_intersections.append(ip)

        planes_intersections = torch.stack(planes_intersections)

        points_on_cube = torch.argsort(torch.sum(torch.square(planes_intersections - self.cube_center), dim=-1), dim=0)
        intersecting_points = planes_intersections[points_on_cube, torch.arange(points_on_cube.shape[1])]

        dist = torch.sqrt(torch.sum(torch.square(intersecting_points[0:2, ...] - ray_origins), dim=-1)).T
        dist_bool = torch.where(dist == torch.inf, True, False)

        case_1 = torch.logical_or(dist_bool[:, 0], dist_bool[:, 1])

        temp_ip = torch.where(intersecting_points == torch.inf, 0, intersecting_points)
        case_2_first_point = ((temp_ip[0] - ray_origins[0])[:, None, :] @ ray_directions[..., None]).reshape(-1) < 0
        case_2_second_point = ((temp_ip[1] - ray_origins[0])[:, None, :] @ ray_directions[..., None]).reshape(-1) < 0

        dist[case_1, :] = torch.Tensor([0, 0]).to(ray_origins.device)
        # dist[case_2_first_point, 0] = 1e-9
        # dist[case_2_second_point, 1] = 1e-9

        dist, _ = torch.sort(dist, dim=1)

        return dist
    
    def are_points_inside(self, points: torch.Tensor, offset: Tuple[float, float, float] = (0, 0, 0)) -> torch.Tensor:

        points_shape = points.shape
        points = points.reshape((-1, 3))

        # Transform the points into the local coordinate system of the OOBB
        local_points = torch.matmul(points - self.translation, self.orientation.T)
        
        # Perform containment check for each point
        inside_x = torch.abs(local_points[:, 0]) <= self.w_x + offset[0]
        inside_y = torch.abs(local_points[:, 1]) <= self.w_y + offset[1]
        inside_z = torch.abs(local_points[:, 2]) <= self.w_z + offset[2]
        
        inside = torch.logical_and(torch.logical_and(inside_x, inside_y), inside_z)
        inside = inside.reshape(points_shape[0:2]).unsqueeze(-1)
        
        return inside