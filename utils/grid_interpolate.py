import torch
import torch.nn.functional as F


def coords2heatmap(coords, height, width, kernel_size=1.):
    batch_size, n_coords, _ = coords.size()

    coords_ft = coords.reshape(batch_size, -1, 1, 1).repeat(1, 1, height, width)

    mesh_x = 2.0 * (torch.arange(width)[None].expand(height, width).float()+0.5) / width - 1.0
    mesh_y = 2.0 * (torch.arange(height)[:,None].expand(height, width).float()+0.5) / height - 1.0
    coords = torch.stack((mesh_x, mesh_y), dim=0)
    coords = coords.unsqueeze(0).repeat(batch_size, n_coords, 1, 1).to(coords_ft.device) # (B, 2, H, W)

    offset = coords_ft - coords # (B, n_coords*2, F, F)
    offset = offset.view(batch_size, n_coords, 2, height, width) # (B, n_coords, 2, H, W)
    dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2) + 1e-8) # (B, n_coords, H, W)

    heatmap = (kernel_size - dis) / kernel_size # (B, n_coords, F, F)

    return F.softmax(heatmap.view(batch_size, n_coords, -1) * 20, dim=-1)
