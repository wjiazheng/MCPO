import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, sys, json
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt as edt

warnings.filterwarnings("ignore")

class SpatialTransformer(nn.Module):#输入的变形场（flow）来对源图像（src）进行空间变换

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, is_grid_out=False, mode=None, align_corners=True):

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if mode is None:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=self.mode)
        else:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=mode)

        if is_grid_out:
            return out, new_locs
        return out

class registerSTModel(nn.Module):

    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(registerSTModel, self).__init__()

        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, img, flow, is_grid_out=False, align_corners=True):
        out = self.spatial_trans(img, flow, is_grid_out=is_grid_out, align_corners=align_corners)

        return out

#enforce inverse consistency of forward and backward transform
# 强制执行前向和反向变换的逆一致性
def inverse_consistency(disp_field1s,disp_field2s,iter=20):
    B,C,H,W,D = disp_field1s.size()
    #make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D)).permute(0,4,1,2,3).to(disp_field1s.device).to(disp_field1s.dtype)
        for i in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5*(disp_field1s-F.grid_sample(disp_field2s,(identity+disp_field1s).permute(0,2,3,4,1)))
            disp_field2i = 0.5*(disp_field2s-F.grid_sample(disp_field1s,(identity+disp_field2s).permute(0,2,3,4,1)))

    return disp_field1i,disp_field2i

#solve two coupled convex optimisation problems for efficient global regularisation
# 解决两个耦合的凸优化问题以实现高效的全局正则化，交替优化光滑性和相似性
def coupled_convex_mean(ssd,ssd_argmin,disp_mesh_t,grid_sp,shape):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);

    disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)
    disp_soft_all = []
    coeffs = torch.tensor([0.003,0.01,0.03,0.1,0.3,1])
    for j in range(6):
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H//grid_sp):

                coupled = ssd[:,i,:,:]+coeffs[j]*(disp_mesh_t-disp_soft[:,:,i].view(3,1,-1)).pow(2).sum(0).view(-1,W//grid_sp,D//grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled,0)
                # ssd_coupled_argmin[i] = torch.argmax(coupled, 0)

        disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_coupled_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)
        disp_soft_all.append(disp_soft)

    disp_soft_mean = sum(disp_soft_all) / len(disp_soft_all) ##mean

    return disp_soft_mean

#correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(mind_fix,mind_mov,disp_hw,grid_sp,shape, ch=12):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);

    with torch.no_grad():
        mind_unfold = F.unfold(F.pad(mind_mov,(disp_hw,disp_hw,disp_hw,disp_hw,disp_hw,disp_hw)).squeeze(0),disp_hw*2+1)
        mind_unfold = mind_unfold.view(ch,-1,(disp_hw*2+1)**2,W//grid_sp,D//grid_sp)

    ssd = torch.zeros((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp,dtype=mind_fix.dtype, device=mind_fix.device)#.cuda().half()
    with torch.no_grad():
        for i in range(disp_hw*2+1):
            mind_sum = (mind_fix.permute(1,2,0,3,4)-mind_unfold[:,i:i+H//grid_sp]).pow(2).sum(0,keepdim=True)
            ssd[i::(disp_hw*2+1)] = F.avg_pool3d(F.avg_pool3d(mind_sum.transpose(2,1),3,stride=1,padding=1),3,stride=1,padding=1).squeeze(1)
        ssd = ssd.view(disp_hw*2+1,disp_hw*2+1,disp_hw*2+1,H//grid_sp,W//grid_sp,D//grid_sp).transpose(1,0).reshape((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp)
        ssd_argmin = torch.argmin(ssd,0)#

    return ssd, ssd_argmin

def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2(
        (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                       kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item() * 0.001, mind_var.mean().item() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind

def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

def extract_features(
    img_fixed: torch.Tensor,
    img_moving: torch.Tensor,
    mind_r: int,
    mind_d: int,
    use_mask: bool,
    mask_fixed: torch.Tensor,
    mask_moving: torch.Tensor,
) -> (torch.Tensor, torch.Tensor):
    """Extract MIND and/or semantic nnUNet features"""

    # MIND features
    if use_mask:
        H,W,D = img_fixed.shape[-3:]

        #replicate masking
        avg3 = nn.Sequential(nn.ReplicationPad3d(1),nn.AvgPool3d(3,stride=1))
        avg3.cuda()
        
        mask = (avg3(mask_fixed.view(1,1,H,W,D).cuda())>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        fixed_r = F.interpolate((img_fixed[::2,::2,::2].cuda().reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        fixed_r.view(-1)[mask.view(-1)!=0] = img_fixed.cuda().reshape(-1)[mask.view(-1)!=0]

        mask = (avg3(mask_moving.view(1,1,H,W,D).cuda())>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        moving_r = F.interpolate((img_moving[::2,::2,::2].cuda().reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        moving_r.view(-1)[mask.view(-1)!=0] = img_moving.cuda().reshape(-1)[mask.view(-1)!=0]

        features_fix = MINDSSC(fixed_r.cuda(),mind_r,mind_d).half()
        features_mov = MINDSSC(moving_r.cuda(),mind_r,mind_d).half()
    else:
        img_fixed = img_fixed.unsqueeze(0).unsqueeze(0)
        img_moving = img_moving.unsqueeze(0).unsqueeze(0)
        features_fix = MINDSSC(img_fixed.cuda(),mind_r,mind_d).half()
        features_mov = MINDSSC(img_moving.cuda(),mind_r,mind_d).half()
    
    return features_fix, features_mov

def validate_image(img: Union[torch.Tensor, np.ndarray, sitk.Image], dtype=float) -> torch.Tensor:
    """Validate image input"""
    if not isinstance(img, torch.Tensor):
        if isinstance(img, sitk.Image):
            img = sitk.GetArrayFromImage(img)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.astype(dtype))
        else:
            raise ValueError("Input image must be a torch.Tensor, a numpy.ndarray or a SimpleITK.Image")
    return img

def find_rigid_3d(x, y):
    x_mean = x[:, :3].mean(0)
    y_mean = y[:, :3].mean(0)
    u, s, v = torch.svd(torch.matmul((x[:, :3]-x_mean).t(), (y[:, :3]-y_mean)))
    m = torch.eye(v.shape[0], v.shape[0]).to(x.device)
    m[-1,-1] = torch.det(torch.matmul(v, u.t()))
    rotation = torch.matmul(torch.matmul(v, m), u.t())
    translation = y_mean - torch.matmul(rotation, x_mean)
    T = torch.eye(4).to(x.device)
    T[:3,:3] = rotation
    T[:3, 3] = translation
    return T

def least_trimmed_rigid(fixed_pts, moving_pts, iter=5):
    idx = torch.arange(fixed_pts.shape[0]).to(fixed_pts.device)
    for i in range(iter):
        x = find_rigid_3d(fixed_pts[idx,:], moving_pts[idx,:]).t()
        residual = torch.sqrt(torch.sum(torch.pow(moving_pts - torch.mm(fixed_pts, x), 2), 1))
        _, idx = torch.topk(residual, fixed_pts.shape[0]//2, largest=False)
    return x.t()

def get_mask(ref_data):
    """
    通过图像数据生成二值 mask，假设所有值大于 0 的区域为前景，生成二值化的 mask。
    """
    # 将大于 0 的部分设为 1，其它部分设为 0，得到二值化图像
    img_data = (ref_data > 0).float()
    return img_data

def convex_adam_pt_nofu(
    img_fixed: Union[torch.Tensor, np.ndarray, sitk.Image],
    img_moving: Union[torch.Tensor, np.ndarray, sitk.Image],
    mind_r: int = 2,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    use_mask: bool = False,
    
) -> None:
    """Coupled convex optimisation with adam instance optimisation"""
    img_fixed = validate_image(img_fixed)
    img_moving = validate_image(img_moving)
    img_fixed = img_fixed.float()
    img_moving = img_moving.float()
    H, W, D = img_fixed.shape
    reg_model = registerSTModel([H, W, D], 'nearest').cuda()
    if use_mask:
        mask_fixed = get_mask(img_fixed).float()
        mask_moving = get_mask(img_moving).float()
    else:
        mask_fixed = None
        mask_moving = None

    grid_sp1 = grid_sp-1
    grid_sp2 = grid_sp-2
    grid_sp3 = grid_sp-3

    disp_hw0 = disp_hw
    disp_hw1 = disp_hw

    # H, W, D = img_fixed.shape
    affine = F.affine_grid(torch.eye(3,4).cuda().unsqueeze(0),(1,1,H,W,D),align_corners=False)

    # compute features and downsample (using average pooling)
    with torch.no_grad():

        features_fix1, features_mov1 = extract_features(img_fixed=img_fixed,
                                                      img_moving=img_moving,
                                                      mind_r=mind_r,
                                                      mind_d=mind_d,
                                                      use_mask=use_mask,
                                                      mask_fixed=mask_fixed,
                                                      mask_moving=mask_moving)

        features_fix_smooth1 = F.avg_pool3d(features_fix1,grid_sp,stride=grid_sp)
        features_mov_smooth1 = F.avg_pool3d(features_mov1,grid_sp,stride=grid_sp)

        n_ch1 = features_fix_smooth1.shape[1]
    del features_fix1, features_mov1

    # compute correlation volume with SSD
    ssd,ssd_argmin = correlate(features_fix_smooth1,features_mov_smooth1,disp_hw0,grid_sp,(H,W,D), n_ch1)

    # provide auxiliary mesh grid
    disp_mesh_t1 = F.affine_grid(disp_hw0*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw0*2+1,disp_hw0*2+1,disp_hw0*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)

    # perform coupled convex optimisation
    disp_soft_1 = coupled_convex_mean(ssd, ssd_argmin, disp_mesh_t1, (grid_sp), (H, W, D))
    
    if ic:
        scale = torch.tensor([H//(grid_sp)-1,W//(grid_sp)-1,D//(grid_sp)-1]).view(1,3,1,1,1).cuda().half()/2
        
        ssd_,ssd_argmin_ = correlate(features_mov_smooth1,features_fix_smooth1,disp_hw0,(grid_sp),(H,W,D), n_ch1)

        disp_soft_1_ = coupled_convex_mean(ssd_,ssd_argmin_,disp_mesh_t1,(grid_sp),(H,W,D))
          
        disp_ice1,disp_ice1_ = inverse_consistency((disp_soft_1/scale).flip(1),(disp_soft_1_/scale).flip(1),iter=15)

        deformation_field1 = F.interpolate(disp_ice1.flip(1)*scale*(grid_sp),size=(H,W,D),mode='trilinear',align_corners=False)
    else:
        deformation_field1 = F.interpolate(disp_soft_1*(grid_sp),size=(H,W,D),mode='trilinear',align_corners=False)
 
    disp0 = deformation_field1.cuda().float().permute(0,2,3,4,1)/torch.tensor([H-1,W-1,D-1]).cuda().view(1,1,1,1,3)*2
    disp0 = disp0.flip(4)
    affine_sp = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//(grid_sp),W//(grid_sp),D//(grid_sp)),align_corners=False)
            
    mask_fix = F.avg_pool3d((img_fixed>0).cuda().float().unsqueeze(0).unsqueeze(0),(grid_sp),stride=(grid_sp))>.5
        
    affine_sp = affine_sp.reshape(-1,3)[torch.nonzero(mask_fix.reshape(-1)),:]

    T1 = F.grid_sample(affine.permute(0,4,1,2,3),affine_sp.reshape(1,-1,1,1,3))
    T2 = F.grid_sample((affine+disp0).permute(0,4,1,2,3),affine_sp.reshape(1,-1,1,1,3))
    T1 = torch.cat((T1.squeeze().t().cuda(),torch.ones(affine_sp.shape[0],1).cuda()),1)
    T2 = torch.cat((T2.squeeze().t().cuda(),torch.ones(affine_sp.shape[0],1).cuda()),1)
        
    R = least_trimmed_rigid(T1,T2,15)#torch.cat((T1,T1_),0),torch.cat((T2,T2_),0))
    affineR = F.affine_grid(R[:3].unsqueeze(0),(1,1,H,W,D),align_corners=False)
    disp1 = affineR - affine
    disp_hr_rigid0 = disp1.flip(-1)
    scaling_factor = torch.tensor([H-1, W-1, D-1]).float().view(1, 1, 1, 1, 3).cuda()
    disp_hr_rigid0 = disp_hr_rigid0 * scaling_factor / 2
    disp_hr_rigid0 = disp_hr_rigid0.permute(0, 4, 1, 2, 3)
    
    out_t0 = reg_model(img_moving.cuda().view(1, 1, H, W, D), disp_hr_rigid0)
    out_t0=out_t0.squeeze()
    
    if use_mask:
        mask_warped0 = get_mask(out_t0).float()
    else:
        mask_warped0 = None

    # compute features and downsample (using average pooling)
    with torch.no_grad():

        features_fix1, features_mov1 = extract_features(img_fixed=img_fixed,
                                                        img_moving=out_t0,
                                                        mind_r=mind_r,
                                                        mind_d=mind_d,
                                                        use_mask=use_mask,
                                                        mask_fixed=mask_fixed,
                                                        mask_moving=mask_warped0)

        features_fix_smooth1 = F.avg_pool3d(features_fix1, (grid_sp1), stride=(grid_sp1))
        features_mov_smooth1 = F.avg_pool3d(features_mov1, (grid_sp1), stride=(grid_sp1))

        n_ch1 = features_fix_smooth1.shape[1]
    del features_fix1, features_mov1

    # compute correlation volume with SSD
    ssd, ssd_argmin = correlate(features_fix_smooth1, features_mov_smooth1, disp_hw1, (grid_sp1), (H, W, D), n_ch1)

    # provide auxiliary mesh grid
    disp_mesh_t1 = F.affine_grid(disp_hw1 * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                 (1, 1, disp_hw1 * 2 + 1, disp_hw1 * 2 + 1, disp_hw1 * 2 + 1), align_corners=True).permute(
        0, 4, 1, 2, 3).reshape(3, -1, 1)

    # perform coupled convex optimisation
    disp_soft_1 = coupled_convex_mean(ssd, ssd_argmin, disp_mesh_t1, (grid_sp1), (H, W, D))

    if ic:
        scale = torch.tensor([H // (grid_sp1) - 1, W // (grid_sp1) - 1, D // (grid_sp1) - 1]).view(1, 3, 1, 1,
                                                                                          1).cuda().half() / 2

        ssd_, ssd_argmin_ = correlate(features_mov_smooth1, features_fix_smooth1, disp_hw1, (grid_sp1), (H, W, D), n_ch1)

        disp_soft_1_ = coupled_convex_mean(ssd_, ssd_argmin_, disp_mesh_t1, (grid_sp1), (H, W, D))

        disp_ice1, disp_ice1_ = inverse_consistency((disp_soft_1 / scale).flip(1), (disp_soft_1_ / scale).flip(1),
                                                    iter=15)

        deformation_field1 = F.interpolate(disp_ice1.flip(1) * scale * (grid_sp1), size=(H, W, D), mode='trilinear',
                                           align_corners=False)

        deformation_field1 = disp_hr_rigid0 + deformation_field1

    else:
        deformation_field1 = F.interpolate(disp_soft_1 * (grid_sp1), size=(H, W, D), mode='trilinear', align_corners=False)

        deformation_field1 = disp_hr_rigid0 + deformation_field1

    disp0 = deformation_field1.cuda().float().permute(0, 2, 3, 4, 1) / torch.tensor([H - 1, W - 1, D - 1]).cuda().view(
        1, 1, 1, 1, 3) * 2
    disp0 = disp0.flip(4)
    affine_sp = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, H // (grid_sp1), W // (grid_sp1), D // (grid_sp1)),
                              align_corners=False)

    mask_fix = F.avg_pool3d((img_fixed > 0).cuda().float().unsqueeze(0).unsqueeze(0), (grid_sp1), stride=(grid_sp1)) > .5

    affine_sp = affine_sp.reshape(-1, 3)[torch.nonzero(mask_fix.reshape(-1)), :]

    T1 = F.grid_sample(affine.permute(0, 4, 1, 2, 3), affine_sp.reshape(1, -1, 1, 1, 3))
    T2 = F.grid_sample((affine + disp0).permute(0, 4, 1, 2, 3), affine_sp.reshape(1, -1, 1, 1, 3))
    T1 = torch.cat((T1.squeeze().t().cuda(), torch.ones(affine_sp.shape[0], 1).cuda()), 1)
    T2 = torch.cat((T2.squeeze().t().cuda(), torch.ones(affine_sp.shape[0], 1).cuda()), 1)

    R = least_trimmed_rigid(T1, T2, 15)  # torch.cat((T1,T1_),0),torch.cat((T2,T2_),0))
    affineR = F.affine_grid(R[:3].unsqueeze(0), (1, 1, H, W, D), align_corners=False)
    disp1 = affineR - affine
    disp_hr_rigid1 = disp1.flip(-1)
    scaling_factor = torch.tensor([H - 1, W - 1, D - 1]).float().view(1, 1, 1, 1, 3).cuda()
    disp_hr_rigid1 = disp_hr_rigid1 * scaling_factor / 2
    disp_hr_rigid1 = disp_hr_rigid1.permute(0, 4, 1, 2, 3)

    out_t1 = reg_model(img_moving.cuda().view(1, 1, H, W, D), disp_hr_rigid1)
    out_t1 = out_t1.squeeze()

    if use_mask:
        mask_warped1 = get_mask(out_t1).float()
    else:
        mask_warped1 = None
        
    with torch.no_grad():

        features_fix2, features_mov2 = extract_features(img_fixed=img_fixed,
                                                      img_moving=out_t1,
                                                      mind_r=mind_r,
                                                      mind_d=mind_d,
                                                      use_mask=use_mask,
                                                      mask_fixed=mask_fixed,
                                                      mask_moving=mask_warped1)

        features_fix_smooth2 = F.avg_pool3d(features_fix2,grid_sp2,stride=grid_sp2)
        features_mov_smooth2 = F.avg_pool3d(features_mov2,grid_sp2,stride=grid_sp2)

        n_ch2 = features_fix_smooth2.shape[1]
    del features_fix2, features_mov2

    # compute correlation volume with SSD
    ssd,ssd_argmin = correlate(features_fix_smooth2,features_mov_smooth2,disp_hw,grid_sp2,(H,W,D), n_ch2)

    # provide auxiliary mesh grid
    disp_mesh_t2 = F.affine_grid((disp_hw)*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,(disp_hw)*2+1,(disp_hw)*2+1,(disp_hw)*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)

    # perform coupled convex optimisation
    disp_soft_2 = coupled_convex_mean(ssd, ssd_argmin, disp_mesh_t2, grid_sp2, (H, W, D))
    
    if ic:
        scale = torch.tensor([H//(grid_sp2)-1,W//(grid_sp2)-1,D//(grid_sp2)-1]).view(1,3,1,1,1).cuda().half()/2
        
        ssd_,ssd_argmin_ = correlate(features_mov_smooth2,features_fix_smooth2,disp_hw,grid_sp2,(H,W,D), n_ch2)

        disp_soft_2_= coupled_convex_mean(ssd_, ssd_argmin_, disp_mesh_t2, grid_sp2, (H, W, D))
          
        disp_ice2,disp_ice2_ = inverse_consistency((disp_soft_2/scale).flip(1),(disp_soft_2_/scale).flip(1),iter=15)

        deformation_field2 = F.interpolate(disp_ice2.flip(1)*scale*(grid_sp2),size=(H,W,D),mode='trilinear',align_corners=False)
        
        deformation_field2= disp_hr_rigid1 + deformation_field2
    else:
        disp_soft_h2 = F.interpolate(disp_soft_2*(grid_sp2),size=(H,W,D),mode='trilinear',align_corners=False)
        deformation_field2 = disp_hr_rigid1 + disp_soft_h2
    
    disp0 = deformation_field2.cuda().float().permute(0,2,3,4,1)/torch.tensor([H-1,W-1,D-1]).cuda().view(1,1,1,1,3)*2
    disp0 = disp0.flip(4)
    affine_sp = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//(grid_sp2),W//(grid_sp2),D//(grid_sp2)),align_corners=False)
            
    mask_fix = F.avg_pool3d((img_fixed>0).cuda().float().unsqueeze(0).unsqueeze(0),(grid_sp2),stride=(grid_sp2))>.5
        
    affine_sp = affine_sp.reshape(-1,3)[torch.nonzero(mask_fix.reshape(-1)),:]

    T1 = F.grid_sample(affine.permute(0,4,1,2,3),affine_sp.reshape(1,-1,1,1,3))
    T2 = F.grid_sample((affine+disp0).permute(0,4,1,2,3),affine_sp.reshape(1,-1,1,1,3))
    T1 = torch.cat((T1.squeeze().t().cuda(),torch.ones(affine_sp.shape[0],1).cuda()),1)
    T2 = torch.cat((T2.squeeze().t().cuda(),torch.ones(affine_sp.shape[0],1).cuda()),1)
        
    R = least_trimmed_rigid(T1,T2,15)#torch.cat((T1,T1_),0),torch.cat((T2,T2_),0))
    affineR = F.affine_grid(R[:3].unsqueeze(0),(1,1,H,W,D),align_corners=False)
    disp1 = affineR - affine
    disp_hr_rigid2 = disp1.flip(-1)
    scaling_factor = torch.tensor([H-1, W-1, D-1]).float().view(1, 1, 1, 1, 3).cuda()
    disp_hr_rigid2 = disp_hr_rigid2 * scaling_factor / 2
    disp_hr_rigid2 = disp_hr_rigid2.permute(0, 4, 1, 2, 3)
    
    out_t2 = reg_model(img_moving.cuda().view(1, 1, H, W, D), disp_hr_rigid2)
    out_t2=out_t2.squeeze()

    if use_mask:
        mask_warped2 = get_mask(out_t2).float()
    else:
        mask_warped2 = None

    with torch.no_grad():

        features_fix3, features_mov3 = extract_features(img_fixed=img_fixed,
                                                      img_moving=out_t2,
                                                      mind_r=mind_r,
                                                      mind_d=mind_d,
                                                      use_mask=use_mask,
                                                      mask_fixed=mask_fixed,
                                                      mask_moving=mask_warped2)

        features_fix_smooth3 = F.avg_pool3d(features_fix3,grid_sp3,stride=grid_sp3)
        features_mov_smooth3 = F.avg_pool3d(features_mov3,grid_sp3,stride=grid_sp3)

        n_ch3 = features_fix_smooth3.shape[1]
    del features_fix3, features_mov3

    # compute correlation volume with SSD
    ssd,ssd_argmin = correlate(features_fix_smooth3,features_mov_smooth3,disp_hw,grid_sp3,(H,W,D), n_ch3)

    # provide auxiliary mesh grid
    disp_mesh_t3 = F.affine_grid((disp_hw)*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,(disp_hw)*2+1,(disp_hw)*2+1,(disp_hw)*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)

    # perform coupled convex optimisation
    disp_soft_3 = coupled_convex_mean(ssd, ssd_argmin, disp_mesh_t3, grid_sp3, (H, W, D))

    if ic:
        scale = torch.tensor([H//(grid_sp3)-1,W//(grid_sp3)-1,D//(grid_sp3)-1]).view(1,3,1,1,1).cuda().half()/2

        ssd_,ssd_argmin_ = correlate(features_mov_smooth3,features_fix_smooth3,disp_hw,grid_sp3,(H,W,D), n_ch3)

        disp_soft_3_= coupled_convex_mean(ssd_, ssd_argmin_, disp_mesh_t3, grid_sp3, (H, W, D))

        disp_ice3,disp_ice3_ = inverse_consistency((disp_soft_3/scale).flip(1),(disp_soft_3_/scale).flip(1),iter=15)

        deformation_field3 = F.interpolate(disp_ice3.flip(1)*scale*(grid_sp3),size=(H,W,D),mode='trilinear',align_corners=False)

        deformation_field3= disp_hr_rigid2 + deformation_field3
    else:
        disp_soft_h3 = F.interpolate(disp_soft_3*(grid_sp3),size=(H,W,D),mode='trilinear',align_corners=False)
        deformation_field3 = disp_hr_rigid2 + disp_soft_h3

    disp0 = deformation_field3.cuda().float().permute(0,2,3,4,1)/torch.tensor([H-1,W-1,D-1]).cuda().view(1,1,1,1,3)*2
    disp0 = disp0.flip(4)
    affine_sp = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//(grid_sp3),W//(grid_sp3),D//(grid_sp3)),align_corners=False)

    mask_fix = F.avg_pool3d((img_fixed>0).cuda().float().unsqueeze(0).unsqueeze(0),(grid_sp3),stride=(grid_sp3))>.5

    affine_sp = affine_sp.reshape(-1,3)[torch.nonzero(mask_fix.reshape(-1)),:]

    T1 = F.grid_sample(affine.permute(0,4,1,2,3),affine_sp.reshape(1,-1,1,1,3))
    T2 = F.grid_sample((affine+disp0).permute(0,4,1,2,3),affine_sp.reshape(1,-1,1,1,3))
    T1 = torch.cat((T1.squeeze().t().cuda(),torch.ones(affine_sp.shape[0],1).cuda()),1)
    T2 = torch.cat((T2.squeeze().t().cuda(),torch.ones(affine_sp.shape[0],1).cuda()),1)

    R = least_trimmed_rigid(T1,T2,15)#torch.cat((T1,T1_),0),torch.cat((T2,T2_),0))
    affineR = F.affine_grid(R[:3].unsqueeze(0),(1,1,H,W,D),align_corners=False)
    disp1 = affineR - affine
    disp_hr_rigid3 = disp1.flip(-1)
    scaling_factor = torch.tensor([H-1, W-1, D-1]).float().view(1, 1, 1, 1, 3).cuda()
    disp_hr_rigid3 = disp_hr_rigid3 * scaling_factor / 2
    disp_hr_rigid3 = disp_hr_rigid3.permute(0, 4, 1, 2, 3)
        
    x = disp_hr_rigid3[0,0,:,:,:].cpu().half().data.numpy()
    y = disp_hr_rigid3[0,1,:,:,:].cpu().half().data.numpy()
    z = disp_hr_rigid3[0,2,:,:,:].cpu().half().data.numpy()
    displacements = np.stack((x,y,z),3).astype(float)
    torch.cuda.empty_cache()

    return displacements, disp_hr_rigid3

def convex_adam_nofu(
    path_img_fixed: Union[Path, str],
    path_img_moving: Union[Path, str],
    mind_r: int = 1,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_niter_rigid: int = 500,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    use_mask: bool = False,
    result_path: Union[Path, str] = './',
) -> None:
    """Coupled convex optimisation with adam instance optimisation"""
    img_fixed = torch.from_numpy(nib.load(path_img_fixed).get_fdata()).float()
    img_moving = torch.from_numpy(nib.load(path_img_moving).get_fdata()).float()
    print('ic',ic)
    print('use_mask',use_mask)

    displacements, disp_flow = convex_adam_pt_nofu(
        img_fixed=img_fixed,
        img_moving=img_moving,
        mind_r=mind_r,
        mind_d=mind_d,
        lambda_weight=lambda_weight,
        grid_sp=grid_sp,
        disp_hw=disp_hw,
        selected_niter=selected_niter,
        selected_smooth=selected_smooth,
        grid_sp_adam=grid_sp_adam,
        ic=ic,
        use_mask=use_mask,
    )

    flow = disp_flow
    disp_out = displacements

    return flow, disp_out

#然后那个mi的loss用这个
# class GmiLoss_soft(_Loss):
class GmiLoss_soft(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, eps=1e-7):
        """
        计算归一化后的互信息损失
        输入:
            x: (dimensions, n_features)
            y: (dimensions, n_features)
        输出:
            mi_loss: 标量（互信息）
        """
        # 归一化为概率分布
        x_prob = F.softmax(x, dim=1)
        y_prob = F.softmax(y, dim=1)

        # 计算边缘熵 H(X) 和 H(Y)
        H_x = - (x_prob * torch.log2(x_prob + eps)).sum(dim=1).mean()
        H_y = - (y_prob * torch.log2(y_prob + eps)).sum(dim=1).mean()

        # 计算联合熵 H(X,Y)
        joint_prob = torch.bmm(x_prob.unsqueeze(2), y_prob.unsqueeze(1))  # (B, n_feat, n_feat)
        H_xy = - (joint_prob * torch.log2(joint_prob + eps)).sum(dim=(1, 2)).mean()
        cmif = H_x + H_y - H_xy

        return cmif

def get_cuboid_points(point1, point2):
    """
    根据立方体的两个对角线端点坐标，返回立方体内所有整数点的坐标
    """
    # 提取各维度坐标并排序
    
    x_coords = sorted([point1[0,0,0], point2[0,0,0]])
    y_coords = sorted([point1[0,0,1], point2[0,0,1]])
    z_coords = sorted([point1[0,0,2], point2[0,0,2]])

    # 生成各维度的坐标范围（包含端点）
    x_range = torch.arange(x_coords[0], x_coords[1])
    y_range = torch.arange(y_coords[0], y_coords[1])
    z_range = torch.arange(z_coords[0], z_coords[1])

    # 生成三维网格坐标
    xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')

    # 组合成坐标列表并返回
    return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).tolist()

def block_filtering_and_sampling(image, block_size = 4):
    """
    基于 PyTorch 的图像分块与筛选
    参数:
        image_path (str): 图像路径
    返回:
        selected_coords (torch.Tensor): 选中块的坐标，形状为 (N, 2)
    """
    # 调整尺寸为 4 的倍数
    D, H, W = image.shape
    new_D = D // block_size * block_size
    new_H = H // block_size * block_size
    new_W = W // block_size * block_size
    img_tensor = image[:new_D, :new_H, :new_W]  # 裁剪图像

    # 将图像分割为 4x4x4 块
    blocks = img_tensor.unfold(0, block_size, block_size).unfold(1, block_size, block_size).unfold(2, block_size, block_size)  # (D/4, H/4, W/4, 4, 4, 4)
    blocks = blocks.contiguous().view(-1, block_size, block_size, block_size)  # (N, 4, 4, 4)

    # 计算每个块的方差
    block_means = blocks.mean(dim=(1, 2, 3), keepdim=True)  # (N, 1, 1, 1)
    block_vars = ((blocks - block_means) ** 2).mean(dim=(1, 2, 3))  # (N,)

    # 筛选方差不为0的块
    # k = int(blocks.shape[0] * 0.25)  # 前 25% 的块数
    # top_var_indices = torch.topk(block_vars, k=k).indices  # 方差最大的块索引
    nonzero_var_blocks = torch.nonzero(block_vars)

    return nonzero_var_blocks

def sample_block_coord(nonzero_var_blocks, image_size, block_size=4, num_samples=1000):

    D, H, W = image_size[0], image_size[1], image_size[2]

    k = nonzero_var_blocks.shape[0]
    # 从筛选的块中随机挑选 num_samples
    if num_samples > k:
        num_random = k
    else:
        num_random = num_samples
    random_indices = torch.randperm(k)[:num_random]  # 随机挑选索引
    selected_indices = nonzero_var_blocks[random_indices]  # 最终选中的块索引

    # 计算选中块的坐标
    block_centers = torch.stack([
        selected_indices // ((H // block_size) * (W // block_size)),  # 深度坐标
        (selected_indices % ((H // block_size) * (W // block_size))) // (W // block_size),  # 高度坐标
        selected_indices % (W // block_size)  # 宽度坐标
    ], dim=-1) * block_size  # 转换为图像中的坐标


    sampled_blocks_coords = []
    for idx in range(num_random):
        center = torch.asarray(block_centers[idx:idx+1]).cuda()  # 中心点坐标
        # 计算局部区域的边界
        start = torch.clamp(center - block_size // 2, min=0)
        end = torch.clamp(center + block_size // 2, max=torch.tensor(image_size).cuda() - 1)
        # 保存区域中所有点的坐标
        block_coords = get_cuboid_points(start, end)  # 转换为全局坐标
        # sampled_blocks_coords.append(block_coords)
        sampled_blocks_coords.extend(block_coords)

    return torch.tensor(sampled_blocks_coords)

def sparse_sampling_prep(img_size, _coords):
    _coords = _coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    _coords=_coords.to('cuda:0')
    for idx_, sha in enumerate(img_size):
        _coords[...,idx_] = 2 * (_coords[...,idx_]/(sha-1) - 0.5)
    _coords_out = _coords.contiguous()[..., [2,1,0]]

    return  _coords_out

def global_block_adam_nofu(
    img_fixed: Union[torch.Tensor, np.ndarray, sitk.Image],
    img_moving: Union[torch.Tensor, np.ndarray, sitk.Image],
    g_displacements: Union[torch.Tensor, np.ndarray, sitk.Image],
    mind_r: int = 2,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    selected_niter: int = 80,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    use_mask: bool = False,
) -> None:
    """Coupled convex optimisation with adam instance optimisation"""
    img_fixed = validate_image(img_fixed)
    img_moving = validate_image(img_moving)
    img_fixed = img_fixed.float()
    img_moving = img_moving.float()

    if use_mask:
        mask_fixed = get_mask(img_fixed).float()
        mask_moving = get_mask(img_moving).float()
    else:
        mask_fixed = None
        mask_moving = None

    H, W, D = img_fixed.shape

    # compute features and downsample (using average pooling)
    with torch.no_grad():

        features_fix1, features_mov = extract_features(img_fixed=img_fixed,
                                                      img_moving=img_moving,
                                                      mind_r=mind_r,
                                                      mind_d=mind_d,
                                                      use_mask=use_mask,
                                                      mask_fixed=mask_fixed,
                                                      mask_moving=mask_moving)
       
    
    g_displacements = torch.tensor(g_displacements, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    g_displacements = g_displacements.unsqueeze(0).permute(0, 4, 1, 2, 3)  # 重新整理为 (N, C, D, H, W)


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = torch.cat((F.avg_pool3d(features_fix1,grid_sp_adam,stride=grid_sp_adam),F.avg_pool3d(features_fix1,grid_sp_adam,stride=grid_sp_adam)),1)
            patch_features_mov = torch.cat((F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam),F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam)),1)
            

        criterion_cmifs = GmiLoss_soft().to(patch_features_fix.device)
            
        img_movingr = F.interpolate(img_moving.unsqueeze(0).unsqueeze(0),
                                            size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                            mode='trilinear', align_corners=False)
        img_movingr = img_movingr.squeeze(0).squeeze(0)
        img_fixedr = F.interpolate(img_fixed.unsqueeze(0).unsqueeze(0),
                                           size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                           mode='trilinear', align_corners=False)
        img_fixedr = img_fixedr.squeeze(0).squeeze(0)

        # create optimisable displacement grid
        disp_lr = F.interpolate(g_displacements, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                        mode='trilinear', align_corners=False)
            
        net = nn.Sequential(
                    nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
            
        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                                      (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                      align_corners=False)
            
        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
                    optimizer.zero_grad()
            
                    block_imag=block_filtering_and_sampling(img_fixedr, 3)
                    image_size=[H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam]
                    coords_ori=sample_block_coord(block_imag, image_size, 3, 200)
                    
                    disp_sample = F.avg_pool3d(
                        F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3,
                        stride=1, padding=1).permute(0, 2, 3, 4, 1)

                    dy = torch.abs(disp_sample[:, 1:, :, :] - disp_sample[:, :-1, :, :]) **2
                    dx = torch.abs(disp_sample[:, :, 1:, :] - disp_sample[:, :, :-1, :]) **2
                    dz = torch.abs(disp_sample[:, :, :, 1:] - disp_sample[:, :, :, :-1]) **2

                    reg_loss = lambda_weight * (dy.mean() + dx.mean() + dz.mean())

            
                    scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                          (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
                    grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
                    
                    scale_w = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                          (D // grid_sp_adam - 1) / 2]).view(1,3,1,1,1).cuda()
                    
                    disp = grid0.permute(0, 4, 1, 2, 3).cuda().float() + (disp_sample.permute(0, 4, 1, 2, 3) / scale_w).float()
                    
                    reg_model = registerSTModel([H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam], 'nearest').cuda()
                    image_warped=reg_model(img_movingr.cuda().view(1, 1, H //grid_sp_adam, W //grid_sp_adam, D //grid_sp_adam), disp)
                    
                    coords_ = sparse_sampling_prep(image_size, coords_ori.float())
                    
                    sampled_x = F.grid_sample(image_warped.cuda().float(), coords_.float(),
                                              mode='bilinear').squeeze()
                    
                    sampled_y = F.grid_sample(img_fixedr.unsqueeze(0).unsqueeze(0).cuda().float(), coords_.float(),
                                              mode='bilinear').squeeze()
                    
                    sampled_x = sampled_x.unsqueeze(0)
                    sampled_y = sampled_y.unsqueeze(0)
                    
                    loss_cmif = -100*criterion_cmifs(sampled_x, sampled_y)
            
                    patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                                      grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam,
                                                                     D // grid_sp_adam, 3).cuda(), align_corners=False,
                                                      mode='bilinear')
            
                    sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 15
                    # loss_mind = 0.5 * sampled_cost.mean()
                    loss_mind = sampled_cost.mean()
                    (loss_mind + reg_loss + loss_cmif).backward()
            
                    optimizer.step()

                    print("iter %d, loss_mind: %.4f, loss_cmif: %.4f, reg_loss: %.4f" % (
                    iter, loss_mind.item(), loss_cmif.item(), reg_loss.item()), end='\r')
            
        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear',
                                        align_corners=False)
            
        if selected_smooth == 5:
                    kernel_smooth = 5
                    padding_smooth = kernel_smooth // 2
                    disp_hr = F.avg_pool3d(
                        F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1),
                                     kernel_smooth,
                                     padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)
            
        if selected_smooth == 3:
                    kernel_smooth = 3
                    padding_smooth = kernel_smooth // 2
                    disp_hr = F.avg_pool3d(
                        F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1),
                                     kernel_smooth,
                                     padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)
    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)
    torch.cuda.empty_cache()

    return displacements

if __name__=="__main__":
    if 'ipykernel' in sys.modules:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", '--datasets_path', type = str, default = "/input/")
    parser.add_argument('--mind_r', type=int, default=3)
    parser.add_argument('--mind_d', type=int, default=3)
    parser.add_argument('--lambda_weight', type=float, default=1.25) #1.25
    parser.add_argument('--grid_sp', type=int, default=6)
    parser.add_argument('--disp_hw', type=int, default=6)
    parser.add_argument('--selected_niter', type=int, default=20)
    parser.add_argument('--selected_smooth', type=int, default=5)
    parser.add_argument('--grid_sp_adam', type=int, default=2)
    parser.add_argument('--ic', choices=('True','False'), default='True')
    parser.add_argument('--use_mask', choices=('True','False'), default='True')
    parser.add_argument('--result_path', type=str, default='/output/')

    args = parser.parse_args()

    fixed_mod = '0000'
    moving_mod = '0001' # 0002

    # select inference cases
    list_case = sorted([k.split('_')[1] for k in os.listdir(args.datasets_path) if f'{moving_mod}.nii' in k])
    print(list_case)
    print(f"Number total cases: {len(list_case)}")

    for case in list_case:
        torch.cuda.synchronize()
        t0 = time.time()
        # Load image using SimpleITK
        fix_path = os.path.join(args.datasets_path, f"ReMIND2Reg_{case}_{fixed_mod}.nii.gz")
        mov_path = os.path.join(args.datasets_path, f"ReMIND2Reg_{case}_{moving_mod}.nii.gz")

        flow4, disp_out4 = convex_adam_nofu(
                path_img_fixed=fix_path,
                path_img_moving=mov_path,
                mind_r=args.mind_r,
                mind_d=args.mind_d,
                lambda_weight=args.lambda_weight,
                grid_sp=args.grid_sp,
                disp_hw=args.disp_hw,
                selected_niter=args.selected_niter,
                selected_smooth=args.selected_smooth,
                grid_sp_adam=args.grid_sp_adam,
                ic=(args.ic == 'True'),
                use_mask=(args.use_mask == 'True'),
                result_path=args.result_path
            )
        _, _, H, W, D = flow4.shape
        affine = nib.load(fix_path).affine
        reg_model = registerSTModel([H, W, D], 'nearest').cuda()
        print('ok')
        img_fixed = torch.from_numpy(nib.load(fix_path).get_fdata()).float()
        img_moving = torch.from_numpy(nib.load(mov_path).get_fdata()).float()

        disp_out = disp_out4
        
        disp_out=global_block_adam_nofu(
                img_fixed=img_fixed,
                img_moving=img_moving,
                g_displacements=disp_out,
                mind_r=args.mind_r,
                mind_d=args.mind_d,
                lambda_weight=args.lambda_weight,
                selected_niter=args.selected_niter,
                selected_smooth=args.selected_smooth,
                grid_sp_adam=args.grid_sp_adam,
                use_mask=(args.use_mask == 'True'),
            )
        disp_nii_t = nib.Nifti1Image(disp_out, affine)
        
        # deformation_field=torch.from_numpy(disp_out)
        # deformation_field=deformation_field.cuda().float().permute(3,0,1,2)
        # deformation_field=deformation_field.unsqueeze(0)
        # out_t = reg_model(img_moving.cuda().view(1, 1, 256, 256, 256), deformation_field)
        # out_nii_t = nib.Nifti1Image(out_t.view(256, 256, 256).cpu().detach().numpy(), affine)

        mov = mov_path.split("_", 1)[1].split(".", 1)[0]
        fix = fix_path.split("_", 1)[1].split(".", 1)[0]

        # nib.save(out_nii_t, os.path.join(args.result_path,'out_' + fix + '_' + mov + '.nii.gz'))
        nib.save(disp_nii_t, os.path.join(args.result_path,'disp_' + fix + '_' + mov + '.nii.gz'))

        torch.cuda.synchronize()
        t1 = time.time()
        case_time = t1 - t0
        print('case time: ', case_time)

        torch.cuda.empty_cache()