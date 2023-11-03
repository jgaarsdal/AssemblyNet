import numpy as np
import torch
from pytorch3d.transforms import Rotate
from scipy.spatial.transform import Rotation

def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]

def jitter_point_cloud(batch_data, p=0.5, sigma=0.001, clip=0.005):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
          or batch_pc: BxPxNx3 (in case of two-path network)
        Return:
          BxNx3 array, jittered batch of point clouds
          or batch_pc: BxPxNx3 (in case of two-path network)
    """
    assert(clip > 0)

    if torch.rand(1) >= p:
        return batch_data
    else:
        if batch_data.ndim == 4: # two-path network
            B, P, N, C = batch_data.shape
            jittered_data = np.clip(sigma * np.random.randn(B, P, N, 3), -1 * clip, clip)
            batch_data[:, :, :, 0:3] += jittered_data
        else:
            B, N, C = batch_data.shape
            jittered_data = np.clip(sigma * np.random.randn(B, N, 3), -1 * clip, clip)
            batch_data[:, :, 0:3] += jittered_data

        return jittered_data

def shift_point_cloud(batch_data, shift_range=0.2):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
          or batch_pc: BxPxNx3 (in case of two-path network)
        Return:
          BxNx3 array, shifted batch of point clouds
          or batch_pc: BxPxNx3 (in case of two-path network)
    """
    if batch_data.ndim == 4: # two-path network
        B, P, N, C = batch_data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
        for batch_index in range(B):
            batch_data[batch_index, :, :, 0:3] += shifts[batch_index, :]
    else:
        B, N, C = batch_data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
        for batch_index in range(B):
            batch_data[batch_index, :, 0:3] += shifts[batch_index, :]

    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.66, scale_high=1.5):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNxC array, original batch of point clouds
            or batch_pc: BxPxNxC (in case of two-path network)
        Return:
            BxNxC array, scaled batch of point clouds
            or batch_pc: BxPxNxC (in case of two-path network)
    """
    if batch_data.ndim == 4: # two-path network
        B, P, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index, :, :, 0:3] *= scales[batch_index]
    else:
        B, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index, :, 0:3] *= scales[batch_index]

    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    ''' or batch_pc: BxPxNx3 (in case of two-path network)'''
    N_index = batch_pc.ndim - 2
    for b in range(batch_pc.shape[0]):
        dropout_ratio_0 =  np.random.random() * max_dropout_ratio # 0~0.875
        drop_idx_0 = np.where(np.random.random((batch_pc.shape[N_index])) <= dropout_ratio_0)[0]

        if batch_pc.ndim == 4: # two-path network
            dropout_ratio_1 =  np.random.random() * max_dropout_ratio # 0~0.875
            drop_idx_1 = np.where(np.random.random((batch_pc.shape[N_index])) <= dropout_ratio_1)[0]

            if len(drop_idx_0) > 0:
                batch_pc[b, 0, drop_idx_0, :] = batch_pc[b, 0, 0, :] # set to the first point

            if len(drop_idx_1) > 0:
                batch_pc[b, 1, drop_idx_1, :] = batch_pc[b, 1, 0, :] # set to the first point
        else:
            batch_pc[b, drop_idx_0, :] = batch_pc[b, 0, :] # set to the first point

    return batch_pc

# For the classifier
def random_rotate_point_cloud_cls(p, targets, points, directions, num_class=26):
    rotated_targets = torch.randint(0, num_class, (points.shape[0],))
    rotated_points = torch.clone(points)

    for batch_index in range(len(targets)):
        if torch.rand(1) >= p:
            rotated_targets[batch_index] = targets[batch_index]
        else:
            target_dir = directions[targets[batch_index]]
            new_target_dir = directions[rotated_targets[batch_index]]
            
            rot_mat = Rotation.align_vectors([new_target_dir], [target_dir])[0]
            rot_mat = rot_mat.as_matrix()
            rot_mat = torch.from_numpy(rot_mat)
            pc_rot = Rotate(rot_mat)

            if points.ndim == 4: # In case of two-path network
                B, P, N, C = points.shape
                rotated_points[batch_index, 0, :, 0:3] = pc_rot.transform_points(points[batch_index, 0, :, 0:3])
                rotated_points[batch_index, 1, :, 0:3] = pc_rot.transform_points(points[batch_index, 1, :, 0:3])
                
                if C > 3: # normals
                    rotated_points[batch_index, 0, :, 3:6] = pc_rot.transform_points(points[batch_index, 0, :, 3:6])
                    rotated_points[batch_index, 1, :, 3:6] = pc_rot.transform_points(points[batch_index, 1, :, 3:6])

            else:
                B, N, C = points.shape
                rotated_points[batch_index, :, 0:3] = pc_rot.transform_points(points[batch_index, :, 0:3])

                if C > 3: # normals
                    rotated_points[batch_index, :, 3:6] = pc_rot.transform_points(points[batch_index, :, 3:6])

    return rotated_points, rotated_targets

# For the regressor
def random_rotate_point_cloud_reg(p, targets, points):
    rotated_targets = torch.clone(targets)
    rotated_points = torch.clone(points)

    rotation_axes = [ 'x', 'y', 'z' ]
    angles = torch.rand(points.shape[0]) * 360
    axes = torch.randint(0, 3, (points.shape[0],))

    for batch_index in range(len(targets)):
        if torch.rand(1) < p:
            axis_str = rotation_axes[axes[batch_index]]

            rot_mat = Rotation.from_euler(axis_str, angles[batch_index], degrees=True)   

            new_target = rot_mat.apply(targets[batch_index])
            rotated_targets[batch_index] = torch.from_numpy(new_target).cpu()

            rot_mat = rot_mat.as_matrix()
            rot_mat = torch.from_numpy(rot_mat)
            pc_rot = Rotate(rot_mat)

            if points.ndim == 4: # In case of two-path network
                B, P, N, C = points.shape
                rotated_points[batch_index, 0, :, 0:3] = pc_rot.transform_points(points[batch_index, 0, :, 0:3])
                rotated_points[batch_index, 1, :, 0:3] = pc_rot.transform_points(points[batch_index, 1, :, 0:3])
                
                if C > 3: # normals
                    rotated_points[batch_index, 0, :, 3:6] = pc_rot.transform_points(points[batch_index, 0, :, 3:6])
                    rotated_points[batch_index, 1, :, 3:6] = pc_rot.transform_points(points[batch_index, 1, :, 3:6])

            else:
                B, N, C = points.shape
                rotated_points[batch_index, :, 0:3] = pc_rot.transform_points(points[batch_index, :, 0:3])

                if C > 3: # normals
                    rotated_points[batch_index, :, 3:6] = pc_rot.transform_points(points[batch_index, :, 3:6])
        
    return rotated_points, rotated_targets