from data_utils.AssemblyNetDataLoader import AssemblyNetDataLoader
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
import math
from pytorch3d.transforms import Rotate
from scipy.spatial.transform import Rotation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

class_directions = [ 
    [0.0000, 0.0000, 1.0000],      # 0
    [0.0000, 0.0000, -1.0000],     # 1
    
    [0.0000, -0.7071, 0.7071],     # 2
    [0.0000, -1.0000, 0.0000],     # 3
    [0.0000, -0.7071, -0.7071],    # 4
    [0.0000, 0.7071, -0.7071],     # 5
    [0.0000, 1.0000, 0.0000],      # 6
    [0.0000, 0.7071, 0.7071],      # 7
        
    [0.7071, 0.0000, 0.7071],      # 8
    [1.0000, 0.0000, 0.0000],      # 9
    [0.7071, 0.0000, -0.7071],     # 10
    [-0.7071, 0.0000, -0.7071],    # 11
    [-1.0000, 0.0000, 0.0000],     # 12
    [-0.7071, 0.0000, 0.7071],     # 13
        
    [0.5000, -0.7071, 0.5000],     # 14
    [-0.5000, -0.7071, -0.5000],   # 15
    [-0.5000, 0.7071, -0.5000],    # 16
    [0.5000, 0.7071, 0.5000],      # 17
        
    [0.7071, -0.7071, 0.0000],     # 18
    [-0.7071, -0.7071, 0.0000],    # 19
    [-0.7071, 0.7071, 0.0000],     # 20
    [0.7071, 0.7071, 0.0000],      # 21
        
    [0.5000, -0.7071, -0.5000],    # 22
    [-0.5000, -0.7071, 0.5000],    # 23
    [-0.5000, 0.7071, 0.5000],     # 24
    [0.5000, 0.7071, -0.5000]]     # 25


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='2p-dgcnn', help='Model name [default: 2p-dgcnn]',
                        choices = ['dgcnn', 'pointnet', '2p-dgcnn', '2p-pointnet2', 'pointnet2', 'simpleview'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training [default: 32]')
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to a pretrained model')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]', choices = [512, 1024])
    parser.add_argument('--n_knn', default=20, type=int, help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    parser.add_argument('--loss_vector_kernel', type=str, default='cosine_distance', help='Kernel to use for calculating loss [default: cosine_distance]',
                        choices = ['cosine_distance', 'euclidean_distance', 'manhattan_distance'])
    parser.add_argument('--log_dir', type=str, default='reg_test', help='Experiment root [default: reg_test]')
    parser.add_argument('--use_cuda', action='store_true', help='Use the GPU via Cuda [default: True]')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false', help='Use the GPU via Cuda [default: True]')
    parser.set_defaults(use_cuda=True)
    return parser.parse_args()


def dir_cosine_distance(y_true, y_pred):
    cos_dist = np.zeros_like(y_true)

    for i in range(len(y_true)):
        pred = torch.tensor(y_pred[i])
        true = torch.tensor(y_true[i])

        cos_dist[i] = 1.0 - F.cosine_similarity(pred, true, eps=1e-08, dim=-1)

    return cos_dist


def class_cosine_distance(y_true, y_pred, num_classes = 26):
    per_class_dist = np.zeros(num_classes)

    for i in range(num_classes):
        true_class = y_true[y_true == i]
        pred_class = y_pred[y_true == i]
        total_pred = true_class.shape[0]

        if total_pred > 0:
            class_dist = []
            for pred_idx in range(total_pred):
                true_dir = torch.tensor(class_directions[true_class[pred_idx]])
                pred_dir = torch.tensor(class_directions[pred_class[pred_idx]])
                class_dist.append(1.0 - F.cosine_similarity(pred_dir, true_dir, eps=1e-08, dim=-1))
            
            per_class_dist[i] = np.mean(class_dist)
        else:
            per_class_dist[i] = 0.0

    return per_class_dist


def per_class_score(y_true, y_pred, num_classes = 26):
    per_class_acc = np.zeros(num_classes)

    for i in range(num_classes):
        true_class = y_true[y_true == i]
        pred_class = y_pred[y_true == i]
        total_pred = true_class.shape[0]
        # Check if there are instances of the class
        if total_pred > 0:
            correct_pred = np.sum(true_class == pred_class)
            per_class_acc[i] = correct_pred / total_pred
        else:
            per_class_acc[i] = 0.0

    return per_class_acc


def accuracy_score(y_true, y_pred, average='micro'):
    if average == 'micro':
        # Count total correct predictions
        correct_pred = np.sum(y_true == y_pred)
        # Compute total number of predictions
        total_pred = y_true.shape[0]
        # Compute and return micro accuracy
        return correct_pred / total_pred
    
    elif average == 'macro':
        num_classes = 26

    elif average == 'balanced':
        num_classes = len(np.unique(y_true))

    per_class_acc = per_class_score(y_true, y_pred, num_classes)
            
    return np.sum(per_class_acc) / num_classes


def get_nearest_class(directions):
    converted_directions = np.zeros((len(directions)))

    for i in range(len(directions)):
        t_dir = torch.tensor(directions[i])
        smallest_cos_dist = 2.0

        for j in range(len(class_directions)):
            t_class_dir = torch.tensor(class_directions[j])

            cos_dist = 1.0 - F.cosine_similarity(t_dir, t_class_dir, eps=1e-08, dim=-1)
            if cos_dist < smallest_cos_dist:
                converted_directions[i] = j
                smallest_cos_dist = cos_dist

    return converted_directions.astype(int)


def test(args, model, loader, device, prefix='test', num_class=26):
    orig_val_pred = []
    orig_val_true = []
    converted_val_pred = []
    converted_val_true = []
    rotated_conv_val_pred = []
    rotated_conv_val_true = []

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        raw_points, raw_target = data

        points = raw_points.transpose(raw_points.dim() - 1, raw_points.dim() - 2)
        points = points.to(device)

        with torch.cuda.amp.autocast():
            pred = model(points)

        pred = nn.functional.normalize(pred, dim=1).detach().cpu().numpy()
        target = raw_target.squeeze().cpu().numpy()

        orig_val_true.append(target)
        orig_val_pred.append(pred)

        target_to_class = get_nearest_class(target)
        converted_val_true.append(target_to_class)
        converted_val_pred.append(get_nearest_class(pred))

        raw_cls_idx = torch.tensor(target_to_class)
        for class_index in range(num_class):
            rotated_points = torch.clone(raw_points)
            rotated_cls_idx = torch.clone(raw_cls_idx)
            for batch_index in range(len(raw_cls_idx)):
                orig_class = raw_cls_idx[batch_index]

                if class_index == orig_class:
                    continue

                rotated_cls_idx[batch_index] = class_index

                target_dir = class_directions[orig_class]
                new_target_dir = class_directions[class_index]

                rot_mat = Rotation.align_vectors([new_target_dir], [target_dir])[0]
                rot_mat = rot_mat.as_matrix()
                rot_mat = torch.from_numpy(rot_mat)
                pc_rot = Rotate(rot_mat)

                if raw_points.ndim == 4: # In case of two-path network
                    rotated_points[batch_index, 0] = pc_rot.transform_points(raw_points[batch_index, 0])
                    rotated_points[batch_index, 1] = pc_rot.transform_points(raw_points[batch_index, 1])
                else:
                    rotated_points[batch_index] = pc_rot.transform_points(raw_points[batch_index])

            points = rotated_points.transpose(rotated_points.dim() - 1, rotated_points.dim() - 2)
            points = points.to(device)

            with torch.cuda.amp.autocast():
                pred = model(points)

            pred = nn.functional.normalize(pred, dim=1).detach().cpu().numpy()

            rotated_conv_val_true.append(rotated_cls_idx.cpu().numpy())
            rotated_conv_val_pred.append(get_nearest_class(pred))

    orig_val_true = np.concatenate(orig_val_true)
    orig_val_pred = np.concatenate(orig_val_pred)

    converted_val_true = np.concatenate(converted_val_true)
    converted_val_pred = np.concatenate(converted_val_pred)
    converted_val_acc = accuracy_score(converted_val_true, converted_val_pred, average='micro')
    converted_val_macro_acc = accuracy_score(converted_val_true, converted_val_pred, average='macro')
    converted_val_class_acc = accuracy_score(converted_val_true, converted_val_pred, average='balanced')
    converted_val_per_class_acc = per_class_score(converted_val_true, converted_val_pred, num_class)

    rotated_conv_val_true = np.concatenate(rotated_conv_val_true)
    rotated_conv_val_pred = np.concatenate(rotated_conv_val_pred)

    orig_val_cos_dist = dir_cosine_distance(orig_val_true, orig_val_pred)
    converted_val_cos_dist = class_cosine_distance(converted_val_true, converted_val_pred, num_class)

    orig_data = np.concatenate((orig_val_pred, orig_val_true), axis=1).astype(float)
    converted_data = np.asarray([converted_val_pred, converted_val_true]).transpose().astype(int)
    rotated_conv_data = np.asarray([rotated_conv_val_pred, rotated_conv_val_true]).transpose().astype(int)

    # Save to csv files
    if not os.path.exists("csv_data"):
        os.makedirs("csv_data")
    np.savetxt("csv_data/" + prefix + "_raw_reg.csv", orig_data, delimiter=",", fmt='%1.4f')
    np.savetxt("csv_data/" + prefix + "_converted_reg.csv", converted_data, delimiter=",", fmt='%1.1i')
    np.savetxt("csv_data/" + prefix + "_converted_allrot_reg.csv", rotated_conv_data, delimiter=",", fmt='%1.1i')

    return converted_val_acc, converted_val_class_acc, converted_val_macro_acc, converted_val_per_class_acc, orig_val_cos_dist, converted_val_cos_dist


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath('reg')
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile_name = args.model + '_' + timestr
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, logfile_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    if args.pretrained_model is None or args.pretrained_model == '':
        log_string('No pretrained model path was provided...')
        exit()
    
    '''GPU'''
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        torch.cuda.is_available = lambda : False
        device = torch.device('cpu')

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = os.path.join(BASE_DIR, 'data/assemblynet')

    is_two_path = args.model == '2p-dgcnn' or args.model == '2p-pointnet2'
    
    val_dataset = AssemblyNetDataLoader(root=data_path, label_type='reg', class_directions=class_directions, npoint=args.num_point, split='val', two_path=is_two_path)
    test_dataset = AssemblyNetDataLoader(root=data_path, label_type='reg', class_directions=class_directions, npoint=args.num_point, split='test', two_path=is_two_path)
    
    val_dataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    test_dataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)

    regressor = model.get_model(args, output_channels=3, device=device).to(device)

    try:
        checkpoint = torch.load(args.pretrained_model)
        regressor.load_state_dict(checkpoint['model_state_dict'])
        log_string('Loaded pretrained model')
    except:
        log_string('Failed to load pretrained model')
        exit()

    '''VAL DATA'''
    with torch.no_grad():
        converted_val_acc, converted_val_class_acc, converted_val_macro_acc, converted_val_per_class_acc, orig_val_cos_dist, converted_val_cos_dist = test(args, regressor.eval(), val_dataLoader, device, prefix='val')

    log_string('Converted val acc: %.6f, val class acc: %.6f, val macro acc: %.6f' % (converted_val_acc, converted_val_class_acc, converted_val_macro_acc))

    converted_val_per_class_acc = [[i, converted_val_per_class_acc[i]] for i in range(len(converted_val_per_class_acc))]

    orig_val_cos_dist = [[i, orig_val_cos_dist[i]] for i in range(len(orig_val_cos_dist))]

    converted_val_cos_dist = [[i, converted_val_cos_dist[i]] for i in range(len(converted_val_cos_dist))]

    '''TEST DATA'''
    with torch.no_grad():
        converted_test_acc, converted_test_class_acc, converted_test_macro_acc, converted_test_per_class_acc, orig_test_cos_dist, converted_test_cos_dist = test(args, regressor.eval(), test_dataLoader, device, prefix='test')

    log_string('Converted test acc: %.6f, test class acc: %.6f, test macro acc: %.6f' % (converted_test_acc, converted_test_class_acc, converted_test_macro_acc))

    converted_test_per_class_acc = [[i, converted_test_per_class_acc[i]] for i in range(len(converted_test_per_class_acc))]

    orig_test_cos_dist = [[i, orig_test_cos_dist[i]] for i in range(len(orig_test_cos_dist))]

    converted_test_cos_dist = [[i, converted_test_cos_dist[i]] for i in range(len(converted_test_cos_dist))]


if __name__ == '__main__':
    args = parse_args()

    args.pretrained_model = 'saved_checkpoints/reg/reg_2pdgcnn_best_model.pth'

    main(args)