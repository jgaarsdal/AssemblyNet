from data_utils.AssemblyNetDataLoader import AssemblyNetDataLoader
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
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
    parser.add_argument('--loss_function', type=str, default='LS', help='Loss function used for training [default: LS]',
                        choices = ['CE', 'LS', 'KL', 'Soft-CE', 'KL-SORD', 'Soft-CE-SORD', 'Combined-DiffCE'])
    parser.add_argument('--sord_kernel', type=str, default='cosine_distance', help='Kernel to use for calculating SORD soft labels [default: cosine_distance]',
                        choices = ['cosine_distance', 'euclidean_distance', 'manhattan_distance'])
    parser.add_argument('--log_dir', type=str, default='cls_test', help='Experiment root [default: cls_test]')
    parser.add_argument('--use_cuda', action='store_true', help='Use the GPU via Cuda [default: True]')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false', help='Use the GPU via Cuda [default: True]')
    parser.set_defaults(use_cuda=True)
    return parser.parse_args()


def cosine_distance(y_true, y_pred, num_classes = 26):
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


def test(args, model, loader, device, prefix='test', num_class=26):
    orig_val_pred = []
    orig_val_true = []
    rotated_val_pred = []
    rotated_val_true = []

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        raw_points, raw_targets = data
        raw_targets = raw_targets.squeeze()

        points = raw_points.transpose(raw_points.dim() - 1, raw_points.dim() - 2)
        points = points.to(device)

        with torch.cuda.amp.autocast():
            pred = model(points)

        pred_choice = pred.data.max(1)[1].detach().cpu().numpy()

        orig_val_true.append(raw_targets.cpu().numpy())
        orig_val_pred.append(pred_choice)

        for class_index in range(num_class):
            rotated_points = torch.clone(raw_points)
            rotated_cls_idx = torch.clone(raw_targets)
            for batch_index in range(len(raw_targets)):
                orig_class = raw_targets[batch_index]

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

            pred_choice = pred.data.max(1)[1].detach().cpu().numpy()

            rotated_val_true.append(rotated_cls_idx.cpu().numpy())
            rotated_val_pred.append(pred_choice)

    orig_val_true = np.concatenate(orig_val_true)
    orig_val_pred = np.concatenate(orig_val_pred)
    val_acc = accuracy_score(orig_val_true, orig_val_pred, average='micro')
    val_macro_acc = accuracy_score(orig_val_true, orig_val_pred, average='macro')
    val_class_acc = accuracy_score(orig_val_true, orig_val_pred, average='balanced')

    rotated_val_true = np.concatenate(rotated_val_true)
    rotated_val_pred = np.concatenate(rotated_val_pred)
    rotated_val_acc = accuracy_score(rotated_val_true, rotated_val_pred, average='micro')
    rotated_val_macro_acc = accuracy_score(rotated_val_true, rotated_val_pred, average='macro')
    rotated_val_class_acc = accuracy_score(rotated_val_true, rotated_val_pred, average='balanced')
    rotated_val_per_class_acc = per_class_score(rotated_val_true, rotated_val_pred, num_class)

    val_cos_dist = cosine_distance(rotated_val_true, rotated_val_pred, num_class)

    no_rot_data = np.asarray([orig_val_pred, orig_val_true]).transpose().astype(int)
    rot_data = np.asarray([rotated_val_pred, rotated_val_true]).transpose().astype(int)

    # Save to csv files
    if not os.path.exists("csv_data"):
        os.makedirs("csv_data")
    np.savetxt("csv_data/" + prefix + "_norot_cls.csv", no_rot_data, delimiter=",", fmt='%1.1i')
    np.savetxt("csv_data/" + prefix + "_allrot_cls.csv", rot_data, delimiter=",", fmt='%1.1i')

    return val_acc, val_class_acc, val_macro_acc, rotated_val_acc, rotated_val_macro_acc, rotated_val_class_acc, rotated_val_per_class_acc, val_cos_dist


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath('cls')
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
    
    val_dataset = AssemblyNetDataLoader(root=data_path, label_type='cls', class_directions=class_directions, loss_function=args.loss_function, 
                                         sord_kernel=args.sord_kernel, npoint=args.num_point, split='val', two_path=is_two_path)
    test_dataset = AssemblyNetDataLoader(root=data_path, label_type='cls', class_directions=class_directions, loss_function=args.loss_function, 
                                         sord_kernel=args.sord_kernel, npoint=args.num_point, split='test', two_path=is_two_path)
    
    val_dataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    test_dataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)

    '''MODEL LOADING'''
    num_class = 26
    model = importlib.import_module(args.model)

    classifier = model.get_model(args, output_channels=num_class, device=device).to(device)

    try:
        checkpoint = torch.load(args.pretrained_model)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Loaded pretrained model')
    except:
        log_string('Failed to load pretrained model')
        exit()

    '''VAL DATA'''
    with torch.no_grad():
        val_acc, val_class_acc, val_macro_acc, rotated_val_acc, rotated_val_macro_acc, rotated_val_class_acc, rotated_val_per_class_acc, val_cos_dist = test(args, classifier.eval(), val_dataLoader, device, prefix='val')

    log_string('Val acc: %.6f, val class acc: %.6f, val macro acc: %.6f' % (val_acc, val_class_acc, val_macro_acc))
    log_string('Rotated val acc: %.6f, rotated val class acc: %.6f, rotated val macro acc: %.6f' % (rotated_val_acc, rotated_val_class_acc, rotated_val_macro_acc))

    rotated_val_per_class_acc = [[i, rotated_val_per_class_acc[i]] for i in range(len(rotated_val_per_class_acc))]
    
    val_cos_dist = [[i, val_cos_dist[i]] for i in range(len(val_cos_dist))]

    '''TEST DATA'''
    with torch.no_grad():
        test_acc, test_class_acc, test_macro_acc, rotated_test_acc, rotated_test_macro_acc, rotated_test_class_acc, rotated_test_per_class_acc, test_cos_dist = test(args, classifier.eval(), test_dataLoader, device, prefix='test')

    log_string('Test acc: %.6f, test class acc: %.6f, test macro acc: %.6f' % (test_acc, test_class_acc, test_macro_acc))
    log_string('Rotated test acc: %.6f, rotated test class acc: %.6f, rotated test macro acc: %.6f' % (rotated_test_acc, rotated_test_class_acc, rotated_test_macro_acc))

    rotated_test_per_class_acc = [[i, rotated_test_per_class_acc[i]] for i in range(len(rotated_test_per_class_acc))]

    test_cos_dist = [[i, test_cos_dist[i]] for i in range(len(test_cos_dist))]


if __name__ == '__main__':
    args = parse_args()
    args.pretrained_model = 'saved_checkpoints/cls/cls_2pdgcnn_best_model.pth'

    main(args)