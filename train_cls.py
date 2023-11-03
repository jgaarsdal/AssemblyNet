from data_utils.AssemblyNetDataLoader import AssemblyNetDataLoader
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
from data_utils import provider
from data_utils.loss_functions_cls import select_loss
import importlib
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


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
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size in validation [default: 32]')
    parser.add_argument('--epoch', default=250, type=int, help='Number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=0.0001, help='Decay rate [default: 0.0001]')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Momentum to use for SGD optimizer [default: 0.9]')
    parser.add_argument('--adam_eps', type=float, default=0.0001, help='Epsilon to use for Adam and AdamW optimizer [default: 0.0001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: Adam]',
                        choices = ['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--loss_function', type=str, default='LS', help='Loss function used for training [default: LS]',
                        choices = ['CE', 'LS', 'KL', 'Soft-CE', 'KL-SORD', 'Soft-CE-SORD', 'Combined-DiffCE'])
    parser.add_argument('--sord_kernel', type=str, default='cosine_distance', help='Kernel to use for calculating SORD soft labels [default: cosine_distance]',
                        choices = ['cosine_distance', 'euclidean_distance', 'manhattan_distance'])
    parser.add_argument('--alpha', type=float, default=0.1, help='Used by the combined loss function to decide how much weight is put on the two terms [default: 0.1]')
    parser.add_argument('--smoothing', type=float, default=0.1, help='The smoothing value used for label smoothing [default: 0.1]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]', choices = [512, 1024])
    parser.add_argument('--log_dir', type=str, default='cls', help='Experiment root [default: cls]')
    parser.add_argument('--n_knn', default=20, type=int, help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    parser.add_argument('--clip_val', default=1.0, type=float, help='The clip value for the gradients [default: 1.0]')
    parser.add_argument('--continue_from', default='none', type=str, help='Continue from the best stored model (if any) or the latest [default: none]',
                        choices = ['best', 'latest', 'none'])
    parser.add_argument('--use_cuda', action='store_true', help='Use the GPU via Cuda [default: True]')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false', help='Use the GPU via Cuda [default: True]')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--save_latest', action='store_true', help='Save the latest model after each epoch [default: False]')
    parser.add_argument('--no-save_latest', dest='save_latest', action='store_false', help='Save the latest model after each epoch [default: False]')
    parser.set_defaults(save_latest=False)
    parser.add_argument('--rotation', action='store_true', help='Should point cloud rotation augmentation be used [default: True]')
    parser.add_argument('--no-rotation', dest='rotation', action='store_false', help='Should point cloud rotation augmentation be used [default: True]')
    parser.set_defaults(rotation=True)
    parser.add_argument('--val_rotation', action='store_true', help='Should point cloud rotation augmentation be used for the validation [default: True]')
    parser.add_argument('--no-val_rotation', dest='val_rotation', action='store_false', help='Should point cloud rotation augmentation be used for the validation [default: True]')
    parser.set_defaults(val_rotation=True)
    parser.add_argument('--jitter', action='store_true', help='Should point jitter augmentation be used [default: False]')
    parser.add_argument('--no-jitter', dest='jitter', action='store_false', help='Should point jitter augmentation be used [default: False]')
    parser.set_defaults(jitter=False)
    parser.add_argument('--dropout', action='store_true', help='Should dropout point augmentation be used [default: True]')
    parser.add_argument('--no-dropout', dest='dropout', action='store_false', help='Should dropout point augmentation be used [default: True]')
    parser.set_defaults(dropout=True)
    parser.add_argument('--restarts', action='store_true', help='Should the cosine annealing scheduler use warm restarts [default: False]')
    parser.add_argument('--no-restarts', dest='restarts', action='store_false', help='Should the cosine annealing scheduler use warm restarts [default: False]')
    parser.set_defaults(restarts=False)
    return parser.parse_args()


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
            
    return np.sum(per_class_acc) / num_classes


def test(args, model, criterion, loader, get_loss_labels, device, num_class=26):
    val_loss = 0.0
    count = 0.0
    val_pred = []
    val_true = []

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        
        if args.val_rotation:
            points, cls_idx = provider.random_rotate_point_cloud_cls(0.3, target, points, class_directions, num_class)
            target = torch.tensor(get_loss_labels(cls_idx))

        if args.loss_function == "CE" or args.loss_function == "LS" or args.loss_function == "Combined-DiffCE":
            # Using class index labels
            target = target[:, 0].long()
        elif args.loss_function == "KL" or args.loss_function == "Soft-CE":
            # Using one hot labels
            target = target.float()

        points = points.transpose(points.dim() - 1, points.dim() - 2)
        points, target = points.to(device), target.to(device)

        batch_size = points.size()[0]

        with torch.cuda.amp.autocast():
            pred = model(points)
            loss = criterion(pred, target)

        pred_choice = pred.data.max(1)[1]
        
        count += batch_size
        val_loss += loss.item() * batch_size
        val_true.append(target.cpu().numpy())
        val_pred.append(pred_choice.detach().cpu().numpy())

    val_loss = val_loss * 1.0 / count

    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)
    val_acc = accuracy_score(val_true, val_pred, average='micro')
    val_macro_acc = accuracy_score(val_true, val_pred, average='macro')
    val_class_acc = accuracy_score(val_true, val_pred, average='balanced')

    return val_loss, val_acc, val_class_acc, val_macro_acc


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''Learning Rate'''
    # Scale learning rate with batch size
    # Assuming batch size 32 as the base
    args.learning_rate = (args.batch_size / 32.0) * args.learning_rate

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
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
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

    train_dataset = AssemblyNetDataLoader(root=data_path, label_type='cls', class_directions=class_directions, loss_function=args.loss_function, 
                                          sord_kernel=args.sord_kernel, npoint=args.num_point, split='train', two_path=is_two_path)
    val_dataset = AssemblyNetDataLoader(root=data_path, label_type='cls', class_directions=class_directions, loss_function=args.loss_function, 
                                        sord_kernel=args.sord_kernel, npoint=args.num_point, split='val', two_path=is_two_path)
    train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_dataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, drop_last=True, num_workers=8)

    '''MODEL LOADING'''
    num_class = 26
    model = importlib.import_module(args.model)

    classifier = model.get_model(args, output_channels=num_class, device=device).to(device)
    criterion = select_loss(loss_function=args.loss_function, smoothing=True, smoothing_val=args.smoothing, alpha=args.alpha, reduction="mean", device=device).to(device)

    classifier.apply(init_weights)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.decay_rate,
            eps=1e-4
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            weight_decay=args.decay_rate
        )

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    if not args.continue_from == 'none':
        if args.continue_from == 'best' or args.continue_from == 'latest':
            existing_model = 'best_model.pth' if args.continue_from == 'best' else 'latest_model.pth'
            existing_model = str(experiment_dir) + '/checkpoints/' + existing_model

        try:
            checkpoint = torch.load(existing_model)
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            log_string('Use pretrained model')
        except:
            log_string('No existing model, starting training from scratch...')

    if args.restarts:
        scheduler = CosineAnnealingWarmRestarts(optimizer, 5, (args.epoch - start_epoch), eta_min=args.learning_rate)
        iters = len(train_dataLoader)
    else:
        scheduler = CosineAnnealingLR(optimizer, (args.epoch - start_epoch), eta_min=args.learning_rate)

    global_epoch = 0
    best_val_acc = 0.0
    best_val_class_acc = 0.0
    best_val_macro_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        if not args.restarts:
            scheduler.step()

        classifier.train()

        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []

        for batch_id, data in tqdm(enumerate(train_dataLoader, 0), total=len(train_dataLoader), smoothing=0.9):
            points, target = data
            
            # Data augmentation

            if args.rotation:
                points, cls_idx = provider.random_rotate_point_cloud_cls(0.3, target, points, class_directions, num_class)
                target = torch.tensor(train_dataset.get_loss_labels(cls_idx))
            
            if args.loss_function == "CE" or args.loss_function == "LS" or args.loss_function == "Combined-DiffCE":
                # Using class index labels
                target = target[:, 0].long()
            elif args.loss_function == "KL" or args.loss_function == "Soft-CE":
                # Using one hot labels
                target = target.float()
            
            points = points.data.numpy()
            
            if args.dropout:
                points = provider.random_point_dropout(points, max_dropout_ratio=0.5)
            
            points = provider.random_scale_point_cloud(points, scale_low=0.66, scale_high=1.5)
            points = provider.shift_point_cloud(points, shift_range=0.2)

            if args.jitter:
                points = provider.jitter_point_cloud(points, p=0.5)

            points = torch.Tensor(points)
            points = points.transpose(points.dim() - 1, points.dim() - 2)

            points, target = points.to(device), target.to(device)

            batch_size = points.size()[0]

            # Training
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred = classifier(points)
                loss = criterion(pred, target)

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_value_(classifier.parameters(), clip_value=args.clip_val)

            scaler.step(optimizer)
            scaler.update()

            if args.restarts:
                scheduler.step(epoch + batch_id / iters)

            pred_choice = pred.data.max(1)[1]

            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(target.cpu().numpy())
            train_pred.append(pred_choice.detach().cpu().numpy())

        train_loss = train_loss * 1.0 / count

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = accuracy_score(train_true, train_pred, average='micro')
        train_macro_acc = accuracy_score(train_true, train_pred, average='macro')
        train_class_acc = accuracy_score(train_true, train_pred, average='balanced')

        log_string('Train epoch %d, loss: %.6f, train acc: %.6f, train class acc: %.6f, train macro acc: %.6f' % (epoch, train_loss, train_acc, train_class_acc, train_macro_acc))

        if args.save_latest:
            # Save latest model
            logger.info('Save model for this epoch...')
            savepath = str(checkpoints_dir) + '/' + 'latest_model.pth'
            log_string('Saving latest model at %s'% savepath)
            state = {
                'epoch': epoch,
                'instance_acc': train_acc,
                'class_acc': train_class_acc,
                'macro_acc': train_macro_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(state, savepath)

        with torch.no_grad():
            val_loss, val_acc, val_class_acc, val_macro_acc = test(args, classifier.eval(), criterion, val_dataLoader, val_dataset.get_loss_labels, device)

        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_epoch = epoch + 1

        if (val_class_acc >= best_val_class_acc):
            best_val_class_acc = val_class_acc

        if (val_macro_acc >= best_val_macro_acc):
            best_val_macro_acc = val_macro_acc

        log_string('Val epoch %d, loss: %.6f, val acc: %.6f, val class acc: %.6f, val macro acc: %.6f' % (epoch, val_loss, val_acc, val_class_acc, val_macro_acc))

        if (val_acc >= best_val_acc):
            logger.info('Save new best model...')
            savepath = str(checkpoints_dir) + '/' + 'best_model.pth'
            log_string('Saving best model at %s'% savepath)
            state = {
                'epoch': best_epoch,
                'instance_acc': val_acc,
                'class_acc': val_class_acc,
                'macro_acc': val_macro_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(state, savepath)
        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)