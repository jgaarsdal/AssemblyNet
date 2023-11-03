from data_utils.AssemblyNetDataLoader import AssemblyNetDataLoader
import argparse
import os
import torch
import torch.nn as nn
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
from data_utils import provider
from data_utils.loss_functions_reg import select_loss
import importlib
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

rotation_axes = [ 'x', 'y', 'z' ]

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='2p-dgcnn', help='Model name [default: 2p-dgcnn]',
                        choices = ['dgcnn', 'pointnet', '2p-dgcnn', '2p-pointnet2', 'pointnet2', 'simpleview'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training [default: 32]')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size in validation [default: 32]')
    parser.add_argument('--epoch', default=250, type=int, help='Number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate [default: 1e-4]')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Momentum to use for SGD optimizer [default: 0.9]')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Epsilon to use for Adam and AdamW optimizer [default: 1e-8]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: Adam]',
                        choices = ['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--loss_vector_kernel', type=str, default='cosine_distance', help='Kernel to use for calculating loss [default: cosine_distance]',
                        choices = ['cosine_distance', 'euclidean_distance', 'manhattan_distance'])
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]', choices = [512, 1024])
    parser.add_argument('--log_dir', type=str, default='reg', help='Experiment root [default: reg]')
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

def test(model, criterion, loader, device):
    val_loss = 0.0
    count = 0.0
    mae_score = nn.L1Loss()
    all_mae = []

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        
        target = target[:, 0]

        if args.val_rotation:
            points, target = provider.random_rotate_point_cloud_reg(0.3, target, points)

        points = points.transpose(points.dim() - 1, points.dim() - 2)
        points, target = points.to(device), target.to(device)
        
        batch_size = points.size()[0]

        #target = nn.functional.normalize(target, dim=1)

        with torch.cuda.amp.autocast():
            pred = model(points)
            pred = nn.functional.normalize(pred, dim=1)
            loss = criterion(pred, target.float())
            mae = mae_score(target.float(), pred)

        all_mae.append(mae)
        count += batch_size
        val_loss += loss.item() * batch_size

    val_loss = val_loss * 1.0 / count
    val_mae = torch.mean(torch.stack(all_mae))

    return val_mae, val_loss


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
    # Scale learning rate with batch size (higher batch size == higher learning rate)
    # Assuming batch size 32 as the base
    args.learning_rate = (args.batch_size / 32.0) * args.learning_rate

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
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    args = parse_args()
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

    train_dataset = AssemblyNetDataLoader(root=data_path, label_type='reg', npoint=args.num_point, split='train', two_path=is_two_path)
    val_dataset = AssemblyNetDataLoader(root=data_path, label_type='reg', npoint=args.num_point, split='val', two_path=is_two_path)
    train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_dataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, num_workers=8, drop_last=True)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)

    regressor = model.get_model(args, output_channels=3, device=device).to(device)
    criterion = select_loss(vector_kernel=args.loss_vector_kernel).to(device)

    regressor.apply(init_weights)

    mae_score = nn.L1Loss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            regressor.parameters(),
            lr=args.learning_rate,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            regressor.parameters(),
            lr=args.learning_rate,
            weight_decay=args.decay_rate,
            eps=1e-4
        )
    else:
        optimizer = torch.optim.SGD(
            regressor.parameters(),
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
            regressor.load_state_dict(checkpoint['model_state_dict'])
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
    best_val_loss = 1000000.0
    mean_mae = []

    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        if not args.restarts:
            scheduler.step()
        
        regressor.train()

        train_loss = 0.0
        count = 0.0

        for batch_id, data in tqdm(enumerate(train_dataLoader, 0), total=len(train_dataLoader), smoothing=0.9):
            points, target = data

            target = target[:, 0]

            # Data augmentation

            if args.rotation:
                points, target = provider.random_rotate_point_cloud_reg(0.3, target, points)

            points = points.data.numpy()

            if args.dropout:
                points = provider.random_point_dropout(points, max_dropout_ratio=0.5)
            
            points = provider.random_scale_point_cloud(points, scale_low=0.66, scale_high=1.5)
            points = provider.shift_point_cloud(points, shift_range=0.2)

            if args.jitter:
                points = provider.jitter_point_cloud(points)

            points = torch.Tensor(points)    
            points = points.transpose(points.dim() - 1, points.dim() - 2)

            points, target = points.to(device), target.to(device)

            batch_size = points.size()[0]

            #target = nn.functional.normalize(target, dim=1)

            # Training
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred = regressor(points)
                pred = nn.functional.normalize(pred, dim=1)
                loss = criterion(pred, target)
                mae = mae_score(target, pred)

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_value_(regressor.parameters(), clip_value=args.clip_val)

            scaler.step(optimizer)
            scaler.update()

            if args.restarts:
                scheduler.step(epoch + batch_id / iters)

            count += batch_size
            train_loss += loss.item() * batch_size
            mean_mae.append(torch.mean(mae))
        
        train_loss = train_loss * 1.0 / count
        train_mae = torch.round(torch.mean(torch.stack(mean_mae)), decimals=2)
        
        log_string('Train epoch %d, loss: %.6f, train mae: %.6f' % (epoch, train_loss, train_mae))

        if args.save_latest:
            # Save latest model
            logger.info('Save model for this epoch...')
            savepath = str(checkpoints_dir) + '/' + 'latest_model.pth'
            log_string('Saving latest model at %s'% savepath)
            state = {
                'epoch': epoch,
                'instance_loss': train_loss,
                'instance_mae': train_mae,
                'model_state_dict': regressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(state, savepath)

        with torch.no_grad():
            val_mae, val_loss = test(regressor.eval(), criterion, val_dataLoader, device)

            # Using the cosine distance loss as the 'accuracy' score
            if (val_loss <= best_val_loss):
                best_val_loss = val_loss
                best_epoch = epoch + 1

            log_string('Test epoch %d, loss: %.6f, test mae: %.6f' % (epoch, val_loss, val_mae))

            # Using the cosine distance loss as the 'accuracy' score
            if (val_loss <= best_val_loss):
                logger.info('Save new best model...')
                savepath = str(checkpoints_dir) + '/' + 'best_model.pth'
                log_string('Saving best model at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_mae': val_mae,
                    'instance_loss': val_loss,
                    'model_state_dict': regressor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)