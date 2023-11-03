import numpy as np
import warnings
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .sord import construct_sord_labels

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class AssemblyNetDataLoader(Dataset):
    def __init__(self, root, label_type='cls', class_directions=None, loss_function="CE", sord_kernel="cosine_distance", npoint=512, split='train', two_path=False, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.pc_dir = 'resampled_pointclouds_' + str(self.npoints)
        self.label_dir = 'labels'
        self.two_path = two_path
        self.class_directions = class_directions
        self.loss_function = loss_function
        self.label_type = label_type
        self.labels = {}
        self.datapath = []

        if not self.two_path:
            self.npoints = self.npoints * 2
        
        assert (split == 'train' or split == 'val' or split == 'test')
        assert (self.label_type == 'reg' or self.label_type == 'cls')

        assembly_ids = {}
        assembly_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'assemblynet_train.txt'))]
        assembly_ids['val'] = [line.rstrip() for line in open(os.path.join(self.root, 'assemblynet_val.txt'))]
        assembly_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'assemblynet_test.txt'))]

        assembly_names = assembly_ids[split]

        if self.label_type == 'cls':
            if loss_function == "KL" or loss_function == "Soft-CE":
                num_classes = len(class_directions)
                clsidx_labels = torch.Tensor(range(num_classes)).long()
                self.onehot_labels = F.one_hot(clsidx_labels, num_classes=num_classes)
            elif loss_function == "KL-SORD" or loss_function == "Soft-CE-SORD":
                self.sord_labels = construct_sord_labels(torch.Tensor(class_directions), sord_kernel)

        for assembly in assembly_names:
            if self.label_type == 'cls':
                label_file = os.path.join(self.root, self.label_dir, assembly, assembly)
                label_file += '_cls.txt'
            elif self.label_type == 'reg':
                label_file = os.path.join(self.root, self.label_dir, assembly, assembly)
                label_file += '_reg.txt'

            for part_line in open(label_file):
                part_line_parts = part_line.rstrip().split(',')
                part_id = assembly + ',' + part_line_parts[0].replace('_pc.txt', '')
                
                self.datapath.append([part_id, part_line_parts[0:2]])
                self.labels[part_id] = part_line_parts[2:len(part_line_parts)] if self.label_type == 'reg' else part_line_parts[2]

        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, label) tuple


    def get_loss_label(self, cls_idx):
        if self.loss_function == "KL" or self.loss_function == "Soft-CE":
            # Using one hot labels
            onehot_label = self.onehot_labels[cls_idx]
            return np.array(onehot_label).astype(np.int32)
        elif self.loss_function == "KL-SORD" or self.loss_function == "Soft-CE-SORD":
            # Using SORD labels
            return np.array(self.sord_labels[cls_idx]).astype(np.float32)
        else: # if self.loss_function == "CE" or self.loss_function == "LS" or self.loss_function == "Combined-DiffCE":
            # Using class index labels
            return np.array([cls_idx]).astype(np.int32)
        

    def get_loss_labels(self, cls_idcs):
        labels = []
        for i in range(len(cls_idcs)):
            labels.append(self.get_loss_label(cls_idcs[i]))

        if self.loss_function == "KL-SORD" or self.loss_function == "Soft-CE-SORD":
            return np.array(labels).astype(np.float32)
        else:
            return np.array(labels).astype(np.int32)


    def _get_item(self, index):
        if index in self.cache:
            point_set, label = self.cache[index]
        else:
            fn = self.datapath[index]
            label = self.labels[fn[0]]
            
            if self.label_type == 'cls':
                label = int(label)
                label = self.get_loss_label(label)
            if self.label_type == 'reg':
                label = np.array([label]).astype(np.float32)

            part_pc_file = os.path.join(self.root, self.pc_dir, fn[0].split(',')[0].replace(' ', '_'), fn[1][0].replace(' ', '_'))
            assembly_pc_file = os.path.join(self.root, self.pc_dir, fn[0].split(',')[0].replace(' ', '_'), fn[1][1].replace(' ', '_'))

            if not os.path.isfile(part_pc_file):
                print('Could not find the part file %s'%(part_pc_file))

            if not os.path.isfile(assembly_pc_file):
                print('Could not find the assembly file %s'%(assembly_pc_file))

            part_point_set = np.loadtxt(part_pc_file, delimiter=',').astype(np.float32)
            assembly_point_set = np.loadtxt(assembly_pc_file, delimiter=',').astype(np.float32)

            part_point_set = part_point_set[0:self.npoints, 0:3]
            assembly_point_set = assembly_point_set[0:self.npoints, 0:3]

            if self.two_path:
                np.random.shuffle(part_point_set)
                np.random.shuffle(assembly_point_set)

                point_set = np.array([part_point_set, assembly_point_set])
            else:
                point_set = np.concatenate((part_point_set, assembly_point_set), axis=0)

                # Ahould already be normalized
                #point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

                np.random.shuffle(point_set)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, label)
        
        return point_set, label
    

    def __len__(self):
        return len(self.datapath)


    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    '''DATA LOADING'''
    data_path = '../data/assemblynet/'

    train_dataset = AssemblyNetDataLoader(root=data_path, npoint=1024, split='train')
    train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4)

    for point,label in train_dataLoader:
        print(point.shape)
        print(label.shape)
        break