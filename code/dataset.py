import os
import os.path
import numpy as np
import sys

class MotionDataset():
    def __init__(self, 
    data_path,
    train_list,
    test_list,
    batch_size = 8, 
    num_point = 2048, 
    num_frame = 5,
    num_seg = 10,
    split='train', 
    cache_size=50000):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_point = num_point
        self.num_frame = num_frame

        self.shape_cat = {}
        self.split = split
        self.shape_cat['train'] = [line.rstrip().split(' ') for line in open(os.path.join(self.data_path, train_list))]
        self.shape_cat['test'] = [line.rstrip().split(' ') for line in open(os.path.join(self.data_path, test_list))]
        assert(split=='train' or split=='test')

        self.datapath = [os.path.join(self.data_path, 'points', x[0])+'.pts' for x in self.shape_cat[split]]
        self.segpath = [os.path.join(self.data_path, 'seg', x[0][:-4])+'.seg' for x in self.shape_cat[split]]
        self.gt_frames = [int(x[1]) for x in self.shape_cat[split]]

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to tuple

    def __getitem__(self, index): 
        if index in self.cache:
            pc, pc_target, disp_target, mov_seg, part_seg = self.cache[index]
        else:
            pc = np.loadtxt(self.datapath[index],delimiter=' ').astype(np.float32)
            pc_target = np.zeros((self.num_frame, self.num_point, 3), dtype=np.float32)
            disp_target = np.zeros((self.num_frame, self.num_point, 3), dtype=np.float32)
            cur_frame_idx = int(self.datapath[index][-6])*10 + int(self.datapath[index][-5])
            assert(self.gt_frames[index] > self.num_frame)

            if cur_frame_idx < (self.gt_frames[index] - self.num_frame):
                for i in range(self.num_frame):
                    pc_target[i,:,:] = np.loadtxt(self.datapath[index][:-6]+'%02d.pts'%(cur_frame_idx+i+1),delimiter=' ').astype(np.float32)
            else:
                for i in range(self.gt_frames[index]-1 - cur_frame_idx):
                    pc_target[i,:,:] = np.loadtxt(self.datapath[index][:-6]+'%02d.pts'%(cur_frame_idx+i+1),delimiter=' ').astype(np.float32)
                for i in range(self.gt_frames[index]-1 - cur_frame_idx, self.num_frame):
                    pc_target[i,:,:] = np.loadtxt(self.datapath[index][:-6]+'%02d.pts'%(self.gt_frames[index]-1),delimiter=' ').astype(np.float32)

            disp_target[0,:,:] = pc_target[0,:,:] - pc
            for i in range(1, self.num_frame):
                disp_target[i,:,:] = pc_target[i,:,:] - pc_target[i-1,:,:]

            part_seg = np.loadtxt(self.segpath[index]).astype(np.int32)
            mov_seg=np.zeros(part_seg.shape)
            mov_seg[part_seg>=1] = 1

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pc, pc_target, disp_target, mov_seg, part_seg)
        return pc, pc_target, disp_target, mov_seg, part_seg

    def __len__(self):
        return len(self.datapath)

    def get_batch(self, idxs, start_idx, end_idx):
        bsize = end_idx-start_idx
        batch_pc = np.zeros((bsize, self.num_point, 3))
        batch_pc_target = np.zeros((bsize, self.num_frame, self.num_point, 3))
        batch_disp_target = np.zeros((bsize, self.num_frame, self.num_point, 3))
        batch_mov_seg = np.zeros((bsize, self.num_point), dtype=np.int32)
        batch_part_seg = np.zeros((bsize, self.num_point), dtype=np.int32)
        for i in range(bsize):
            pc, pc_target, disp_target, mov_seg, part_seg = self.__getitem__(idxs[i+start_idx])
            batch_pc[i,:,:] = pc
            batch_pc_target[i,:,:,:] = pc_target
            batch_disp_target[i,:,:,:] = disp_target
            batch_mov_seg[i,:] = mov_seg
            batch_part_seg[i,:] = part_seg

        return batch_pc, batch_pc_target, batch_disp_target, batch_mov_seg, batch_part_seg

    def get_name(self, idxs):
        # get corresponding pointcloud names from index
        return self.shape_cat[self.split][idxs][0]

if __name__ == '__main__':
    print("")