import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import zipfile
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import dataset
import cluster

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_rpm', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--train_list', default='datalist/RPM_train.txt', help='Datalist for training')
parser.add_argument('--test_list', default='datalist/RPM_test.txt', help='Datalist for testing')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 1024]')
parser.add_argument('--num_frame', type=int, default=5, help='Frames number need to be generated [default: 9]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--model_path', default='../output/YOUR_MODEL_PATH/ckpts/model.ckpt-90', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--eval_dir', default='../output/YOUR_MODEL_PATH/eval/', help='eval folder path')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frame
GPU_INDEX = FLAGS.gpu
TRAIN_LIST = FLAGS.train_list
TEST_LIST = FLAGS.test_list
DATA_PATH = os.path.join(ROOT_DIR, '../data/')

EVAL_DIR = FLAGS.eval_dir
if not os.path.exists(EVAL_DIR): os.mkdir(EVAL_DIR)
if not os.path.exists(EVAL_DIR+'/pointcloud'): os.mkdir(EVAL_DIR+'/pointcloud')
if not os.path.exists(EVAL_DIR+'/seg'): os.mkdir(EVAL_DIR+'/seg')
LOG_FOUT = open(os.path.join(EVAL_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_PATH = FLAGS.model_path

# Shapenet official train/test split
TEST_DATASET = dataset.MotionDataset(data_path=DATA_PATH, train_list=TRAIN_LIST, test_list=TEST_LIST, num_point=NUM_POINT, num_frame=NUM_FRAME, split='test', batch_size=BATCH_SIZE)

def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)

def evaluate():
	with tf.device('/gpu:'+str(GPU_INDEX)):
		pointclouds_pl, pc_target_pl, disp_target_pl, part_seg_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME)
		gt_mov_seg = tf.cast(tf.greater( part_seg_pl, 0), tf.int32)
		is_training_pl = tf.placeholder(tf.bool, shape=())

		print("--- Get model ---")
		pred_pc, pred_disp, pred_seg, mov_mask, simmat_logits = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl)
		saver = tf.train.Saver()
	
	# Create a session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)

	# Restore variables from disk.
	saver.restore(sess, MODEL_PATH)
	log_string("Model restored.")

	ops = {'pointclouds_pl': pointclouds_pl,
			'pc_target_pl': pc_target_pl,
			'disp_target_pl': disp_target_pl,
			'part_seg_pl': part_seg_pl,
			'is_training_pl': is_training_pl,
			'pred_pc': pred_pc,
			'pred_seg': pred_seg,
			'simmat_logits': simmat_logits}

	eval_one_epoch(sess, ops)

def eval_one_epoch(sess, ops):
	""" ops: dict mapping from string to tf ops """
	is_training = False
	
	log_string(str(datetime.now()))

	test_idxs = np.arange(0, len(TEST_DATASET))
	num_batches = len(TEST_DATASET)

	total_correct_seg = 0
	total_seen_seg = 0
	sum_ap = 0
	batch_idx = 0

	for batch_idx in range(num_batches):
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx+1) * BATCH_SIZE
		batch_pc, batch_pc_target, batch_disp_target, batch_mov_seg, batch_part_seg = TEST_DATASET.get_batch(test_idxs, start_idx, end_idx)

		feed_dict = {ops['pointclouds_pl']: batch_pc,
					 ops['pc_target_pl']: batch_pc_target,
					 ops['disp_target_pl']: batch_disp_target,
					 ops['part_seg_pl']: batch_part_seg,
					 ops['is_training_pl']: is_training}
		pred_pc_val, pred_seg_val, simmat_logits_val = sess.run([ops['pred_pc'], ops['pred_seg'], ops['simmat_logits']], feed_dict=feed_dict)
	
		pred_seg_label = np.argmax(pred_seg_val[0], 1)
		correct_seg = np.sum(pred_seg_label == batch_mov_seg[0])
		total_correct_seg += correct_seg
		total_seen_seg += NUM_POINT
	
		simmat = simmat_logits_val[0]

		out_name = TEST_DATASET.get_name(batch_idx)
		ptspos = batch_pc[0,:,:3]
		mov_seg = pred_seg_label
		gt_part_seg = batch_part_seg[0]

		if np.sum(mov_seg) <= 64:
			part_seg = np.zeros((NUM_POINT))
			log_string("WARING: mov points less than 64")
		else:
			part_seg, proposals = cluster.GroupMergingSimDist(ptspos, simmat, mov_seg)
			ap = cluster.ComputeAP( part_seg, gt_part_seg )
			sum_ap += ap
			log_string('%d: %s'%(batch_idx, out_name))
			log_string('EVAL: AP: %f, movmask_acc: %f\n' % (ap, correct_seg / NUM_POINT))

		for frame in range(NUM_FRAME):
			np.savetxt(EVAL_DIR+'/pointcloud/'+out_name+'_'+str(frame+1)+'.pts', pred_pc_val[0,frame], fmt='%.8f')

		with open(EVAL_DIR+'/seg/'+out_name+'.seg', 'w') as f:
			for i in range(NUM_POINT):
				f.writelines(str(part_seg[i])+'\n')

	log_string('----------------STATISTICS----------------')
	log_string('Mean Mov mask Accuracy: %f'% (total_correct_seg / float(total_seen_seg)))
	log_string('Mean Average Precision: %f'%(sum_ap / num_batches))

if __name__ == "__main__":
	log_string('pid: %s'%(str(os.getpid())))
	evaluate()
	LOG_FOUT.close()
