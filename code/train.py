import argparse
import math
from datetime import datetime

import numpy as np
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
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_rpm', help='Model name [default: pointnet2_cls_ssg]') 
parser.add_argument('--train_list', default='datalist/RPM_train.txt', help='Datalist for training')
parser.add_argument('--test_list', default='datalist/RPM_test.txt', help='Datalist for testing')
parser.add_argument('--save_folder', default='../output/', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 1024]')
parser.add_argument('--num_frame', type=int, default=5, help='Frames number need to be generated [default: 9]')
parser.add_argument('--max_epoch', type=int, default=121, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frame
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
TRAIN_LIST = FLAGS.train_list
TEST_LIST = FLAGS.test_list
DATA_PATH = os.path.join(ROOT_DIR, '../data/')

SAVE_DIR = os.path.join(FLAGS.save_folder, '%s_%s_%s' % (FLAGS.model, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), 'all'))
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

LOG_FOUT = open(os.path.join(SAVE_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write("Comments: \n\n")
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

MODEL = importlib.import_module(FLAGS.model) # import network module

code_folder = os.path.abspath(os.path.dirname(__file__))
zip_name = os.path.join(SAVE_DIR) + "/code.zip"
filelist = []
for root, dirs, files in os.walk(code_folder):
	for name in files:
		filelist.append(os.path.join(root, name))
zip_code = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
for tar in filelist:
	arcname = tar[len(code_folder):]
	zip_code.write(tar, arcname)
zip_code.close()

folder_ckpt = os.path.join(SAVE_DIR, 'ckpts')
if not os.path.exists(folder_ckpt): os.makedirs(folder_ckpt)

folder_summary = os.path.join(SAVE_DIR, 'summary')
if not os.path.exists(folder_summary): os.makedirs(folder_summary)

TRAIN_DATASET = dataset.MotionDataset(data_path=DATA_PATH, train_list=TRAIN_LIST, test_list=TEST_LIST, num_point=NUM_POINT, num_frame=NUM_FRAME, split='train', batch_size=BATCH_SIZE)

def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)

def get_learning_rate(batch):
	learning_rate = tf.train.exponential_decay(
						BASE_LEARNING_RATE,  # Base learning rate.
						batch * BATCH_SIZE,  # Current index into the dataset.
						DECAY_STEP,          # Decay step.
						DECAY_RATE,          # Decay rate.
						staircase=True)
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate        

def get_bn_decay(batch):
	bn_momentum = tf.train.exponential_decay(
					  BN_INIT_DECAY,
					  batch*BATCH_SIZE,
					  BN_DECAY_DECAY_STEP,
					  BN_DECAY_DECAY_RATE,
					  staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

def train():
	with tf.Graph().as_default():
		with tf.device('/gpu:'+str(GPU_INDEX)):
			pointclouds_pl, pc_target_pl, disp_target_pl, part_seg_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME)
			gt_mov_seg = tf.cast(tf.greater(part_seg_pl, 0), tf.int32)
			is_training_pl = tf.placeholder(tf.bool, shape=())
			
			batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
			bn_decay = get_bn_decay(batch)
			tf.summary.scalar('bn_decay', bn_decay)

			print("--- Get model and loss ---")
			pred_pc, pred_disp, pred_seg, mov_mask, simmat_logits = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl, bn_decay=bn_decay)
			loss_ref = MODEL.get_ref_loss(pred_pc, pc_target_pl, gt_mov_seg)
			loss_mov = MODEL.get_mov_loss(pred_pc, pc_target_pl, gt_mov_seg)
			loss_mov_seg = MODEL.get_movseg_loss(pred_seg, gt_mov_seg)
			loss_disp = MODEL.get_disp_loss(pred_disp, disp_target_pl, gt_mov_seg)
			loss_partseg, part_err = MODEL.get_partseg_loss(simmat_logits, mov_mask, part_seg_pl)
			loss_generator = loss_mov + loss_ref + loss_mov_seg + loss_disp
			total_loss = loss_generator + loss_partseg
			tf.summary.scalar('losses/generator_loss', loss_generator)
			tf.summary.scalar('losses/partseg_loss', loss_partseg)

			print("--- Get training operator ---")
			learning_rate = get_learning_rate(batch)
			tf.summary.scalar('learning_rate', learning_rate)

			optimizer = tf.train.AdamOptimizer(learning_rate)

			generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
			generator_op = optimizer.minimize(loss_generator, var_list=generator_vars, global_step=batch)
			partseg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "partseg")
			partseg_op = optimizer.minimize(loss_partseg, var_list=partseg_vars, global_step=batch)
			rpm_op = optimizer.minimize(total_loss, global_step=batch)
			saver = tf.train.Saver(max_to_keep=30)
		
		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(os.path.join(folder_summary, 'train'), sess.graph)

		init = tf.global_variables_initializer()
		sess.run(init)

		ops = {'pointclouds_pl': pointclouds_pl,
			   'pc_target_pl': pc_target_pl,
			   'disp_target_pl': disp_target_pl,
			   'part_seg_pl': part_seg_pl,
			   'is_training_pl': is_training_pl,
			   'pred_pc': pred_pc,
			   'pred_seg': pred_seg,
			   'loss_mov': loss_mov,
			   'loss_ref': loss_ref,
			   'loss_disp': loss_disp,
			   'loss_mov_seg': loss_mov_seg,
			   'loss_partseg': loss_partseg,
			   'part_err': part_err,
			   'generator_op': generator_op,
			   'partseg_op': partseg_op,
			   'rpm_op': rpm_op,
			   'merged': merged,
			   'step': batch}

		for epoch in range(MAX_EPOCH):
			log_string('**** EPOCH %03d ****' % (epoch))
			sys.stdout.flush()
			stage1_epochs=30
			stage2_epochs=60
			stage3_epochs=30
			if epoch <= stage1_epochs:
				train_one_epoch(sess, ops, train_writer, 'generator_op')
			elif epoch <= stage1_epochs + stage2_epochs:
				train_one_epoch(sess, ops, train_writer, 'partseg_op')
			else:
				train_one_epoch(sess, ops, train_writer, 'rpm_op')

			if epoch % 30 == 0 and epoch != 0:
				save_path = saver.save(sess, os.path.join(folder_ckpt, "model.ckpt"), global_step=epoch)
				log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer, training_type):
	""" ops: dict mapping from string to tf ops """
	is_training = True
	log_string(str(datetime.now()))

	# Shuffle train samples
	train_idxs = np.arange(0, len(TRAIN_DATASET))
	np.random.shuffle(train_idxs)
	num_batches = len(TRAIN_DATASET)//BATCH_SIZE

	loss_sum_mov = 0
	loss_sum_ref = 0
	loss_sum_disp = 0	
	loss_sum_movseg = 0
	loss_sum_partseg = 0
	total_seen_seg = 0
	total_correct_seg = 0
	total_part_err = 0

	batch_idx = 0
	for batch_idx in range(num_batches):
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx+1) * BATCH_SIZE
		batch_pc, batch_pc_target, batch_disp_target, batch_mov_seg, batch_part_seg = TRAIN_DATASET.get_batch(train_idxs, start_idx, end_idx)

		feed_dict = {ops['pointclouds_pl']: batch_pc,
					 ops['pc_target_pl']: batch_pc_target,
					 ops['disp_target_pl']: batch_disp_target,
					 ops['part_seg_pl']: batch_part_seg,
					 ops['is_training_pl']: is_training}
		summary, step, _, loss_mov_val, loss_ref_val, loss_movseg_val, pred_seg_val, loss_disp_val, loss_partseg_val, part_err_val = sess.run(
				[ops['merged'], ops['step'], ops[training_type], ops['loss_mov'], ops['loss_ref'], ops['loss_mov_seg'], ops['pred_seg'], ops['loss_disp'], ops['loss_partseg'], ops['part_err']], feed_dict=feed_dict)
		train_writer.add_summary(summary, step)

		for batch in range(BATCH_SIZE):
			pred_seg_label = np.argmax(pred_seg_val[batch], 1)
			correct_seg = np.sum(pred_seg_label == batch_mov_seg[batch])
			total_correct_seg += correct_seg
			total_seen_seg += NUM_POINT

			loss_sum_mov += loss_mov_val
			loss_sum_ref += loss_ref_val
			loss_sum_disp += loss_disp_val
			loss_sum_movseg += loss_movseg_val
			loss_sum_partseg += loss_partseg_val
			total_part_err += part_err_val
	
	log_string('EPOCH STAT:')
	log_string('mean mov loss: %f' % (loss_sum_mov / num_batches))
	log_string('mean ref loss: %f' % (loss_sum_ref / num_batches))
	log_string('mean disp loss: %f' % (loss_sum_disp / num_batches))	
	log_string('mean mov seg loss: %f' % (loss_sum_movseg / num_batches))
	log_string('mean part seg loss: %f' % (loss_sum_partseg / num_batches))
	log_string('mov seg acc: %f'% (total_correct_seg / float(total_seen_seg)))
	log_string('part seg err: %f' % (total_part_err / num_batches))

if __name__ == "__main__":
	log_string('pid: %s'%(str(os.getpid())))
	train()
	LOG_FOUT.close()
