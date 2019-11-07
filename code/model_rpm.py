import os
import sys
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, num_point, num_frame):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pc_target_pl = tf.placeholder(tf.float32, shape=(batch_size, num_frame, num_point, 3))
    disp_target_pl = tf.placeholder(tf.float32, shape=(batch_size, num_frame, num_point, 3))
    seg_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, pc_target_pl, disp_target_pl, seg_pl

def get_model(point_cloud, num_frame, is_training, bn_decay=None):
    """ Part segmentation PointNet, input is BxNx3, output BxNx2 """
    """ Classification PointNet, input is BxNx3, output Bx3 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    ''' shape=(batch_size, num_point, 3) '''
    with tf.variable_scope("generator"):
        l0_xyz = point_cloud
        l0_points = None
        # Set Abstraction layers
        ''' shape=(batch_size, 1024, 128) '''
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer1')
        ''' shape=(batch_size, 384, 256) '''
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.2, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer2')
        ''' shape=(batch_size, 128, 512) '''
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4, nsample=64, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer3')
        ''' shape=(batch_size, 1, 1024) '''
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[512,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='sa_layer4')

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=1024, forget_bias=1.0, name='lstm')
        init_input = tf.reshape(l4_points, [batch_size, -1])

        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        pc = []
        disp = []
        with tf.variable_scope("RNN"):
            for time_step in range(num_frame):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                if time_step == 0:
                    (cell_output, state) = lstm_cell(init_input, init_state)
                else:
                    (cell_output, state) = lstm_cell(init_input, state)

                cell_output = tf.reshape(cell_output, [batch_size,1,1024])
                # Feature Propagation layers
                ''' shape=(batch_size, 128, 256) '''
                l3_points_ = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, cell_output, [512,512], is_training, bn_decay, scope='fp_layer1')            
                ''' shape=(batch_size, 128, 256) '''
                l2_points_ = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_, [512,512], is_training, bn_decay, scope='fp_layer2')
                ''' shape=(batch_size, 512, 128) '''
                l1_points_ = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points_, [256,128], is_training, bn_decay, scope='fp_layer3')
                ''' shape=(batch_size, 1024, 128) '''
                l0_points_ = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_, [128,128,128], is_training, bn_decay, scope='fp_layer4')

                # FC layers for feature extraction
                ''' shape = (batch_size, num_point, 128) '''
                fea_fc = tf_util.conv1d(l0_points_, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fea_fc1', bn_decay=bn_decay)
                ''' shape = (batch_size, num_point, 64) '''
                fea_fc = tf_util.conv1d(fea_fc, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fea_fc2', bn_decay=bn_decay)
                fea_fc = tf_util.dropout(fea_fc, keep_prob=0.5, is_training=is_training, scope='fea_dp1')
                ''' shape = (batch_size, num_point, 3) '''
                disp_out = tf_util.conv1d(fea_fc, 3, 1, padding='VALID', activation_fn=None, scope='fea_fc3')
                disp.append(disp_out)
                pc.append(point_cloud+disp_out)
                point_cloud += disp_out

        '''shape=(num_frame, batch_size, num_point, 3)'''
        pc = tf.stack(pc)
        disp = tf.stack(disp)
        '''shape=(batch_size, num_frame, num_point, 3)'''
        pc = tf.transpose(pc, [1,0,2,3])
        disp = tf.transpose(disp, [1,0,2,3])

        # FC layers for segmentation
        seg_fc = tf.reshape(tf.transpose(disp, [0,2,1,3]), [batch_size,num_point,num_frame*3])
        seg_fc = tf_util.conv1d(seg_fc, 8, 1, padding='VALID', bn=True, is_training=is_training, scope='seg_fc1', bn_decay=bn_decay)
        seg_dp = tf_util.dropout(seg_fc, keep_prob=0.5, is_training=is_training, scope='seg_dp1')
        seg_fc = tf_util.conv1d(seg_dp, 2, 1, padding='VALID', bn=True, is_training=is_training, scope='seg_fc2', bn_decay=bn_decay)
        mov_seg = tf.reshape(seg_fc, [batch_size,num_point,2])

    with tf.variable_scope("partseg"):
        seg_l0_points = tf.reshape(disp, (batch_size,num_point,3*num_frame))

        mov_mask = tf.cast(tf.greater( tf.argmax(mov_seg, 2), 0), tf.int32)
        mask_tiled = tf.cast(tf.expand_dims(mov_mask, -1), tf.float32)
        mask_tiled = tf.tile(mask_tiled, [1,1,3*num_frame])
        seg_l0_points = seg_l0_points * mask_tiled

        ''' shape=(batch_size, 1024, 128) '''
        seg_l1_xyz, seg_l1_points, seg_l1_indices = pointnet_sa_module(l0_xyz, seg_l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer5')
        ''' shape=(batch_size, 384, 256) '''
        seg_l2_xyz, seg_l2_points, seg_l2_indices = pointnet_sa_module(seg_l1_xyz, seg_l1_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer6')
        ''' shape=(batch_size, 128, 512) '''
        seg_l3_xyz, seg_l3_points, seg_l3_indices = pointnet_sa_module(seg_l2_xyz, seg_l2_points, npoint=128, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer7')
        ''' shape=(batch_size, 1, 1024) '''
        seg_l4_xyz, seg_l4_points, seg_l4_indices = pointnet_sa_module(seg_l3_xyz, seg_l3_points, npoint=32, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='sa_layer8')
        # Feature Propagation layers
        ''' shape=(batch_size, 128, 256) '''
        seg_l3_points_ = pointnet_fp_module(seg_l3_xyz, seg_l4_xyz, seg_l3_points, seg_l4_points, [256,256], is_training, bn_decay, scope='fp_layer5')            
        ''' shape=(batch_size, 128, 256) '''
        seg_l2_points_ = pointnet_fp_module(seg_l2_xyz, seg_l3_xyz, seg_l2_points, seg_l3_points_, [256,256], is_training, bn_decay, scope='fp_layer6')
        ''' shape=(batch_size, 512, 128) '''
        seg_l1_points_ = pointnet_fp_module(seg_l1_xyz, seg_l2_xyz, seg_l1_points, seg_l2_points_, [256,128], is_training, bn_decay, scope='fp_layer7')
        ''' shape=(batch_size, 1024, 128) '''
        seg_l0_points_ = pointnet_fp_module(l0_xyz, seg_l1_xyz, seg_l0_points, seg_l1_points_, [128,128,128], is_training, bn_decay, scope='fp_layer8')

        sim_features = tf_util.conv1d(seg_l0_points_, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='seg_fc3', bn_decay=bn_decay)
        sim_features = tf_util.dropout(sim_features, keep_prob=0.5, is_training=is_training, scope='seg_dp2')
        sim_features = tf_util.conv1d(sim_features, 128, 1, padding='VALID', activation_fn=None, scope='seg_fc4')

        r = tf.reduce_sum(sim_features * sim_features, 2)
        r = tf.reshape(r, [batch_size, -1, 1])
        print(r.get_shape(),sim_features.get_shape())
        # (x-y)^2 = x^2 - 2*x*y + y^2
        D = r - 2 * tf.matmul(sim_features, tf.transpose(sim_features, perm=[0, 2, 1])) + tf.transpose(r, perm=[0, 2, 1])
        simmat_logits = tf.maximum(10 * D, 0.)

    return pc, disp, mov_seg, mov_mask, simmat_logits

def get_partseg_loss(simmat_logits, mov_mask, gt_part_seg):
    MAX_PART_NUM = 20
    num_point = simmat_logits.get_shape()[1].value
    MAT_THRESHOLD = tf.constant(80., name="MAT_THRESHOLD")
    gt_part_seg_onehot = tf.one_hot(gt_part_seg, MAX_PART_NUM, on_value=1.0, off_value=0.0, axis=-1)
    B = gt_part_seg_onehot.get_shape()[0]
    N = gt_part_seg_onehot.get_shape()[1]

    onediag = tf.ones([B,N], tf.float32)

    part_mat_label = tf.matmul(gt_part_seg_onehot,tf.transpose(gt_part_seg_onehot, perm=[0, 2, 1])) #BxNxN: (i,j) if i and j in the same part
    part_mat_label = tf.matrix_set_diag(part_mat_label,onediag)
    samepart_mat_label = part_mat_label
    diffpart_mat_label = tf.subtract(1., part_mat_label)

    # only compute mov points' losses
    mask_tiled = tf.cast(tf.expand_dims(mov_mask, -1), tf.float32)
    mask_tiled = tf.tile(mask_tiled, [1,1,num_point])
    samepart_mat_label = samepart_mat_label * mask_tiled * tf.transpose(mask_tiled, [0,2,1])
    diffpart_mat_label = diffpart_mat_label * mask_tiled * tf.transpose(mask_tiled, [0,2,1])

    same_loss = tf.multiply(samepart_mat_label, simmat_logits) # minimize distances if in the same part
    diff_loss = tf.multiply(diffpart_mat_label, tf.maximum(tf.subtract(MAT_THRESHOLD, simmat_logits), 0.))
    loss_partseg = tf.reduce_mean(same_loss + diff_loss)

    simmat_label = part_mat_label
    simmat_label = tf.greater(simmat_label, tf.constant(0.5))
    simmat = tf.less(simmat_logits, MAT_THRESHOLD)
    parterr = tf.reduce_mean( tf.abs(tf.cast(simmat, tf.float32) - tf.cast(simmat_label, tf.float32)) )

    tf.summary.scalar('part_error', parterr)
    tf.summary.scalar('loss_partseg', loss_partseg)
    tf.add_to_collection('losses', loss_partseg)

    return loss_partseg, parterr


def get_mov_loss(pred_pc, target, seg):
    """ pred_pc: BxFxNx3,
        target: BxFxNx3,
        seg: BxN """
    batch_size = pred_pc.get_shape()[0].value
    num_frame = pred_pc.get_shape()[1].value
    num_point = pred_pc.get_shape()[2].value
    seg_mask = tf.cast(tf.not_equal(seg,0), dtype=tf.bool)
    seg = tf.cast(seg, dtype=tf.float32)
    num_mov = tf.reduce_sum(seg, axis=-1)

    frameweight = tf.constant([1.0,1.1,1.2,1.3,1.4], dtype=tf.float32)
    shapeLoss = 0
    densityLoss = 0
    for k in range(batch_size):
        pred_pc_n = tf.boolean_mask(tf.reshape(pred_pc[k,:,:,:], [num_frame, num_point, 3]), seg_mask[k], axis=1)
        target_pc_n = tf.boolean_mask(tf.reshape(target[k,:,:,:], [num_frame, num_point, 3]), seg_mask[k], axis=1)
        # calculate shape loss
        square_dist = pairwise_l2_norm2_batch(target_pc_n, pred_pc_n)
        dist = tf.sqrt(square_dist)
        minRow = tf.reduce_min(dist, axis=2, keepdims=False)
        minCol = tf.reduce_min(dist, axis=1, keepdims=False)
        a = tf.reduce_mean(tf.reduce_mean(minRow, axis=-1)*frameweight)
        b = tf.reduce_mean(tf.reduce_mean(minCol, axis=-1)*frameweight)
        shapeLoss += a + b

        # calculate density loss
        square_dist2 = pairwise_l2_norm2_batch(target_pc_n, target_pc_n)
        dist2 = tf.sqrt(square_dist2)
        knndis = tf.nn.top_k(tf.negative(dist), k=8)
        knndis2 = tf.nn.top_k(tf.negative(dist2), k=8)
        a = tf.reduce_mean(tf.reduce_mean(tf.abs(knndis.values - knndis2.values), axis=-1), axis=-1)
        densityLoss += tf.reduce_mean(a*frameweight)
    shapeLoss = shapeLoss / batch_size * 5
    densityLoss = densityLoss / batch_size * 5

    loss_mov = shapeLoss + densityLoss
    tf.summary.scalar('loss_mov', loss_mov)
    tf.add_to_collection('losses', loss_mov)
    return loss_mov

def get_ref_loss(pred_pc, target, seg):
    """ pred_pc: BxFxNx3,
        target: BxFxNx3, 
        seg: BxN """
    batch_size = pred_pc.get_shape()[0].value
    num_frame = pred_pc.get_shape()[1].value
    num_point = pred_pc.get_shape()[2].value
    seg_reverse = tf.cast(tf.equal(seg,0), dtype=tf.float32)
    num_ref = tf.reduce_sum(seg_reverse, axis=-1)

    l2 = tf.norm(target-pred_pc, axis=-1)
    perframe_ref_loss = []
    for i in range(batch_size):
        perframe_ref_loss.append(tf.reduce_sum(tf.multiply(l2[i], seg_reverse[i]), axis=-1) / num_ref[i])

    perframe_ref_loss = tf.stack(perframe_ref_loss)
    loss_ref = tf.reduce_mean(perframe_ref_loss) * 10

    tf.summary.scalar('loss_ref', loss_ref)
    tf.add_to_collection('losses', loss_ref)
    return loss_ref

def get_disp_loss(pred_disp, disp, seg):
    '''shape=(batch_size, num_frame, num_point, 3)'''
    batch_size = pred_disp.get_shape()[0].value
    num_frame = pred_disp.get_shape()[1].value
    num_point = pred_disp.get_shape()[2].value
    seg_mask = tf.cast(tf.not_equal(seg,0), dtype=tf.bool)

    var_sum = 0
    ang_sum = 0
    for k in range(batch_size):
        pred_disp_n = tf.boolean_mask(tf.reshape(pred_disp[k,:,:,:], [num_frame, num_point, 3]), seg_mask[k], axis=1)
        disp_n = tf.boolean_mask(tf.reshape(disp[k,:,:,:], [num_frame, num_point, 3]), seg_mask[k], axis=1)
        
        ''' same distance '''        
        movdis = tf.linalg.norm(pred_disp_n, axis=-1, keepdims=False)
        movdis = tf.transpose(movdis, [1,0])
        mean, var = tf.nn.moments(movdis, axes=[-1])
        var_sum += tf.reduce_mean(var)

        ''' same angle '''
        l2 = tf.norm(pred_disp_n-disp_n, axis=-1)
        ang_sum += tf.reduce_mean(l2)

    loss_disp = (var_sum + ang_sum) / batch_size
    tf.summary.scalar('loss_displ2', loss_disp)
    tf.add_to_collection('losses', loss_disp)
    return loss_disp

def get_movseg_loss(pred_seg, seg):
    """ pred_seg: BxN,
        seg: BxN, """
    loss_seg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_seg, labels=seg)
    loss_seg = tf.reduce_mean(loss_seg) * 2
    tf.summary.scalar('loss_seg', loss_seg)
    tf.add_to_collection('losses', loss_seg)
    return loss_seg

def pairwise_l2_norm2_batch(x, y, scope=None):
    with tf.name_scope(scope, 'pairwise_l2_norm2_batch', [x, y]):
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 2)

        return square_dist

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((8,2048,3))
        pred = tf.zeros((8,9,2048,3))
        mo = tf.zeros((8,6))
