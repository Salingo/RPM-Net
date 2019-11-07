import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

def GroupMergingSimDist(pcpos, simmat_raw, mov_seg):
    # simmat: (N,N) of 0/1 di
    # mov_seg: (N) of 0/1 digits
    num_point = simmat_raw.shape[0]
    refpos = np.where(mov_seg==0)[0]
    movpos = np.where(mov_seg==1)[0]
    refpts = pcpos[refpos]
    movpts = pcpos[movpos]

    grp_threshhold = 80
    iou_threshold = .5
    min_points = 64
    group_seg = np.zeros(num_point,dtype=np.int32)
    simmat = (simmat_raw>grp_threshhold).astype(np.int32)
    movmat = simmat[np.ix_(movpos,movpos)]

    clustering = DBSCAN(eps=10, min_samples=min_points).fit(movmat)
    raw_group_seg = clustering.labels_
    goodindex = np.where(raw_group_seg!=-1)[0]
    badindex = np.where(raw_group_seg==-1)[0]
    if len(badindex)>0:
        kdtree = KDTree(movpts[goodindex], leaf_size=10)
        dist, nnindices = kdtree.query(movpts[badindex], k=1)
        for i, nnindex in enumerate(nnindices):
            raw_group_seg[badindex[i]] = raw_group_seg[ goodindex[nnindex[0]] ]
        assert((raw_group_seg==-1).sum()==0)
    group_seg[movpos] = raw_group_seg+1
    proposals = []
    for i in range(group_seg.max()+1):
        proposals.append((group_seg==i).astype(np.int32))
    proposals = np.array(proposals)

    return group_seg, proposals

def ComputeAP(group_seg, gt_group_seg):
    groups, gt_groups = [], []
    for i in np.unique(group_seg):
        groups.append( group_seg==i )
    for i in np.unique(gt_group_seg):
        gt_groups.append( gt_group_seg==i )
    groups = np.array(groups)
    gt_groups = np.array(gt_groups)
    iou_mat = np.zeros((groups.shape[0], gt_groups.shape[0]))
    for i, p in enumerate(groups):
        for j, gt_p in enumerate(gt_groups):
            if np.sum( p | gt_p)==0:
                print(np.unique(group_seg))
                print(np.unique(gt_group_seg))
            iou = float(np.sum(p & gt_p)) / np.sum( p | gt_p)#uniou
            iou_mat[i,j] = iou
    # comput AP@[.5:.95] (Instance segmentation metric in COCO)
    ious_thresh_candidates = list(reversed(np.arange(.5, 1., .05)))
    AP = 0.
    Precisions, Recalls = [1.], [0.]
    for iou_thresh in ious_thresh_candidates:
        ap_mat = (iou_mat>iou_thresh).astype(int)
        TruePositive, FalsePositive, FalseNegative = 0., 0., 0.
        for i in range(groups.shape[0]):
            FalsePositive += (ap_mat[i,:].sum()==0).astype(float)
        for i in range(gt_groups.shape[0]):
            TruePositive += (ap_mat[:,i].sum()>0).astype(float)
            FalseNegative += (ap_mat[:,i].sum()==0).astype(float)
        if TruePositive+FalsePositive==0:
            Precisions.append( 1. )
        else:
            Precisions.append( TruePositive/(TruePositive+FalsePositive) )
        Recalls.append( TruePositive/(TruePositive+FalseNegative) )
        AP += (Recalls[-1] - Recalls[-2]) * Precisions[-1]

    return AP
