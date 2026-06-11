def precision_skeleton(pred,gt):
    '''
    Precision of the detected skeleton
    Args: pred, a list of predicted *undirected* edges
          gt, ground-truth *undirected* egde list
    '''
    num_fp = 0
    for edge in pred:
        begin,end = edge
        if edge not in gt and (end,begin) not in gt:
            num_fp +=1
    return 1 - num_fp/len(pred)


def recall_skeleton(pred,gt):
    '''
    Recall of the detected skeleton
    Args: pred, a list of *predicted* undirected edges
          gt, ground-truth *undirected* egde list
    '''
    num_fn = 0
    for edge in gt:
        begin,end = edge
        if edge not in pred and (end,begin) not in pred:
            num_fn +=1
    return 1 - num_fn/len(gt)


def precision(pred,gt):
    '''
    Precision of the detected sDAG
    Agrs: pred, a list of predicted *directed* edges
          gt, ground-truth *directed* edges
    '''
    num_fp = 0
    for edge in pred:
        if edge not in gt:
            num_fp += 1
    return 1-num_fp/len(pred)

def recall(pred,gt):
    '''
    Recall of the detected sDAG
    Agrs: pred, a list of predicted *directed* edges
          gt, ground-truth *directed* edges
    '''
    num_fn = 0
    for edge in gt:
        if edge not in pred:
            num_fn += 1
    return 1-num_fn/len(gt)

from collections import defaultdict

def evaluate_bidirectional_elimination(mag_before, mag_after):
    def extract_true_bidirectional_pairs(edges):
        edge_set = set(edges)
        bidirectional_pairs = set()
        for src, tgt in edges:
            src_var, src_t = src.split('_')
            tgt_var, tgt_t = tgt.split('_')

            if src_t < tgt_t and src_var != tgt_var:
                reverse_edge = (tgt_var + '_' + src_t, src_var + '_' + tgt_t)
                if reverse_edge in edge_set:
                    pair = frozenset([(src, tgt), reverse_edge])
                    bidirectional_pairs.add(pair)
        return bidirectional_pairs

    bidir_before = extract_true_bidirectional_pairs(mag_before)
    bidir_after = extract_true_bidirectional_pairs(mag_after)

    eliminated = len(bidir_before - bidir_after)
    total = len(bidir_before)
    rate = 100.0 * eliminated / total if total > 0 else None

    # return {
    #     'original_bidirectional_count': total,
    #     'current_bidirectional_count': len(bidir_after),
    #     'eliminated_bidirectional_count': eliminated,
    #     'elimination_rate': rate
    # }
    return rate
from cdt.metrics import SID, SHD
def backRE(tar_DAG, P_KCI):
    # sid_val=np.max([SID(tar_DAG, P_KCI),SID(P_KCI, tar_DAG)])
    #sid_val = np.max([sid(tar_DAG, P_KCI), sid(P_KCI, tar_DAG)])
    shd_val = SHD(tar_DAG, P_KCI)
    #return [shd_val, 0]
    return shd_val


def compute_f1(precision, recall):

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
