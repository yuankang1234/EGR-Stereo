import torch.nn.functional as F

from models.loss_functions import disp2distribute, CEloss


def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5 * 0.5, 0.5 * 0.7, 0.5 * 1.0, 1 * 0.5, 1 * 0.7, 1 * 1.0, 2 * 0.5, 2 * 0.7, 2 * 1.0]  # 0.5,1,2
    all_losses = []
    gt_distribute = disp2distribute(disp_gt, 2, b=2)
    loss2 = CEloss(disp_gt, 192, gt_distribute, disp_gt)
    all_losses.append(0.1 * loss2)
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


