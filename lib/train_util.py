import torch

def calc_loss(mask, image_gt, albedo_hat, light_hat, transport_hat,
                albedo_gt, light_gt, transport_gt, loss):
    albedo_hat = albedo_hat * mask[:, None, :, :]
    transport_hat = transport_hat * mask[:, None, :, :]

    tmp_light = light_hat.permute(0, 2, 1) # [2, 3, 9]
    shading = torch.einsum('ijk,iklm->ijlm', tmp_light, transport_hat)
    image_hat = albedo_hat * shading

    
    return loss(albedo_hat, light_hat, transport_hat, image_hat, albedo_gt, light_gt, transport_gt, image_gt)

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr
