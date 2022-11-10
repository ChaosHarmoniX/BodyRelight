import torch

def calc_loss(mask, image, albedo_hat, light_hat, transport_hat,
                albedo_gt, light_gt, transport_gt, loss):
    mask = mask.reshape((-1, 1, 512, 512))
    
    for i in range(3):
        albedo_hat[:, 0, :, :] = albedo_hat[:, i, :, :] * mask[:, 0, :, :]
    
    for i in range(9):
        transport_hat[:, i, :, :] = transport_hat[:, i, :, :] * mask[:, 0, :, :]

    image = image.permute(0, 1, 3, 2) # 为了保证是按H reshape
    image_gt = image.reshape((image.shape[0], image.shape[1], -1))

    light_hat = light_hat.reshape((-1, 3, 9))        
    albedo_hat = albedo_hat.reshape((albedo_hat.shape[0], albedo_hat.shape[1], -1))
    transport_hat = transport_hat.reshape((transport_hat.shape[0], transport_hat.shape[1], -1))
    image_hat = albedo_hat * torch.bmm(light_hat, transport_hat) # 因为light_hat和transport_hat的维度是颠倒的，所以矩阵乘法也颠倒一下
    
    return loss(albedo_hat, light_hat, transport_hat, image_hat, albedo_gt, light_gt, transport_gt, image_gt)