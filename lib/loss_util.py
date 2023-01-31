import torch

def L1_loss(y, y_hat):
    return torch.abs(y_hat - y).sum()

def tv_loss(y_hat):
    image_size = 1024
    # :param y_hat: [batch_size, c_3, N]
    y_hat = y_hat.view((y_hat.shape[0], y_hat.shape[1], image_size, image_size))
    # y_hat: [batch_size, c_3, W, H]
    tmp1 = torch.cat((y_hat[:, :, 1:, :], y_hat[:, :, 0, :].unsqueeze(2)), 2)
    tmp2 = torch.cat((y_hat[:, :, :, 1:], y_hat[:, :, :, 0].unsqueeze(3)), 3)
    return L1_loss(tmp1, y_hat) + L1_loss(tmp2, y_hat)
    # return (torch.pow(tmp1, 2) + torch.pow(tmp2, 2)).sum()

def loss(albedo_hat, light_hat, transport_hat, image_hat, albedo, light, transport, image):
    # image = albedo * (transport @ light)
    batch_num = albedo_hat.shape[0]

    shading         = torch.bmm(light, transport)
    tranhxlight     = torch.bmm(light, transport_hat)
    tranxlighth     = torch.bmm(light_hat, transport)
    tranhxlighth    = torch.bmm(light_hat, transport_hat)

    sfs_losses = L1_loss(albedo, albedo_hat) + L1_loss(light, light_hat) + \
                    L1_loss(transport, transport_hat) + L1_loss(image, image_hat)
    
    tv_losses = tv_loss(albedo_hat) + tv_loss(transport_hat)
    
    shading_losses = L1_loss(shading, tranhxlight) + L1_loss(shading, tranxlighth) + \
                        L1_loss(shading, tranhxlighth)
    
    reconstruction_losses = L1_loss(image, albedo * tranxlighth) + L1_loss(image, albedo * tranhxlight) + \
                                L1_loss(image, albedo * tranhxlighth) + L1_loss(image, albedo_hat * shading) + \
                                L1_loss(image, albedo_hat * tranxlighth) + L1_loss(image, albedo_hat * tranhxlight)

    return (sfs_losses + tv_losses + shading_losses + reconstruction_losses) / batch_num