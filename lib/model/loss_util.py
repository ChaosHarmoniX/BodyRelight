import torch

def L1_loss(y, y_hat):
    return torch.abs(y_hat - y).sum()

def tv_loss(y_hat):
    # :param y_hat: [batch_size, c_3, h_size, w_size]
    return torch.pow((y_hat[:, :, :-1, :-1] - y_hat[:, :, 1:, :-1]), 2) + \
            torch.pow((y_hat[:, :, :-1, :-1] - y_hat[:, :, :-1, 1:]), 2)

def loss(albedo_hat, light_hat, transport_hat, image_hat, albedo, light, transport, image):
    # image = albedo * (transport * light)
    batch_num = albedo_hat.shape[0]

    for i in range(batch_num):
        if i == 0:
            shading = transport[i] @ light[i]
            tranhxlight = transport_hat[i] @ light[i]
            tranxlighth = transport[i] @ light_hat[i]
            tranhxlighth = transport_hat[i] @ light_hat[i]
        else:
            shading = torch.concat((shading, transport[i] @ light[i]), 0)
            tranhxlight = torch.concat((tranhxlight, transport_hat[i] @ light[i]), 0)
            tranxlighth = torch.concat((tranxlighth, transport[i] @ light_hat[i]), 0)
            tranhxlighth = torch.concat((tranhxlighth, transport_hat[i] @ light_hat[i]), 0)


    sfs_losses = L1_loss(albedo, albedo_hat) + L1_loss(light, light_hat) + \
                L1_loss(transport, transport_hat) + L1_loss(image, image_hat)
    
    tv_losses = L1_loss(tv_loss(albedo, albedo_hat)) + L1_loss(tv_loss(transport, transport_hat))
    
    shading_losses = L1_loss(shading, tranhxlight) + L1_loss(shading, tranxlighth) + \
                        L1_loss(shading, tranhxlighth)
    
    reconstruction_losses = L1_loss(image, albedo * tranxlighth) + L1_loss(image, albedo * tranhxlight) + \
                                L1_loss(image, albedo * tranhxlighth) + L1_loss(image, albedo_hat * shading) + \
                                L1_loss(image, albedo_hat * tranxlighth) + L1_loss(image, albedo_hat * tranhxlight)

    return (sfs_losses + tv_losses + shading_losses + reconstruction_losses) / batch_num