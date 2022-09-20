import torch

def L1_loss(y, y_hat):
    return torch.abs(y_hat - y).sum()

def tv_loss(y_hat):
    # :param y_hat: [batch_size, h_size, w_size]
    return torch.pow((y_hat[:, :-1, :-1] - y_hat[:, 1:, :-1]), 2) + \
            torch.pow((y_hat[:, :-1, :-1] - y_hat[:, :-1, 1:]), 2)

def loss(albedo_hat, light_hat, transport_hat, image_hat, albedo, light, transport, image):
    # image = albedo * (transport * light)
    shading = transport * light
    tranhxlight = transport_hat * light
    tranxlighth = transport * light_hat
    tranhxlighth = transport_hat * light_hat

    sfs_losses = L1_loss(albedo, albedo_hat) + L1_loss(light, light_hat) + \
                L1_loss(transport, transport_hat) + L1_loss(image, image_hat)
    
    tv_losses = L1_loss(tv_loss(albedo, albedo_hat)) + L1_loss(tv_loss(transport, transport_hat))
    
    shading_losses = L1_loss(shading, tranhxlight) + L1_loss(shading, tranxlighth) + \
                        L1_loss(shading, tranhxlighth)
    
    reconstruction_losses = L1_loss(image, albedo * tranxlighth) + L1_loss(image, albedo * tranhxlight) + \
                                L1_loss(image, albedo * tranhxlighth) + L1_loss(image, albedo_hat * shading) + \
                                L1_loss(image, albedo_hat * tranxlighth) + L1_loss(image, albedo_hat * tranhxlight)

    return sfs_losses + tv_losses + shading_losses + reconstruction_losses