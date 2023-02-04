import torch

def L1_loss(y, y_hat):
    return torch.abs(y_hat - y).sum()

def tv_loss(y_hat):
    # TODO: The implementation of tv loss has potential to be wrong. The author has given two implementations and I choose the simple one.
    return L1_loss(y_hat[:,:,1:,:], y_hat[:,:,:-1,:]) + L1_loss(y_hat[:,:,:,1:], y_hat[:,:,:,:-1])

def loss(albedo_hat, light_hat, transport_hat, image_hat, albedo, light, transport, image):
    """
        albedo_hat: torch.Size([2, 3, 1024, 1024])
        light_hat: torch.Size([2, 9, 3])
        transport_hat: torch.Size([2, 9, 1024, 1024])
        image_hat: torch.Size([2, 3, 1024, 1024])
        albedo: torch.Size([2, 3, 1024, 1024])
        light: torch.Size([2, 9, 3])
        transport: torch.Size([2, 9, 1024, 1024])
        image: torch.Size([2, 3, 1024, 1024])
    """
    tmp_light = light.permute(0, 2, 1) # [2, 3, 9]
    tmp_light_hat = light_hat.permute(0, 2, 1)

    tranxlight      = torch.clamp(torch.einsum('ijk,iklm->ijlm', tmp_light, transport), 0, 10.0) # [2, 3, 1024, 1024]
    tranhxlight     = torch.clamp(torch.einsum('ijk,iklm->ijlm', tmp_light, transport_hat), 0, 10.0)
    tranxlighth     = torch.clamp(torch.einsum('ijk,iklm->ijlm', tmp_light_hat, transport), 0, 10.0)
    tranhxlighth    = torch.clamp(torch.einsum('ijk,iklm->ijlm', tmp_light_hat, transport_hat), 0, 10.0)

    sfs_losses = L1_loss(albedo, albedo_hat) + L1_loss(light, light_hat) + \
                    L1_loss(transport, transport_hat) + L1_loss(image, image_hat)
    
    tv_losses = tv_loss(albedo_hat) + tv_loss(transport_hat)
    
    shading_losses = L1_loss(tranxlight, tranhxlight) + L1_loss(tranxlight, tranxlighth) + \
                        L1_loss(tranxlight, tranhxlighth)
    
    reconstruction_losses = L1_loss(image, albedo * tranxlighth) + L1_loss(image, albedo * tranhxlight) + \
                                L1_loss(image, albedo * tranhxlighth) + L1_loss(image, albedo_hat * tranxlight) + \
                                L1_loss(image, albedo_hat * tranxlighth) + L1_loss(image, albedo_hat * tranhxlight)

    return sfs_losses + tv_losses + shading_losses + reconstruction_losses