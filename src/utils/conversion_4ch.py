import torch

def ConversionTo4Channels(image):
    """
    Convert images in a shape with 4 channels

    Args:
        image (torch.Tensor): Labels tensor with shape (batch_size, num_channels, height, width).

    Returns:
        torch.Tensor: Converted labels with shape (batch_size, 4, height, width).
    """
    
    # Return image directly if already on 4 channels
    if image.size(1) == 4:
        return image

    # Create a tensor of zeros with the right shape
    image_4ch = torch.zeros(image.size(0), 4, image.size(2), image.size(3))

    # 
    for i in range(4):
            image_4ch[:, i, :, :] = (image[:,0,:,:] == i).float()

    return image_4ch