import torchvision

# Load single shot detector
ssd = torchvision.models.detection.ssd300_vgg16(pretrained = True)

# put to evaluation mode to not track gradients - just for inference
ssd.eval()