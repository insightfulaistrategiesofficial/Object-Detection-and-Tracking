import torchvision

# Load faster rcnn
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

# put to evaluation mode to not track gradients - just for inference
faster_rcnn.eval()