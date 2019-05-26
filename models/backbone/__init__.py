from models.backbone import resnet, mobilenet


def build_backbone(backbone, output_stride, batchnorm, mc_dropout):
    if backbone == 'resnet':
        return resnet.ResNet50(output_stride, batchnorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, batchnorm, mc_dropout=mc_dropout)
    else:
        raise NotImplementedError
