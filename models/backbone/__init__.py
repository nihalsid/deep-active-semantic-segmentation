from models.backbone import resnet, mobilenet


def build_backbone(backbone, output_stride, batchnorm, mc_dropout, input_channels, pretrained):
    if backbone == 'resnet':
        return resnet.ResNet50(output_stride, batchnorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride=output_stride, batchnorm=batchnorm, mc_dropout=mc_dropout, input_channels=input_channels, pretrained=pretrained)
    else:
        raise NotImplementedError
