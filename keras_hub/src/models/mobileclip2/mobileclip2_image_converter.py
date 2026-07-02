from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.mobileclip2.mobileclip2_backbone import (
    MobileCLIP2Backbone,
)


@keras_hub_export("keras_hub.layers.MobileCLIP2ImageConverter")
class MobileCLIP2ImageConverter(ImageConverter):
    backbone_cls = MobileCLIP2Backbone
