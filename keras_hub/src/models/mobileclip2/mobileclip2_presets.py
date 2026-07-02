"""MobileCLIP2 model preset configurations."""

backbone_presets = {
    "mobileclip2_b": {
        "metadata": {
            "description": (
                "MobileCLIP2-B image-text model. ViT-B/16 vision tower with a "
                "convolutional patch stem (mci) and a 12-layer CLIP text "
                "transformer, image resolution 224x224. Trained by Apple on "
                "DFNDR-2B (13B seen samples) with improved multi-modal "
                "reinforced training."
            ),
            "params": 149778117,
            "path": "mobileclip2",
        },
        "kaggle_handle": "kaggle://keras/mobileclip2/keras/mobileclip2_b/1",
    },
}
