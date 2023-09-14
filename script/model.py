from config                        import *
import segmentation_models_pytorch as smp

def model_setting():
    model = getattr(smp, MODEL)(
        encoder_name    = ENCODER_NAME, 
        encoder_weights = ENCODER_WEIGHTS, 
        classes         = CLASSES, 
        activation      = ACTIVATION
    )
    model = model.to(DEVICE)
    return model