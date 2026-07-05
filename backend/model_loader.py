import torch
import segmentation_models_pytorch as smp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def load_model(self, weights_path="weights/best_multitask_unet.pth"):
        print(f"Загрузка модели на {DEVICE}...")
        model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b3',
            encoder_weights=None, # Веса уже обучены
            in_channels=3,
            classes=1,
            decoder_attention_type='scse',
            aux_params=dict(
                pooling='avg',
                dropout=0.4,
                activation=None,
                classes=2
            )
        )
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        self.model = model
        print("Модель успешно загружена!")
        return model

    def get_model(self):
        return self.model

model_manager = ModelManager()