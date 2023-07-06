import os

import onnxruntime as ort


class CrepeInferenceSession(ort.InferenceSession):
    def __init__(self, model='full', sess_options=None, providers=None, provider_options=None, **kwargs):
        model_path = os.path.join(os.path.dirname(__file__), 'assets', f'{model}.onnx')
        super().__init__(model_path, sess_options, providers, provider_options, **kwargs)
