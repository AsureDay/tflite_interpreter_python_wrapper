from typing import Any
import numpy as np
from tensorflow import keras
import tensorflow as tf
from pathlib import Path


class TFLiteModel():
    def __init__(self, model, ) -> None:
        if isinstance(model, keras.Sequential):
            model = TFLiteModel.quantize_model(model)
        
        self.interpreter = tf.lite.Interpreter(model)
        self.tensor_details = self.interpreter.get_tensor_details()
        # Получение информации о входе и выходе для инференса
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def __call__(self, x) -> Any:
        input_data = np.array(x, dtype=np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])
    

    def get_details(self):
        """### Получить детали o каждом элементи нейросети

        Returns:
            `tuple`: weights:List, quantization_params:List, namesList
            
            `weights` -- веса
            **ВАЖНО**: веса некоторых элементов хранят текущее состояне тензора(можео спутать с обучаемыми весами), 
            например входной слой будет иметь данные предыдущего входа

            `quantization_parametrs` -- параметры квантизации это список пар из `(scale, zero_point)`
            для слоя co следующими параметрами `float32` веса имеют вид `(int8_weights - zero_point)*scale`

            `names` -- имена каждого слоя
        """
        weights = []
        quantization_params = []
        names = []
        for details in self.tensor_details:
            weights.append(self.interpreter.get_tensor(details['index']))
            quantization_params.append(details['quantization'])
            names.append(details['name'])

        return weights, quantization_params, names


    @staticmethod
    def quantize_model(base_model:keras.Sequential, calibrate_x, save_f=None):
        """### Квантизует модель в INT8 используя calibration_data для калибровки

        Args:
            `base_model` (keras.Sequential): модель
            `calibrate_x` (iterable): данные для калибровки (только входные)
        
        Returns:
            `quantized_model`: Бинарный вид квантизованной модели
        """
        def representative_dataset():
            for e in calibrate_x:
                yield [e.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.uint8
        converter.inference_output_type = tf.uint8 
        quantized_model = converter.convert()
        Path(save_f).write_bytes(quantized_model) if save_f else None
        return quantized_model