cases of using tflite 
=====================
* case of quantization `TFLiteModel` `quantize_model` method
* case of getting `weights` from tflite binary file using `tflite.Interpreter ` `get_details` method
  **NOTE**: weights from `get_details` are hard readable for getting nn structure. Please use netron or smth like that to see nn structure
  this method usefull for get wights automatically  