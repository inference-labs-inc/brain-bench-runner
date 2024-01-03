import tensorflow as tf
import onnxruntime as ort
import numpy as np


def run_tflite_inference(model_path, input_data):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


def run_onnx_inference(model_path, input_data):
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output_data = sess.run([output_name], {input_name: input_data})[0]
    return output_data
