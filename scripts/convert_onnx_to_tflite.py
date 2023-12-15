#!/usr/bin/env python3

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import fire

def convert_onnx_to_tflite(onnx_model_path="./models/MNIST/network.onnx", tf_model_path="./models/MNIST/network.tf", tflite_model_path="./models/MNIST/network.tflite"):
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    fire.Fire(convert_onnx_to_tflite)
