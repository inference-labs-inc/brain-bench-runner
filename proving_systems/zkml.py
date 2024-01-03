from util.monitor_process import run_with_monitoring
import os
import json
from PIL import Image
from util.helpers import assert_success, check_file_sizes_after_run
from constants.constants import Model
import logging
import numpy as np
import torch
import onnxruntime as ort
import re
import subprocess
import tempfile
import tensorflow as tf

import msgpack
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from util.inference import run_tflite_inference
from tqdm import tqdm
from util.system_stats import get_machine_name

logger = logging.getLogger(__name__)
addr = None


def setup_mnist(**kwargs):
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=kwargs.get("model_path"))
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]["shape"]
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]["index"])

        data = dict(input_data=input_data.tolist(), output_data=output_data.tolist())

        # Serialize data into file:
        json.dump(data, open(kwargs.get("input_path"), "w"))
        return True
    except Exception as e:
        logger.fatal("Failed to setup MNIST due to an error:\n", exc_info=e)
        exit(1)


def setup_not_found(model):
    logger.fatal("No setup configuration for model {}".format(model))
    exit(1)


def perform_setup(model, **kwargs):
    setup_function = {Model.MNIST.value: setup_mnist}.get(model)

    if setup_function:
        setup_function(**kwargs)
    else:
        setup_not_found(model)


class ZKML:
    def __init__(self, model, iterations):
        logger.info("Initializing ZKML benchmark...")
        self.model = model.value
        self.iterations = iterations
        self.paths = self.setup_paths(self.model)
        perform_setup(
            self.model,
            input_path=self.paths.get("input_path"),
            cal_path=self.paths.get("cal_path"),
            model_path=self.paths.get("model_path"),
        )

    def setup_paths(self, model):
        paths = dict()
        paths["model_path"] = os.path.join("models", model, "network.tflite")
        paths["msgpack_model_path"] = os.path.join("models", model, "network.msgpack")
        paths["input_path"] = os.path.join("models", model, "input.msgpack")
        paths["output_path"] = os.path.join("output", get_machine_name(), "zkml.json")
        paths["test_circuit_path"] = os.path.join(
            "zkml", "target", "release", "test_circuit"
        )
        paths["time_circuit_path"] = os.path.join(
            "zkml", "target", "release", "time_circuit"
        )
        return paths

    def run_all(self):
        print(
            "Running ZKML benchmark on {} for {} iterations".format(
                self.model, self.iterations
            )
        )
        aggregated_metrics = {}
        for method_name in ["accuracy_checker", "predict_via_zkml", "prove"]:
            method = getattr(self, method_name)
            _, metrics = method()
            print(method_name, metrics)
            for metric_name, metric_value in metrics.items():
                aggregated_metrics[f"{method_name}_{metric_name}"] = metric_value
        aggregated_metrics["proving_key_size"] = aggregated_metrics["prove_file_sizes"][
            "pkey_size_bytes"
        ]
        aggregated_metrics["verification_key_size"] = aggregated_metrics[
            "prove_file_sizes"
        ]["vkey_size_bytes"]
        # Output the aggregated metrics as JSON
        output_dir = os.path.dirname(self.paths.get("output_path"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(self.paths.get("output_path"), "w") as json_file:
            json.dump({"results": [aggregated_metrics]}, json_file, indent=4)
        return True

    @check_file_sizes_after_run(
        ["pkey", "vkey", "proof", "public_vals", "out.msgpack", "params_kzg/17.params"],
        toplevel=True,
    )
    @run_with_monitoring()
    def prove(self):
        return subprocess.run(
            [
                self.paths.get("time_circuit_path"),
                self.paths.get("msgpack_model_path"),
                self.paths.get("input_path"),
                "kzg",
            ],
            stdout=subprocess.PIPE,
        )

    @run_with_monitoring()
    def predict_via_zkml(self):
        return subprocess.run(
            [
                self.paths.get("test_circuit_path"),
                self.paths.get("msgpack_model_path"),
                self.paths.get("input_path"),
                "kzg",
            ],
            stdout=subprocess.PIPE,
        )

    def predict(self, image):
        image = (image * 1024).numpy().astype(np.int32)
        input_path = self.paths.get("input_path")
        with open(input_path, "wb") as f:
            data = [
                {"idx": 0, "shape": (1, 28, 28, 1), "data": image.flatten().tolist()}
            ]
            f.write(msgpack.packb(data, use_bin_type=True))

        result, metrics = self.predict_via_zkml()
        result = list(
            filter(
                lambda x: "final out[" in x, result.stdout.decode("utf-8").split("\n")
            )
        )
        result = result[[i for i, s in enumerate(result) if "final out[0]" in s][1] :]
        predictions = [None] * 10
        for line in result:
            index = int(line.split()[1].replace("out[", "").replace("]", ""))
            value = float(line.split()[3])
            predictions[index] = value
        prediction = np.argmax(predictions)
        return prediction, metrics

    def accuracy_checker(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=True
        )
        for i in range(self.iterations):
            image = next(iter(test_loader))
            total = 0
            correct = 0

            prediction, metrics = self.predict(image[0])
            tflite_predictions = run_tflite_inference(
                self.paths.get("model_path"), np.squeeze(image[0].numpy(), axis=1)
            )
            tflite_prediction = np.argmax(tflite_predictions)

            total += 1
            if prediction == tflite_prediction:
                correct += 1
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        metrics["accuracy"] = accuracy
        return accuracy, metrics
