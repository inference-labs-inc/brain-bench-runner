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
from tqdm import tqdm


logger = logging.getLogger(__name__)
addr = None

def setup_mnist(**kwargs):
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=kwargs.get('model_path'))
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])

        data = dict(input_data = input_data.tolist(), output_data=output_data.tolist())

        # Serialize data into file:
        json.dump(data, open(kwargs.get('input_path'), 'w' ))
        return True
    except Exception as e:
        logger.fatal("Failed to setup MNIST due to an error:\n", exc_info=e)
        exit(1)

def setup_not_found(model):
    logger.fatal("No setup configuration for model {}".format(model))
    exit(1)

def perform_setup(model, **kwargs):
    setup_function = {
        Model.MNIST.value: setup_mnist
    }.get(model)

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
        perform_setup(self.model, input_path=self.paths.get('input_path'), cal_path=self.paths.get('cal_path'), model_path=self.paths.get('model_path'))

    def setup_paths(self, model):
        paths = dict()
        paths['model_path'] = os.path.join('models', model, 'network.tflite')
        paths['msgpack_model_path'] = os.path.join('models', model, 'network.msgpack')
        paths['input_path'] = os.path.join('models', model, 'input.msgpack')
        paths['output_path'] = os.path.join('output', model, 'output.json')
        paths['test_circuit_path'] = os.path.join('zkml', 'target', 'release', 'test_circuit')
        return paths

    def run_all(self):
        print("Running ZKML benchmark on {} for {} iterations".format(self.model, self.iterations))
        aggregated_metrics = {}
        for method_name in ['accuracy_checker']:
            method = getattr(self, method_name)
            _, metrics = method()
            print(method_name, metrics)
            aggregated_metrics[method_name] = metrics
        # Output the aggregated metrics as JSON
        output_dir = './output/Darwin-arm-24Ghz-32GB'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'zkml.json'), 'w') as json_file:
            json.dump({"results": [aggregated_metrics]}, json_file, indent=4)
        return True

    @run_with_monitoring()
    def predict_via_zkml(self):
        return subprocess.run([self.paths.get('test_circuit_path'), self.paths.get('msgpack_model_path'),
                                self.paths.get('input_path'), "kzg"], stdout=subprocess.PIPE)


    def predict(self, image):
        image = (image * 1024).numpy().astype(np.int32)
        input_path = self.paths.get('input_path')
        with open(input_path, "wb") as f:
            data = [{"idx": 0, "shape": (
                1, 28, 28, 1), "data": image.flatten().tolist()}]
            f.write(msgpack.packb(
                data, use_bin_type=True))

        result, metrics = self.predict_via_zkml()
        pattern = re.compile(r"final out\[(\d)\] x: (-?\d+)")

        scores = []
        for line in result.stdout.decode("utf-8").split("\n")[-30:]:
            match = pattern.match(line)
            if match:
                index = int(match.group(1))
                score = int(match.group(2))

                assert index == len(scores)
                scores.append(score)

        assert len(scores) == 10
        prediction = np.argmax(scores)
        return prediction, metrics


    def accuracy_checker(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False)

        total = 0
        correct = 0

        for image, labels in tqdm(test_loader):

            prediction = self.predict(image)

            total += 1
            if prediction == labels.item():
                correct += 1

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
