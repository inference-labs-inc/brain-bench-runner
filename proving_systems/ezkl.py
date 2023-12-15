from util.monitor_process import run_with_monitoring
import os
import ezkl
import json
from PIL import Image
from util.helpers import assert_success, check_file_sizes_after_run
from constants.constants import Model
import logging
import numpy as np
import torch
import onnxruntime as ort
import torchvision.transforms as transforms
import torchvision


logger = logging.getLogger(__name__)
addr = None

def setup_mnist(**kwargs):
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=True)

        # Select a random image from the dataset
        image, _ = next(iter(test_loader))
        img_array = image.numpy().squeeze()
        x = img_array.reshape((1, 28, 28))
        data_array = x.reshape([-1]).tolist()

        # Load the ONNX model
        sess = ort.InferenceSession(kwargs.get('model_path'))

        # Run the model to get the output data
        x = np.array(x, dtype=np.float32)
        output_data = sess.run(None, {'input_0': x})[0]

        data = dict(input_data = [data_array], output_data=output_data.tolist())

        # Serialize data into file:
        json.dump(data, open(kwargs.get('cal_path'), 'w' ))
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

class EZKL:

    def __init__(self, model, iterations):
        logger.info("Initializing EZKL benchmark...")
        self.run_args = ezkl.PyRunArgs()
        self.run_args.input_visibility = "public"
        self.run_args.param_visibility = "fixed"
        self.run_args.output_visibility = "public"
        self.run_args.variables = [("batch_size", 1)]
        self.model = model.value
        self.iterations = iterations
        self.paths = self.setup_paths(self.model)
        perform_setup(self.model, input_path=self.paths.get('input_path'), cal_path=self.paths.get('cal_path'), model_path=self.paths.get('model_path'))

    def setup_paths(self, model):
        paths = dict()
        paths['model_path'] = os.path.join('models', model, 'network.onnx')
        paths['compiled_model_path'] = os.path.join('models', model, 'network.ezkl')
        paths['pk_path'] = os.path.join('models', model, 'pk.key')
        paths['vk_path'] = os.path.join('models', model, 'vk.key')
        paths['settings_path'] = os.path.join('models', model, 'settings.json')
        paths['srs_path'] = os.path.join('models', model, 'kzg.srs')
        paths['witness_path'] = os.path.join('models', model, 'witness.json')
        paths['cal_path'] = os.path.join('models', model, 'cal_data.json')
        paths['proof_path'] = os.path.join('models', model, 'proof.json')
        paths['abi_path'] = os.path.join('models', model, 'abi.json')
        paths['sol_code_path'] = os.path.join('models', model, 'verifier.sol')
        paths['address_path'] = os.path.join('models', model, 'address.txt')
        paths['input_path'] = os.path.join('models', model, 'input.png')
        return paths

    def run_all(self):
        print("Running EZKL benchmark on {} for {} iterations".format(self.model, self.iterations))
        aggregated_metrics = {}
        for method_name in ['gen_settings', 'calibrate_settings', 'compile', 'get_srs', 'gen_witness', 'mock_prove', 'setup', 'prove', 'verify', 'create_evm_verifier', 'deploy_evm', 'verify_evm']:
            method = getattr(self, method_name)
            _, metrics = method()
            print(method_name, metrics)
            for metric_name, metric_value in metrics.items():
                aggregated_metrics[f"{method_name}_{metric_name}"] = metric_value
        # Add accuracy checker percentage to aggregated metrics
        accuracy_checker_percentage, average_gen_witness_time = self.accuracy_checker()
        aggregated_metrics["accuracy_checker_percentage"] = accuracy_checker_percentage
        aggregated_metrics["average_gen_witness_time"] = average_gen_witness_time
        print(aggregated_metrics)
        # Output the aggregated metrics as JSON
        output_dir = './output/Darwin-arm-24Ghz-32GB'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'ezkl.json'), 'w') as json_file:
            json.dump({"results": [aggregated_metrics]}, json_file, indent=4)
        return True

    @assert_success
    @check_file_sizes_after_run(['settings.json'])
    @run_with_monitoring()
    def gen_settings(self):
        return ezkl.gen_settings(self.paths.get('model_path'), self.paths.get('settings_path'), py_run_args=self.run_args)

    @assert_success
    @check_file_sizes_after_run(['settings.json'])
    @run_with_monitoring()
    def calibrate_settings(self):
        return ezkl.calibrate_settings(self.paths.get('cal_path'), self.paths.get('model_path'), self.paths.get('settings_path'), "resources", scales = [1, 7])

    @assert_success
    @check_file_sizes_after_run(['network.ezkl'])
    @run_with_monitoring()
    def compile(self):
        return ezkl.compile_circuit(self.paths.get('model_path'), self.paths.get('compiled_model_path'), self.paths.get('settings_path'))

    @assert_success
    @check_file_sizes_after_run(['kzg.srs'])
    @run_with_monitoring()
    def get_srs(self):
        return ezkl.get_srs(self.paths.get('settings_path'))

    @assert_success
    @check_file_sizes_after_run(['witness.json'])
    @run_with_monitoring()
    def gen_witness(self):
        return ezkl.gen_witness(self.paths.get('cal_path'), self.paths.get('compiled_model_path'), self.paths.get('witness_path'))

    @assert_success
    @run_with_monitoring()
    def mock_prove(self):
        return ezkl.mock(self.paths.get('witness_path'), self.paths.get('compiled_model_path'))

    @assert_success
    @check_file_sizes_after_run(['pk.key', 'vk.key', 'settings.json'])
    @run_with_monitoring()
    def setup(self):
        res = ezkl.setup(
            self.paths.get('compiled_model_path'),
            self.paths.get('vk_path'),
            self.paths.get('pk_path'),
        )
        assert os.path.isfile(self.paths.get('vk_path'))
        assert os.path.isfile(self.paths.get('pk_path'))
        assert os.path.isfile(self.paths.get('settings_path'))
        return res

    @assert_success
    @check_file_sizes_after_run(['proof.json'])
    @run_with_monitoring()
    def prove(self):
        res = ezkl.prove(
            self.paths.get('witness_path'),
            self.paths.get('compiled_model_path'),
            self.paths.get('pk_path'),
            self.paths.get('proof_path'),
            "single",
        )
        assert os.path.isfile(self.paths.get('proof_path'))
        return res

    @assert_success
    @run_with_monitoring()
    def verify(self):
        return ezkl.verify(
            self.paths.get('proof_path'),
            self.paths.get('settings_path'),
            self.paths.get('vk_path'),

        )

    @assert_success
    @check_file_sizes_after_run(['verifier.sol', 'abi.json'])
    @run_with_monitoring()
    def create_evm_verifier(self):
        return ezkl.create_evm_verifier(
            self.paths.get('vk_path'),
            self.paths.get('settings_path'),
            self.paths.get('sol_code_path'),
            self.paths.get('abi_path'),
        )

    @assert_success
    @check_file_sizes_after_run(['address.txt'])
    @run_with_monitoring()
    def deploy_evm(self):
        return ezkl.deploy_evm(
            self.paths.get('address_path'),
            self.paths.get('sol_code_path'),
            'http://127.0.0.1:3030'
        )


    @assert_success
    @run_with_monitoring()
    def verify_evm(self):
        with open(self.paths.get('address_path'), 'r') as addr_file:
            addr = addr_file.read()
        return ezkl.verify_evm(
            addr,
            self.paths.get('proof_path'),
            "http://127.0.0.1:3030"
        )

    # Note: This function does not test model accuracy, instead it tests how accurately the proving system is able to replicate the ONNX model.
    def accuracy_checker(self):
        correct_predictions = 0
        total_gen_witness_time = 0
        runs = 10
        for _ in range(runs):
            # setup mnist
            setup_mnist(input_path=self.paths.get('input_path'), cal_path=self.paths.get('cal_path'), model_path=self.paths.get('model_path'))
            data = json.load(open(self.paths.get('cal_path'), 'r'))
            expected_output = torch.argmax(torch.tensor(data['output_data'][0]))
            _, gen_witness_metrics = self.gen_witness()
            total_gen_witness_time += gen_witness_metrics['execution_time']
            witness_output = json.load(open(self.paths.get('witness_path'), 'r'))["outputs"][0]
            predicted_output = torch.argmax(torch.tensor([ezkl.vecu64_to_float(i, 0) for i in witness_output], dtype=torch.float32))
            print(predicted_output, expected_output)
            if predicted_output == self.expected_output:
                correct_predictions += 1
        accuracy = correct_predictions / runs
        avg_gen_witness_time = total_gen_witness_time / runs
        print(f"Accuracy: {accuracy * 100}%")
        print(f"Average gen witness time: {avg_gen_witness_time} seconds")
        return accuracy, avg_gen_witness_time

    def cleanup(self):
        files_to_remove = ['compiled_model_path', 'pk_path', 'vk_path', 'settings_path', 'srs_path', 'witness_path', 'cal_path', 'proof_path', 'abi_path', 'sol_code_path', 'address_path']
        for file_key in files_to_remove:
            file_path = self.paths.get(file_key)
            if os.path.isfile(file_path):
                os.remove(file_path)
