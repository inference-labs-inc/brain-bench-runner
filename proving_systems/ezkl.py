from util.monitor_process import run_with_monitoring
import os
import ezkl
import json
from PIL import Image
from util.helpers import assert_success
from constants.constants import Model
import logging
import numpy as np
import torch
import onnxruntime as ort

logger = logging.getLogger(__name__)
addr = None

def setup_mnist(**kwargs):
    try:
        with Image.open(kwargs.get('input_path')) as img:
            img = img.convert("L")
            img = img.resize((28, 28))
            img_array = np.array(img)
            img_array = img_array / 255.0
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
        self.gen_settings()
        self.calibrate_settings()
        self.compile()
        self.get_srs()
        self.gen_witness()
        self.mock_prove()
        self.setup()
        self.prove()
        return True

    @assert_success
    @run_with_monitoring()
    def gen_settings(self):
        return ezkl.gen_settings(self.paths.get('model_path'), self.paths.get('settings_path'), py_run_args=self.run_args)

    @assert_success
    @run_with_monitoring()
    def calibrate_settings(self):
        return ezkl.calibrate_settings(self.paths.get('cal_path'), self.paths.get('model_path'), self.paths.get('settings_path'), "resources", scales = [1, 7])

    @assert_success
    @run_with_monitoring()
    def compile(self):
        return ezkl.compile_circuit(self.paths.get('model_path'), self.paths.get('compiled_model_path'), self.paths.get('settings_path'))

    @assert_success
    @run_with_monitoring()
    def get_srs(self):
        return ezkl.get_srs(self.paths.get('srs_path'), self.paths.get('settings_path'))

    @assert_success
    @run_with_monitoring()
    def gen_witness(self):
        return ezkl.gen_witness(self.paths.get('cal_path'), self.paths.get('compiled_model_path'), self.paths.get('witness_path'))

    @assert_success
    @run_with_monitoring()
    def mock_prove(self):
        return ezkl.mock(self.paths.get('witness_path'), self.paths.get('compiled_model_path'))

    @assert_success
    @run_with_monitoring()
    def setup(self):
        res = ezkl.setup(
            self.paths.get('compiled_model_path'),
            self.paths.get('vk_path'),
            self.paths.get('pk_path'),
            self.paths.get('srs_path'),
        )
        assert os.path.isfile(self.paths.get('vk_path'))
        assert os.path.isfile(self.paths.get('pk_path'))
        assert os.path.isfile(self.paths.get('settings_path'))
        return res

    @assert_success
    @run_with_monitoring()
    def prove(self):
        res = ezkl.prove(
            self.paths.get('witness_path'),
            self.paths.get('compiled_model_path'),
            self.paths.get('pk_path'),
            self.paths.get('proof_path'),
            self.paths.get('srs_path'),
            "single",
        )
        assert os.path.isfile(self.paths.get('proof_path'))
        return res

    @assert_success
    @run_with_monitoring()
    def verify(self):
        return ezkl.verify(
            self.paths.get('proof_path'),
            self.paths.get('vk_path'),
            self.paths.get('input_path'),
            "single",
        )

    @assert_success
    @run_with_monitoring()
    def create_evm_verifier(self):
        return ezkl.create_evm_verifier(
            self.paths.get('vk_path'),
            self.paths.get('srs_path'),
            self.paths.get('settings_path'),
            self.paths.get('sol_code_path'),
            self.paths.get('abi_path'),
        )

    @assert_success
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
        return ezkl.verify_evm(
            self.paths.get('proof_path'),
            addr,
            "http://127.0.0.1:3030"
        )
