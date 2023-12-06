from util.monitor_process import run_with_monitoring
import os
import ezkl
import json
from PIL import Image
from util.helpers import assert_success
from constants.constants import Model
import logging
import numpy as np

logger = logging.getLogger(__name__)

model_path = os.path.join('network.onnx')
compiled_model_path = os.path.join('network.ezkl')
pk_path = os.path.join('pk.key')
vk_path = os.path.join('vk.key')
settings_path = os.path.join('settings.json')
srs_path = os.path.join('kzg.srs')
witness_path = os.path.join('witness.json')
data_path = os.path.join('input.json')
cal_path = os.path.join('cal_data.json')
proof_path = os.path.join('proof.json')
abi_path = os.path.join('abi.json')
sol_code_path = os.path.join('verifier.sol')
address_path = os.path.join('address.txt')


def setup_mnist():
    try:
        input_image_path = os.path.join('input.png')
        with Image.open(input_image_path) as img:
            # Convert to grayscale
            img = img.convert("L")
            # Resize to 28x28 (for MNIST)
            img = img.resize((28, 28))
            # Convert to a numpy array
            img_array = np.array(img)
            # Normalize the image data to 0-1 range
            img_array = img_array / 255.0
            # Reshape to match the input shape expected by the model: [1, 28, 28]
            x = img_array.reshape((1, 28, 28))
        data_array = x.reshape([-1]).tolist()
        data = dict(input_data = [data_array])
        cal_path = os.path.join('cal_data.json')
        # Serialize data into file:
        json.dump( data, open(cal_path, 'w' ))
    except:
        logger.fatal("Failed to load image file at {}".format(input_image_path))
        exit(1)

def setup_not_found(model):
    logger.fatal("No setup configuration for model {}".format(model))
    exit(1)

def perform_setup(model):
    return {
        Model.MNIST: setup_mnist()
    }.get(model, setup_not_found(model))

class EZKL:

    def __init__(self, model, iterations):
        logger.info("Initializing EZKL benchmark...")
        self.run_args = ezkl.PyRunArgs()
        self.run_args.input_visibility = "public"
        self.run_args.param_visibility = "fixed"
        self.run_args.output_visibility = "public"
        self.run_args.variables = [("batch_size", 1)]
        self.model = model
        self.iterations = iterations
        self.paths = self.setup_paths(model)
        perform_setup(model)

    def setup_paths(self, model):
        paths = dict()
        paths['model_path'] = os.path.join('models', model, 'network.onnx')
        paths['compiled_model_path'] = os.path.join('models', model, 'network.ezkl')
        paths['pk_path'] = os.path.join('models', model, 'pk.key')
        paths['vk_path'] = os.path.join('models', model, 'vk.key')
        paths['settings_path'] = os.path.join('models', model, 'settings.json')
        paths['srs_path'] = os.path.join('models', model, 'kzg.srs')
        paths['witness_path'] = os.path.join('models', model, 'witness.json')
        paths['data_path'] = os.path.join('models', model, 'input.json')
        paths['cal_path'] = os.path.join('models', model, 'cal_data.json')
        paths['proof_path'] = os.path.join('models', model, 'proof.json')
        paths['abi_path'] = os.path.join('models', model, 'abi.json')
        paths['sol_code_path'] = os.path.join('models', model, 'verifier.sol')
        paths['address_path'] = os.path.join('models', model, 'address.txt')
        return paths

    def run_all(self):
        print("Running EZKL benchmark on {} for {} iterations".format(self.model, self.iterations))
        run_with_monitoring()
        return True

    @assert_success
    def gen_settings(self):
        return ezkl.gen_settings(model_path, settings_path, py_run_args=self.run_args)

    @assert_success
    def calibrate_settings(self):
        return ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources", scales = [1, 7])

    @assert_success
    def compile(self):
        return ezkl.compile(model_path, compiled_model_path, settings_path)

    @assert_success
    def get_srs(self):
        return ezkl.get_srs(srs_path, settings_path)

    @assert_success
    def gen_witness(self):
        return ezkl.gen_witness(witness_path, data_path, model_path, settings_path)

    @assert_success
    def mock_prove(self):
        return ezkl.mock(witness_path, compiled_model_path)

    @assert_success
    def setup(self):
        res = ezkl.setup(
            compiled_model_path,
            vk_path,
            pk_path,
            srs_path,
        )
        assert os.path.isfile(vk_path)
        assert os.path.isfile(pk_path)
        assert os.path.isfile(settings_path)
        return res

    @assert_success
    def prove(self):
        res = ezkl.prove(
            witness_path,
            compiled_model_path,
            pk_path,
            proof_path,
            srs_path,
            "single",
        )
        assert os.path.isfile(proof_path)
        return res

    @assert_success
    def verify(self):
        return ezkl.verify(
            proof_path,
            vk_path,
            data_path,
            "single",
        )

    @assert_success
    def create_evm_verifier(self):
        return ezkl.create_evm_verifier(
            vk_path,
            srs_path,
            settings_path,
            sol_code_path,
            abi_path,
        )

    @assert_success
    def deploy_evm(self):
        return ezkl.deploy_evm(
            address_path,
            sol_code_path,
            'http://127.0.0.1:3030'
        )

    @assert_success
    def verify_evm(self):
        return ezkl.verify_evm(
            proof_path,
            addr,
            "http://127.0.0.1:3030"
        )
