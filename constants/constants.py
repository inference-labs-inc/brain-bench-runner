from enum import Enum

class ProvingSystem(Enum):
    EZKL = "EZKL"
    ZKML = "ZKML"
class Model(Enum):
    MNIST = "MNIST"

supported_models = {
    ProvingSystem.EZKL.value: [Model.MNIST],
    ProvingSystem.ZKML.value: [Model.MNIST]
}
