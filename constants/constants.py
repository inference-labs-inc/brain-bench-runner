from enum import Enum

class ProvingSystem(Enum):
    EZKL = "EZKL"
    ZKML = "ZKML"
class Model(Enum):
    MNIST = "MNIST"

supported_models = {
    ProvingSystem.EZKL: [Model.MNIST],
    ProvingSystem.ZKML: [Model.MNIST]
}
