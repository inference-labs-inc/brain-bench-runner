#!/usr/bin/env python3

import logging
import util.logging
import fire

from constants.constants import ProvingSystem, supported_models, Model

logger = logging.getLogger(__name__)

def print_title(func):
    def wrapper(self):
        with open('./constants/title.txt', 'r') as title:
            print("\033[35m")
            print(title.read())
            print("\033[0m")
        return func(self)
    return wrapper

class Benchmark(object):
    @print_title
    def __init__(self):
        self.__doc__ = "A zk benchmarking tool designed to run on client machines and report information relevant to the performance of zkML models."
        pass
    def run(self, model=Model.MNIST, iterations=5, proving_system=ProvingSystem.EZKL):
        # do something here
        if(model not in supported_models[proving_system]):
            logger.fatal("Unsupported model for proving system")
            logger.info("Supported models for selected proving system include {}".format(supported_models[proving_system]))
            exit(1)

        if(proving_system == ProvingSystem.EZKL):
            from proving_systems.ezkl import EZKL
            ezkl = EZKL(model, iterations)
            try:
                ezkl.run_all()
                logger.info("EZKL benchmark completed successfully.")
            except Exception as e:
                logger.fatal("Failed to run EZKL benchmark due to an error:\n", exc_info=e)
                exit(1)


if __name__ == "__main__":
    fire.Fire(Benchmark)
