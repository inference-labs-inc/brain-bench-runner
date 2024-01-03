import logging
import os
import json
import datetime
from util.system_stats import get_machine_name

logger = logging.getLogger(__name__)


def assert_success(function):
    def wrapper(*args, **kwargs):
        res = function(*args, **kwargs)
        if (
            not isinstance(res, dict)
            and not isinstance(res, tuple)
            and res is not True
            or res is None
        ):
            logger.error(
                "Response from failed function {}:\n {}".format(function.__name__, res)
            )
            raise Exception(
                "Function {} failed to execute successfully.".format(function.__name__)
            )
        return res

    return wrapper


def check_file_sizes_after_run(file_list, toplevel=False):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            wrapper.__name__ = "check_file_sizes_after_run(" + str(func.__name__) + ")"
            print(
                f"Running function {func.__name__} and checking file sizes after execution..."
            )

            result, metrics = func(self, *args, **kwargs)

            file_sizes = {}
            for file in file_list:
                try:
                    file_sizes[file.split(".")[0] + "_size_bytes"] = os.path.getsize(
                        os.path.join("models", self.model, file)
                        if not toplevel
                        else os.path.join(file)
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to get size of file {file} due to an error:\n",
                        exc_info=e,
                    )

            metrics["file_sizes"] = file_sizes
            return result, metrics

        return wrapper

    return decorator


def check_proof_size(file_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            wrapper.__name__ = "check_proof_size(" + str(func.__name__) + ")"
            print(
                f"Running function {func.__name__} and checking proof size after execution..."
            )

            result, metrics = func(self, *args, **kwargs)

            with open(os.path.join("models", self.model, file_name), "r") as f:
                json_data = json.load(f)
                proof_size = len(json_data["proof"])

            metrics["file_sizes"] = {"proof_size_bytes": proof_size}
            return result, metrics

        return wrapper

    return decorator


def generate_json_with_metrics(metrics, func, execution_time):
    output = {
        "meta": {"lastUpdated": datetime.datetime.now().isoformat()},
        "frameworks": {
            "ezkl": {
                get_machine_name(): {
                    func.__name__: {
                        "name": func.__name__,
                        "results": [
                            {
                                "metrics": metrics,
                                "name": func.__name__,
                                "time": {"secs": execution_time},
                            }
                        ],
                    }
                }
            }
        },
    }
    print(json.dumps(output, indent=4))
