# zkMNIST

Trained MNIST classifier model, implemented with circuitization and benchmarking. Designed to be extended beyond MNIST, and serve as a proof of concept benchmarking tool for zkML.

## Prerequisites

- [`Pipenv`](https://pipenv.pypa.io/en/latest/)

## Setup

### Launch [`Pipenv`] shell

```bash
pipenv shell
```

### Install dependencies

```bash
pipenv install
```

## Running the benchmark

### Run the benchmark

> [!NOTE]
> This script doesn't require `python` in the CLI in order to run - it's already specified in the shebang.

```bash
./benchmark.py
```

### Benchmarking options

```bash
./benchmark.py --help
```

[`Pipenv`]: https://pipenv.pypa.io/en/latest/ "Pipenv docs"
