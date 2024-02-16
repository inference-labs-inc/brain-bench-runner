# pyBenchZKML

Benchmarks for ZKML frameworks, as shown on the public [BrainBench] site. Based on the work from [zkonduit/zkml-framework-benchmarks], this repo extends benchmarked frameworks and adds new metrics to the benchmarks. This repo also includes a GitHub actions script to automate the benchmarking and aggregation process on a variety of machines.

## Getting started (from [zkonduit/zkml-framework-benchmarks])

To run the benchmarks, you need to first install python (version 3.9.18 specifically), rust, rust jupyter kernel, risc0 toolchain, and scarb on your unix-like machine.

First, you will need to install ezkl cli version 8.0.1 which you can do from [here](https://github.com/zkonduit/ezkl/releases/tag/v8.0.1)

To install the other required dependencies run:

```bash
bash install_dep_run.sh
```

For windows systems, you will need to install the dependencies manually.

For linux systems, you may need to install jq.

```bash
sudo apt-get install jq
```

You may run the following to activate the virtual environment if had been deactivated.

```bash
source .env/bin/activate
```

For linux systems you will also need to set the OS environment variable to linux (default is Mac).

```bash
export OS=linux
```

Finally run this cargo nextest test command to get the benchmarks:

```bash
source .env/bin/activate; cargo nextest run benchmarking_tests::tests::run_benchmarks_ --no-capture
```

The data will stored in a `benchmarks.json` file in the root directory.

If you run into any issues feel free to open a PR and we will try to help you out ASAP.

Enjoy! :)

[BrainBench]: https://brainbench.xyz/ "BrainBench site"
[zkonduit/zkml-framework-benchmarks]: https://github.com/zkonduit/zkml-framework-benchmarks "ZKML Framework Benchmarks Repo"
