# Moebious

ML algorithms benchmarking and synthetic data generation framework

<p align="center">
<img src ="moebius.svg" width="300">
</p>

## Installation

Some of the library code is a native implementation written in Rust.
You can either use Docker (in which case a single `docker-compose up`
command will be sufficient), or manually compile the library into a
wheel and install using the pip dependency manager.

### Docker

```bash
git clone https://github.com/D3lph1/moebius.git moebius && cd moebius
```

```bash
docker-compose up -d
```

### Manual
Clone the repo:
```bash
git clone https://github.com/D3lph1/moebius.git moebius && cd moebius
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Install necessary Rust compiler (tested with version `1.69.0`):
```bash
curl --proto '=https' --tls1.2 -sSf https://sh.rustup.rs/ | sh -s --default-toolchain=1.69.0
```

Install dev-dependency [Maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin==1.0.1
```

Build the library:

```bash
maturin build --release
```

Install the library:

```bash
pip install <path> --force-reinstall
```

where `<path>` is a path to the wheel built by `maturin build` command.

## Usage

See examples of usage in `experiments` folder.

## Attribution

[Image](https://commons.wikimedia.org/wiki/File:Moebius_strip.svg) of Moebius strip by 	Krishnavedala /
[CC BY](https://commons.wikimedia.org/wiki/User:Krishnavedala)

## License

This code is published under the [MIT license](https://opensource.org/licenses/MIT). This means you
can do almost anything with it, as long as the copyright notice and the accompanying license file
is left intact.
