FROM continuumio/anaconda3:latest

RUN pip install maturin

# Update new packages
RUN apt-get update

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN apt install build-essential -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt/notebooks
