FROM python:3.9

RUN pip install maturin

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt/notebooks

RUN pip install numpy pandas matplotlib scikit-learn scipy gmr torch

RUN pip install thegolem
RUN pip install notebook
