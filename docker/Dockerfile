FROM python:3.9

# Build and publish crates with pyo3, cffi and uniffi bindings as well as
# rust binaries as python packages
RUN pip install maturin==1.0.1

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt/notebooks

RUN pip install -r requirements.txt

# Clear build dir in order to use `ls` operator for installing the wheel later
RUN rm -rf /opt/notebooks/target/wheels/
# Build native code to wheel
RUN maturin build --release
# Install the builded wheel
RUN pip install /opt/notebooks/target/wheels/$(ls /opt/notebooks/target/wheels/)
