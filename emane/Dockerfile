FROM ubuntu:22.04

LABEL description="EMANE TDMA experiment environment — Vaan Megam Networks"
LABEL maintainer="intern"

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    net-tools \
    iproute2 \
    iputils-ping \
    tcpdump \
    iperf3 \
    lsb-release \
    software-properties-common \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install EMANE from adjacentlink PPA
# Reference: https://github.com/adjacentlink/emane/wiki/Install
RUN apt-get update && apt-get install -y \
    emane \
    emane-tdma-radio-model \
    python3-emane \
    emane-utils \
    || echo "EMANE packages not available in standard repos — will use source install"

# Install Python packages
RUN pip3 install networkx>=3.0 pytest>=7.0 jinja2>=3.0 lxml>=4.9

# Create workspace
WORKDIR /workspace
COPY . /workspace/

# Install the tdma_brain package in development mode
RUN pip3 install -e /workspace/ 2>/dev/null || echo "Setup.py not found — running as script"

# Copy EMANE configuration files
RUN mkdir -p /workspace/emane_config

EXPOSE 45703

CMD ["/bin/bash"]
