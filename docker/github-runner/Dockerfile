# Use the official Ubuntu 22.04 as a parent image
FROM ubuntu:22.04

# Set environment variables
ARG RUNNER_VERSION=2.319.0
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
    git \
    gnupg \
    gpg-agent \
    dirmngr \
    procps \
    software-properties-common \
    tar \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI 
# De runner only needs the Docker CLI to communicate with the Docker daemon on the host
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo \
       "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
       $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce-cli \
        docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.13
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.13 get-pip.py \
    && rm get-pip.py \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python3 \
    && python -m pip install --upgrade pip

# Create a non-root user to run the GitHub Actions runner
RUN useradd -m runner

# Switch to the runner user
WORKDIR /opt/actions-runner

# Download and install the GitHub Actions runner
RUN curl -L "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" -o actions-runner.tar.gz \
    && tar xzf actions-runner.tar.gz \
    && rm actions-runner.tar.gz \
    && ./bin/installdependencies.sh

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set permissions for the runner directory
RUN mkdir -p /home/runner && chown -R runner:runner /home/runner

# Set environment variable for the runner version
ENV RUNNER_VERSION=${RUNNER_VERSION}

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
