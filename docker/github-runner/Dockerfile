FROM ubuntu:22.04

ARG RUNNER_VERSION=2.319.0
ENV DEBIAN_FRONTEND=noninteractive

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

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo \
       "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
       $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce-cli \
        docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.13 get-pip.py \
    && rm get-pip.py \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python3 \
    && python -m pip install --upgrade pip

RUN useradd -m runner

WORKDIR /opt/actions-runner

RUN curl -L "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" -o actions-runner.tar.gz \
    && tar xzf actions-runner.tar.gz \
    && rm actions-runner.tar.gz \
    && ./bin/installdependencies.sh

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /home/runner && chown -R runner:runner /home/runner

ENV RUNNER_VERSION=${RUNNER_VERSION}

ENTRYPOINT ["/entrypoint.sh"]
