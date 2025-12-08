FROM ubuntu:22.04

ARG RUNNER_VERSION=2.319.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
    git \
    procps \
    tar \
    unzip \
    && rm -rf /var/lib/apt/lists/*

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
