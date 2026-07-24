ARG PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim

FROM ${PYTHON_IMAGE}

WORKDIR /app

ARG DEBIAN_MIRROR=https://mirrors.aliyun.com/debian
ARG DEBIAN_SECURITY_MIRROR=https://mirrors.aliyun.com/debian-security
ARG UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG UV_VERSION=0.11.28

RUN sed -i \
      -e "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
      -e "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
    /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && python -m pip install --no-cache-dir \
         --index-url "${UV_DEFAULT_INDEX}" \
         "uv==${UV_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}
ENV PATH="/opt/venv/bin:$PATH"

RUN uv sync --locked --no-dev --no-install-project

COPY . .
RUN cd LinearFold && make
RUN cp config_docker.py config.py && cp tasks_docker.py tasks.py

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV REDIS_HOST=redis

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "120", "wsgi:app"]
