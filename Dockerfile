# Pull base image - Using Python 3.11 for better numba compatibility
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set work directory called `app`
RUN mkdir -p /code
WORKDIR /code
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt update && \
    apt upgrade -y

# Install dependencies
COPY requirements.txt /tmp/requirements.txt

RUN  --mount=type=cache,target=/root/.cache set -ex && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/

# Copy local project
COPY . /code/

# Expose port 8000
EXPOSE 8000

# Use gunicorn on port 8000
CMD ["gunicorn", "--bind", ":8000", "--workers", "2", "datacollection.wsgi"]
