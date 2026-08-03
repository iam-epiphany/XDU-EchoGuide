FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY echoguard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY echoguard /app/echoguard

RUN useradd --create-home --uid 10001 echoguard && \
    mkdir -p /data && chown -R echoguard:echoguard /app /data

USER echoguard
EXPOSE 8200

CMD ["uvicorn", "echoguard.app:app", "--host", "0.0.0.0", "--port", "8200"]

