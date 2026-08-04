FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY echoguide_guard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY echoguide_guard /app/echoguide_guard

RUN useradd --create-home --uid 10001 echoguide_guard && \
    mkdir -p /data && chown -R echoguide_guard:echoguide_guard /app /data

USER echoguide_guard
EXPOSE 8200

CMD ["uvicorn", "echoguide_guard.app:app", "--host", "0.0.0.0", "--port", "8200"]

