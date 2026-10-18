FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /opt/projectforge && chmod 755 /opt/projectforge /tmp
COPY packages/sandbox/egress_proxy.py /opt/projectforge/egress_proxy.py

EXPOSE 3128
USER 65532:65532
ENTRYPOINT ["python", "/opt/projectforge/egress_proxy.py"]
