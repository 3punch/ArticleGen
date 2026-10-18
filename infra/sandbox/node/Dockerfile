FROM node:20-slim

ENV npm_config_update_notifier=false \
    npm_config_fund=false \
    npm_config_audit=false \
    NODE_ENV=test

# Jest is installed at image-build time; test containers have no package registry access.
RUN npm install --global --no-audit --no-fund jest@29.7.0 \
    && mkdir -p /workspace \
    && chmod 1777 /workspace /tmp

WORKDIR /workspace
USER 65532:65532
