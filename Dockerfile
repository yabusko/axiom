# One image, every distributed worker role -- `command:` in docker-compose.yml
# picks the role (AXIOM_DISTRIBUTED_TASK.md: "all built from one image with a
# different command"). Not used by the API/web dev path, which still runs
# directly on the host per the README.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/ axiom/
COPY db/ db/

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "axiom.workers"]
