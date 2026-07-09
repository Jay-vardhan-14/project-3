# SentinelML Setup Guide

## Local Docker Startup

Default startup:

```bash
docker-compose up -d db redis mlflow
```

This exposes:

- PostgreSQL on `localhost:5432`
- Redis on `localhost:6379`
- MLflow on `localhost:5000`

## Port Conflict Troubleshooting

If another local project already owns PostgreSQL `5432` or Redis `6379`, use the checked-in verification override:

```bash
docker-compose -f docker-compose.yml -f docker-compose.verify.yml up -d db redis mlflow
```

The override only changes host port mappings:

- PostgreSQL: `localhost:15432 -> container:5432`
- Redis: `localhost:16379 -> container:6379`
- MLflow remains `localhost:5000`

Service-to-service URLs inside Docker stay unchanged:

- `db:5432`
- `redis:6379`
- `mlflow:5000`

This means MLflow, Airflow, and serving code can continue using the same internal connection strings while local host conflicts are avoided.
