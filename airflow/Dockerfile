FROM apache/airflow:2.9.3-python3.11

COPY --chown=airflow:root airflow/requirements-airflow.txt /requirements-airflow.txt

RUN pip install --default-timeout=120 --no-cache-dir -r /requirements-airflow.txt
