FROM apache/superset

USER root

RUN pip install psycopg2-binary
RUN pip install elasticsearch-dbapi

USER superset

RUN superset fab create-admin --username admin \
    --firstname Khairi \
    --lastname Abidi \
    --email admin@superset.com \
    --password admin

RUN  superset db upgrade

RUN superset init
