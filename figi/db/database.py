import os

from peewee import PostgresqlDatabase


def get_database():
    host = os.environ.get("FIGI_DB_HOST", "127.0.0.1")
    db = PostgresqlDatabase(
        "figi", user="figi", password="testpwd", host=host, port=5432
    )
    db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector")
    return db
