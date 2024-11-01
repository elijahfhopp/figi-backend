from figi.db.database import get_database
from peewee import (
    AutoField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
)
from pgvector.peewee import VectorField


class BaseModel(Model):
    class Meta:
        database = get_database()


DB_CONNECTION = BaseModel._meta.database


class ImagesModel(BaseModel):
    class Meta:
        db_table = "images"

    id = AutoField(index=True)
    path = TextField()
    filetype = TextField()
    size = IntegerField()


class FacesModel(BaseModel):
    class Meta:
        db_table = "faces"

    id = AutoField(index=True)
    source_image = ForeignKeyField(ImagesModel, backref="faces")
    score = FloatField()
    x = IntegerField()
    y = IntegerField()
    width = IntegerField()
    height = IntegerField()
    embedding = VectorField(dimensions=128)
