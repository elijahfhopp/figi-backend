from db.database import get_database
from peewee import (
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    PrimaryKeyField,
    TextField,
)
from pgvector.peewee import VectorField


class BaseModel(Model):
    class Meta:
        database = get_database()


class ImagesModel(BaseModel):
    id = PrimaryKeyField(index=True)
    path = TextField()
    filetype = TextField()
    size = IntegerField()


class FacesModel(BaseModel):
    id = PrimaryKeyField(index=True)
    source_image = ForeignKeyField(ImagesModel, backref="faces")
    score = FloatField()
    x = IntegerField()
    y = IntegerField()
    left = IntegerField()
    top = IntegerField()
    embedding = VectorField(dimensions=128)
