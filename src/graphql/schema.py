from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import strawberry

from config import CONFIG
from db.models import FacesModel, ImagesModel
from peewee import Expression
from strawberry import UNSET


@strawberry.type
@dataclass
class Image:
    id: strawberry.ID
    path: str
    filetype: str
    size: int

    @classmethod
    def resolve_reference(cls, id: strawberry.ID) -> Image:
        model = ImagesModel.where(ImagesModel.id == id).get_or_none()
        return Image.from_model(model)

    @classmethod
    def from_model(cls, m: ImagesModel):
        if m is None:
            return None
        return cls(m.id, m.path, m.filetype, m.size)


# class FacesModel(BaseModel):
#     id = PrimaryKeyField(index=True)
#     source_image = ForeignKeyField(ImagesModel, backref="faces")
#     score = FloatField()
#     x = IntegerField()
#     y = IntegerField()
#     top = IntegerField()
#     left = IntegerField()
#     embedding = VectorField(dimensions=128)


@strawberry.type
@dataclass
class Face:
    id: int
    path: str
    filetype: str
    size: int

    @classmethod
    def from_model(cls, m: ImagesModel):
        if m is None:
            return None
        return cls(m.id, m.path, m.filetype, m.size)


@strawberry.enum
class VectorSearchType(Enum):
    L1 = "l1"
    L2 = "l2"
    HAMMING = "hamming"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "max_inner_product"
    JACCARD = "jaccard_distance"


# NOTE: one_of is technically not spec, but is supported by strawberry
# and seems to be more ideal than other options. However, I'm a GraphQL noob.
@strawberry.input(one_of=True)
class SearchImageBy:
    id: int | None = UNSET
    path: int | None = UNSET


@strawberry.input
class SearchByFaceEmbedding:
    search_type: VectorSearchType
    embedding: List[float]
    max_distance: float
    limit: Optional[int]


@strawberry.type
class Query:
    def where_for_search_type(
        self, search_type: VectorSearchType, embedding: List[float]
    ) -> Expression:
        # d = Distance
        d = VectorSearchType
        # f = model Field
        f = FacesModel.embedding
        match search_type:
            case d.L1:
                return f.l1_distance(embedding)
            case d.L2:
                return f.l2_distance(embedding)
            case d.MAX_INNER_PRODUCT:
                return f.max_inner_product(embedding)
            case _:
                raise ValueError(
                    f'pgvector.peewee does not currently support search type "{search_type}".'
                )

    def resolve_limit(self, limit: int | None) -> int:
        max_limit = CONFIG["MAX_VECTOR_SEARCH_LIMIT"]
        if limit is None:
            return max_limit
        if limit > max_limit:
            return max_limit
        else:
            return limit

    @strawberry.field
    def image(self, search: SearchImageBy) -> Image:
        if search.id is not UNSET:
            return Image.resolve_reference(search.id)
        elif search.path is not UNSET:
            model = ImagesModel.where(ImagesModel.path == search.path).get_or_none()
            return Image.from_model(model)
        else:
            raise ValueError('"image" query can only handle id & path searches.')

    @strawberry.field
    def similar_faces(self, search: SearchByFaceEmbedding) -> List[Face]:
        select_expression = self.where_for_search_type(
            search.search_type, search.embedding
        )
        limit = self.resolve_limit(search.limit)
        faces = (
            FacesModel.select()
            .where(select_expression < search.max_distance)
            .limit(limit)
            .execute()
        )

        return []
