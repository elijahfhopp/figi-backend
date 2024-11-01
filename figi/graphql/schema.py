from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import strawberry

from cachetools.func import ttl_cache

from figi.config import CONFIG
from figi.db.models import FacesModel, ImagesModel
from peewee import Expression


@ttl_cache(ttl=10)
def load_image_by_id(id: int) -> ImagesModel:
    return ImagesModel.get_or_none(ImagesModel.id == id)


@ttl_cache(ttl=10)
def load_face_by_id(id: int) -> FacesModel:
    return FacesModel.get_or_none(FacesModel.id == id)


@strawberry.type
@dataclass
class BaseImage:
    id: strawberry.ID
    path: str
    filetype: str
    size: int

    @staticmethod
    def from_model(m: ImagesModel):
        if m is None:
            return None
        return BaseImage(m.id, m.path, m.filetype, m.size)


@strawberry.type
@dataclass
class Face:
    id: strawberry.ID
    sourceImage: BaseImage
    score: float
    x: int
    y: int
    top: int
    left: int
    embedding: List[float]

    @classmethod
    def from_model(cls, m: FacesModel):
        if m is None:
            return None
        image = load_image_by_id(m.source_image)
        source_image = BaseImage.from_model(image)
        embedding = m.embedding.tolist()
        return cls(m.id, source_image, m.score, m.x, m.y, m.top, m.left, embedding)


@strawberry.type
class Image(BaseImage):
    @strawberry.field
    def faces(self) -> List[Face]:
        faces = FacesModel.select().where(FacesModel.source_image == self.id)
        return [Face.from_model(face) for face in faces]


@strawberry.type
class ServerInfo:
    @strawberry.field
    def imageEntries(self) -> int:
        return ImagesModel.select().count()

    @strawberry.field
    def faceEntries(self) -> int:
        return FacesModel.select().count()


@strawberry.enum
class VectorSearchType(Enum):
    L1 = "l1"
    L2 = "l2"
    HAMMING = "hamming"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "max_inner_product"
    JACCARD = "jaccard_distance"


@strawberry.input
class SearchByFaceEmbedding:
    startImage: Optional[strawberry.ID] = None
    searchType: VectorSearchType
    embedding: List[float]
    threshold: float
    limit: Optional[int] = None


def _where_for_search_type(
    search_type: VectorSearchType, embedding: List[float]
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
        case d.COSINE:
            return f.cosine_distance(embedding)
        case d.MAX_INNER_PRODUCT:
            return f.max_inner_product(embedding)
        case _:
            raise ValueError(
                f'pgvector.peewee does not currently support search type "{search_type}".'
            )


def _resolve_limit(limit: int | None) -> int:
    max_limit = CONFIG["FIGI_MAX_VECTOR_SEARCH_LIMIT"]
    if limit is None:
        return max_limit
    if limit > max_limit:
        return max_limit
    else:
        return limit


@strawberry.type
class FigiQuery:

    @strawberry.field
    def serverInfo(self) -> ServerInfo:
        return ServerInfo()

    @strawberry.field
    def face(self, id: int) -> Optional[Face]:
        face = load_face_by_id(id)
        return Face.from_model(face)

    @strawberry.field
    def image(self, id: int) -> Optional[Image]:
        return Image.from_model(load_image_by_id(id))

    @strawberry.field
    def image_search(self, path: str) -> Optional[Image]:
        pass

    @strawberry.field
    def similarFaces(self, search: SearchByFaceEmbedding) -> List[Face]:
        # exp = expression
        search_exp = _where_for_search_type(search.searchType, search.embedding)
        search_exp = search_exp > search.threshold
        if search.startImage is not None:
            search_exp = search_exp & (FacesModel.source_image != search.startImage)
        limit = _resolve_limit(search.limit)
        faces = FacesModel.select().where(search_exp).limit(limit).execute()

        return [Face.from_model(face) for face in faces]
