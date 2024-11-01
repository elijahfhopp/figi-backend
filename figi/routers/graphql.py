from strawberry import Schema
from strawberry.fastapi import GraphQLRouter

from figi.graphql.schema import FigiQuery


schema = Schema(query=FigiQuery)
graphql: GraphQLRouter = GraphQLRouter(schema=schema)
