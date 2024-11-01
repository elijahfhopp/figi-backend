from typing import List

from gql import Client, gql
from gql.transport.httpx import HTTPXTransport

# Select your transport with a defined url endpoint
transport = HTTPXTransport(url="http://127.0.0.1:8000/graphql")

# Create a GraphQL client using the defined transport
client = Client(transport=transport, fetch_schema_from_transport=True)
# session = client.connect_sync()


# Provide a GraphQL query
query = gql(
    """
{
image(id: 178) {
path
id
faces {
score
embedding
}
}
}
"""
)

embedding: List[float] = [
    # You don't get the dimensions of my pretty face, :).
]


def face_search(face: List[float]):
    return gql(
        """
        {
          similarFaces(search: {searchType: L2, embedding: %s, threshold: 0.363}) {
            id
            embedding
            sourceImage {
              id
              path
            }
          }
        }
        """
        % (repr(face))
    )


search = face_search(embedding)
# result = client.execute(query)["image"]
result = client.execute(search)
print(result)
