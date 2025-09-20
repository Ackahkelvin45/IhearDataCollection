from pydantic import BaseModel, Field


class PostgresSQLInput(BaseModel):
    query: str = Field(
        ...,
        description="""Execute a SQL query against the database and get back the result..
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.""",
    )
