import os
import re
from functools import lru_cache
from operator import add
from typing import (
    Annotated,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    TypedDict,
    Union,
)

import pandas as pd
import sqlalchemy as sa
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    AIMessage,
)
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_community.utilities.sql_database import SQLDatabase, truncate_word
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from sql_metadata import Parser
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType
import sqlparse
from sqlparse.sql import Parenthesis
from sqlparse.tokens import Keyword

from .schema import PostgresSQLInput

from . import UNSAFE_KEYWORDS

USE_LOCAL=os.getenv("USE_SQLITE")

if USE_LOCAL:
    DB_USER = os.getenv("LOCAL_POSTGRES_USER", "admin")
    DB_PASSWORD = os.getenv("LOCAL_POSTGRES_PASSWORD", "")
    DB_HOST = os.getenv("LOCAL_POSTGRES_HOST", "localhost")
    DB_PORT = int(os.getenv("LOCAL_POSTGRES_PORT", 5432))
    DB_NAME = os.getenv("LOCAL_POSTGRES_DB", "=iheardatadb")

else:
    DB_USER = os.getenv("POSTGRES_USER", "admin")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    DB_NAME = os.getenv("POSTGRES_DB", "=iheardatadb")


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    n_trials: int


class SQLDatabaseWrapper(SQLDatabase):
    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 2,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[dict] = None,
        view_support: bool = False,
        max_string_length: int = 90,
        lazy_table_reflection: bool = True,
        enable_cache: bool = True,
    ):
        super().__init__(
            engine=engine,
            schema=schema,
            metadata=metadata,
            ignore_tables=ignore_tables,
            include_tables=include_tables,
            sample_rows_in_table_info=sample_rows_in_table_info,
            indexes_in_table_info=indexes_in_table_info,
            custom_table_info=custom_table_info,
            view_support=view_support,
            max_string_length=max_string_length,
            lazy_table_reflection=lazy_table_reflection,
        )

        self._reflected_tables: Set[str] = set()
        self._enable_cache = enable_cache
        self._inspector = inspect(self._engine)

        self._cached_table_names = None
        self._cached_dialect = self._engine.dialect.name
        self._cached_sample_rows = {}

        self._dialect_schema_param = self._get_dialect_schema_param()

    def _get_dialect_schema_param(self) -> Any:
        if self._schema:
            if self._cached_dialect == "snowflake":
                return (self._schema,)
            elif self._cached_dialect == "bigquery":
                return (self._schema,)
            elif self._cached_dialect == "postgresql":
                return (self._schema,)
        return None

    @property
    def reflected_tables(self) -> Set[str]:
        return self._reflected_tables

    def reflect_tables(self, table_names: List[str]) -> None:
        """Reflect specific tables on demand"""
        to_reflect = [t for t in table_names if t not in self._reflected_tables]
        if not to_reflect:
            return

        self._metadata.reflect(
            views=self._view_support,
            bind=self._engine,
            only=to_reflect,
            schema=self._schema,
        )
        self._reflected_tables.update(to_reflect)

    @lru_cache(maxsize=20)
    def _get_cached_table(self, table_name: str) -> Table:
        table = (
            self._metadata.tables.get(f"{self._schema}.{table_name}")
            if self._schema
            else self._metadata.tables.get(table_name)
        )
        if table is None:
            raise ValueError(f"Table '{table_name}' not found in database")
        return table

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        all_table_names = self.get_usable_table_names()
        table_names = table_names or list(all_table_names)
        needs_reflection = [t for t in table_names if t not in self._reflected_tables]
        if needs_reflection:
            self.reflect_tables(needs_reflection)

        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        metadata_table_names = [tbl.name for tbl in self._metadata.sorted_tables]
        to_reflect = set(all_table_names) - set(metadata_table_names)
        if to_reflect:
            self._metadata.reflect(
                views=self._view_support,
                bind=self._engine,
                only=list(to_reflect),
                schema=self._schema,
            )

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            for k, v in table.columns.items():
                if type(v.type) is NullType:
                    table._columns.remove(v)

            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"
            has_extra_info = (
                self._indexes_in_table_info or self._sample_rows_in_table_info
            )
            if has_extra_info:
                table_info += "\n\n/*"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                table_info += f"\n{self._get_sample_rows(table)}\n"
            if has_extra_info:
                table_info += "*/"
            tables.append(table_info)
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str

    def _get_sample_rows(self, table: Table) -> str:
        if (
            self._enable_cache
            and table.name in self._reflected_tables
            and table.name in self._cached_sample_rows
        ):
            return self._cached_sample_rows.get(table.name, "")

        query = (
            select(sa.literal_column("*"))
            .select_from(table)
            .limit(self._sample_rows_in_table_info)
        )

        try:
            with self._engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchmany(self._sample_rows_in_table_info)
                sample_rows = "\n".join(["\t".join(map(str, row)) for row in rows])

                if self._enable_cache and sample_rows.strip():
                    self._cached_sample_rows[table.name] = sample_rows

                return sample_rows
        except ProgrammingError:
            return ""

    def run(
        self,
        command: Union[str, Executable],
        fetch: Literal["all", "one", "cursor", "stream"] = "all",
        include_columns: bool = False,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
        stream_results: bool = False,
    ) -> Union[List[Dict[str, Any]], Result[Any], Iterator[Dict[str, Any]]]:
        """Execute a SQL command and return the results.

        Args:
            command: SQL command to execute
            fetch: How to fetch results - "all", "one", "cursor", or "stream"
            include_columns: Whether to include column names in results
            parameters: Parameters to bind to the query
            execution_options: Options to pass to execution
            stream_results: Whether to stream results row by row (overrides fetch="stream")

        Returns:
            - List of rows if fetch="all" or fetch="one"
            - SQLAlchemy Result if fetch="cursor"
            - Iterator of rows if fetch="stream" or stream_results=True
        """
        if stream_results:
            fetch = "stream"

        result = self._execute(
            command, fetch, parameters=parameters, execution_options=execution_options
        )

        if fetch == "cursor":
            return result

        elif fetch == "stream":
            # Return an iterator that processes each row as it's fetched
            def row_iterator():
                for r in result:
                    row = {
                        column: truncate_word(value, length=self._max_string_length)
                        for column, value in r.items()
                    }

                    if not include_columns:
                        yield tuple(row.values())
                    else:
                        yield row

            return row_iterator()

        # Handle "all" and "one" cases
        res = [
            {
                column: truncate_word(value, length=self._max_string_length)
                for column, value in r.items()
            }
            for r in result
        ]

        if not include_columns:
            res = [tuple(row.values()) for row in res]  # type: ignore[misc]

        return res or []

    def _execute(
        self,
        command: Union[str, Executable],
        fetch: Literal["all", "one", "cursor", "stream"] = "all",
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[List[Dict[str, Any]], Result, Iterator[Dict[str, Any]]]:
        """
        Executes SQL command through underlying engine.

        If the statement returns no rows, an empty list is returned.
        """
        parameters = parameters or {}
        execution_options = execution_options or {}

        # If streaming is requested, add stream_results=True to execution options
        if fetch == "stream" and "stream_results" not in execution_options:
            execution_options = {**execution_options, "stream_results": True}

        connection = self._engine.connect()

        try:
            if self._schema is not None:
                if self.dialect == "snowflake":
                    connection.exec_driver_sql(
                        "ALTER SESSION SET search_path = %s",
                        (self._schema,),
                        execution_options=execution_options,
                    )
                elif self.dialect == "bigquery":
                    connection.exec_driver_sql(
                        "SET @@dataset_id=?",
                        (self._schema,),
                        execution_options=execution_options,
                    )
                elif self.dialect == "mssql":
                    pass
                elif self.dialect == "trino":
                    connection.exec_driver_sql(
                        "USE ?",
                        (self._schema,),
                        execution_options=execution_options,
                    )
                elif self.dialect == "duckdb":
                    connection.exec_driver_sql(
                        f"SET search_path TO {self._schema}",
                        execution_options=execution_options,
                    )
                elif self.dialect == "oracle":
                    connection.exec_driver_sql(
                        f"ALTER SESSION SET CURRENT_SCHEMA = {self._schema}",
                        execution_options=execution_options,
                    )
                elif self.dialect == "sqlany":
                    pass
                elif self.dialect == "postgresql":
                    connection.exec_driver_sql(
                        "SET search_path TO %s",
                        (self._schema,),
                        execution_options=execution_options,
                    )

            if isinstance(command, str):
                command = text(command)
            elif isinstance(command, Executable):
                pass
            else:
                raise TypeError(f"Query expression has unknown type: {type(command)}")

            # Execute the command with specified options
            cursor = connection.execute(
                command,
                parameters,
                execution_options=execution_options,
            )

            if cursor.returns_rows:
                if fetch == "all":
                    # Get all rows at once
                    result = [dict(row) for row in cursor.mappings().all()]
                    connection.close()
                    return result
                elif fetch == "one":
                    # Get just the first row
                    first_result = cursor.mappings().first()
                    result = [] if first_result is None else [dict(first_result)]
                    connection.close()
                    return result
                elif fetch == "cursor":
                    # Return the raw cursor
                    # Note: The user will need to manage the connection themselves
                    return cursor
                elif fetch == "stream":
                    # Return the cursor's mapping iterator
                    # We'll close the connection once the iterator is exhausted
                    return cursor.mappings()
                else:
                    connection.close()
                    raise ValueError(
                        "Fetch parameter must be one of: 'one', 'all', 'cursor', 'stream'"
                    )

            connection.close()
            return []

        except Exception as e:
            connection.close()
            raise e


class TextToSQLAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        include_tables: list[str],
        ai_answer: bool = False,
        sample_rows_in_table_info: int = 2,
        indexes_in_table_info: bool = False,
        lazy_table_reflection: bool = True,
        top_k: int = 100,
        default_offset: int = 0,
        max_retries: int = 1,
    ):
        db_user = DB_USER
        db_password = DB_PASSWORD
        db_host = DB_HOST
        db_port = DB_PORT
        db_name = DB_NAME

        if not (db_user and db_password and db_host and db_port and db_name):
            try:
                from django.conf import settings

                db = settings.DATABASES.get("default", {})
                if "postgresql" in db.get("ENGINE", ""):
                    db_user = db.get("USER") or db_user
                    db_password = db.get("PASSWORD") or db_password
                    db_host = db.get("HOST") or db_host
                    db_port = int(db.get("PORT") or db_port or 5432)
                    db_name = db.get("NAME") or db_name
            except Exception:
                pass

        assert (
            db_user and db_password and db_host and db_port and db_name
        ), "Missing database credentials"

        engine = create_engine(
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        self.llm = llm.bind_tools([PostgresSQLInput])
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.db = SQLDatabaseWrapper(
            engine,
            include_tables=include_tables,
            sample_rows_in_table_info=sample_rows_in_table_info,
            indexes_in_table_info=indexes_in_table_info,
            max_string_length=90,
            lazy_table_reflection=lazy_table_reflection,
            enable_cache=True,
        )
        self.output_parser = JsonOutputToolsParser(first_tool_only=True, return_id=True)
        self.max_retries = max_retries
        self.ai_answer = ai_answer
        try:
            self.default_offset = max(0, int(default_offset))
        except (TypeError, ValueError):
            self.default_offset = 0

    @staticmethod
    def _validate_sql_query(
        query: str,
        table_names: list[str] | None,
        max_limit: int = 50,
        default_offset: int = 0,
    ) -> str:
        ALLOWED_FUNCTIONS = {
            "jsonb_array_elements",
            "jsonb_array_elements_text",
            "jsonb_object_keys",
            "json_array_elements",
            "json_array_elements_text",
            "json_extract_path_text",
            "jsonb_extract_path_text",
        }

        patterns = [
            r";\s*\w+\s*=",
            r"\bor\s+1\s*=\s*1\b",
            # r"--",
            r"/\*.*?\*/",
            r"\b(exec|execute|xp_cmdshell)\b",
            r"\b(waitfor|delay)\b",
            r"\b(benchmark|sleep)\b.*?\(",
            r"\b(load_file|outfile|dumpfile)\b",
        ]

        if not query:
            raise ValueError("Operation not allowed: Empty query")

        if (
            any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)
            or query.count("'") % 2 != 0
            or query.count('"') % 2 != 0
        ):
            raise ValueError("Operation not allowed: Potential SQL injection detected")

        parsed = sqlparse.parse(query)
        if not parsed:
            raise ValueError("Operation not allowed: Invalid SQL syntax")

        def _enforce_limit(query: str, max_limit) -> str:
            limit_pattern = r"LIMIT\s+(\d+)(?:\s*(?:OFFSET\s+\d+)?)"
            match = re.search(limit_pattern, query, re.IGNORECASE)

            if match:
                current_limit = int(match.group(1))
                if current_limit > max_limit:
                    new_query = re.sub(
                        limit_pattern,
                        lambda m: f"LIMIT {max_limit}"
                        + (
                            m.group(0)[m.end(1) :] if m.end(1) < len(m.group(0)) else ""
                        ),
                        query,
                        flags=re.IGNORECASE,
                    )
                    return new_query
            return query

        def _apply_limit_offset(query: str, max_limit: int, default_offset: int) -> str:
            q = query.rstrip().rstrip(";")
            has_limit = re.search(r"\bLIMIT\b", q, re.IGNORECASE)
            if not has_limit:
                q = f"{q} LIMIT {max_limit}"
            else:
                q = _enforce_limit(q, max_limit)

            if default_offset and default_offset > 0:
                if re.search(r"\bOFFSET\b", q, re.IGNORECASE):
                    q = re.sub(
                        r"\bOFFSET\s+\d+\b",
                        f"OFFSET {default_offset}",
                        q,
                        flags=re.IGNORECASE,
                    )
                else:
                    q = f"{q} OFFSET {default_offset}"
            return q

        def validate_statement(statement: sqlparse.sql.Statement):
            # first_token = statement.token_first(skip_cm=True)
            # if not first_token or first_token.value.upper() not in ("SELECT", "WITH"):
            #     raise ValueError("Only SELECT operations are allowed")

            parser = Parser(statement.value)
            actual_tables = set()
            for tbl in parser.tables:
                if "." not in tbl and tbl.lower() not in ALLOWED_FUNCTIONS:
                    actual_tables.add(tbl)

            if table_names is not None:
                if not actual_tables.issubset(set(table_names)):
                    unauthorized = actual_tables - set(table_names)
                    if unauthorized:
                        raise ValueError(
                            f"Unauthorized tables referenced: {unauthorized}"
                        )

            for token in statement.flatten():
                if token.ttype in Keyword:
                    if (
                        token.value.upper() in UNSAFE_KEYWORDS
                        and token.value.lower() not in ALLOWED_FUNCTIONS
                    ):
                        raise ValueError(f"Unsafe keyword used: {token.value}")

            for token in statement.tokens:
                if isinstance(token, Parenthesis):
                    inner = token.value[1:-1].strip()
                    if inner:
                        subparsed = sqlparse.parse(inner)
                        if subparsed:
                            validate_statement(subparsed[0])

        for st in parsed:
            if st.get_type() != "UNKNOWN":
                validate_statement(st)

        query = _apply_limit_offset(query, max_limit, default_offset)
        return " ".join(query.split())

    def _filter_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        parser = JsonOutputToolsParser(first_tool_only=True, return_id=True)
        filtered_messages = []
        i = 0
        while i < len(messages):
            if isinstance(messages[i], AIMessage) and parser.invoke(messages[i]):
                if i == len(messages) - 1 or not isinstance(
                    messages[i + 1], ToolMessage
                ):
                    i += 1
                    continue
            filtered_messages.append(messages[i])
            i += 1
        return filtered_messages

    def call_llm(self, state: AgentState):
        system_msg = SystemMessage(
            content=self.system_prompt.format(
                top_k=self.top_k, table_info=self.db.get_table_info()
            )
        )
        messages = [system_msg] + state["messages"]
        # messages = trim_messages(
        #     messages,
        #     max_tokens=6000,
        #     include_system=True,
        #     token_counter=ChatOpenAI(model="gpt-4o"),
        # )
        messages = self._filter_messages(messages)  # type: ignore
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def execute_tool(self, state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1]
        content = last_msg.content
        if isinstance(content, list):
            content = content[0]
            if isinstance(content, dict):
                content = content.get("text", "")
        query = self.extract_sql(content)
        tool_call = None

        if not query:
            tool_call = self.output_parser.invoke(last_msg)
            if tool_call:
                query = tool_call["args"]["query"]
        n_trials = state.get("n_trials", 0) + 1

        if not query and not tool_call:
            return {
                "messages": [
                    HumanMessage(
                        content="Error: Query failed! Please rewrite your query and try again."
                    )
                ],
                "n_trials": n_trials,
            }

        try:
            if not isinstance(query, str):
                error_msg = "Error: Query must be a properly formatted SQL query"
                if tool_call is not None:
                    tool_name = tool_call["type"]
                    tool_id = tool_call["id"]
                    message_obj = ToolMessage(
                        content=error_msg,
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                else:
                    message_obj = HumanMessage(content=error_msg)
                return {
                    "messages": [message_obj],
                    "n_trials": n_trials,
                }
            query = self._validate_sql_query(
                query,
                self.db.get_usable_table_names(),
                max_limit=self.top_k,
                default_offset=self.default_offset,
            )
            res = self.db.run(query, include_columns=True)  # type: ignore

            if not res:
                res: str = (
                    "Error: No results found. Verify that your query is correct and try again."
                )
            if tool_call is not None:
                tool_name = tool_call["type"]
                tool_id = tool_call["id"]
                if isinstance(res, list):
                    df = pd.DataFrame(res)
                    cat_col = df.select_dtypes(exclude="number").columns
                    df[cat_col] = df[cat_col].astype(str)
                    res = df.to_json(orient="records", index=False)

                message_obj = ToolMessage(
                    content=res,
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            else:
                message_obj = HumanMessage(content=res)
            return {
                "messages": [message_obj],
                "n_trials": n_trials,
            }

        except (SQLAlchemyError, ValueError) as e:
            error_msg = "Error " + str(e)
            if tool_call is not None:
                tool_name = tool_call["type"]
                tool_id = tool_call["id"]
                message_obj = ToolMessage(
                    content=error_msg,
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            else:
                message_obj = HumanMessage(content=error_msg)
            return {
                "messages": [message_obj],
                "n_trials": n_trials,
            }

    def extract_sql(self, llm_response: str) -> str | None:
        sqls = re.findall(
            r"\bCREATE\s+TABLE\b.*?\bAS\b.*?;", llm_response, re.DOTALL | re.IGNORECASE
        )
        if sqls:
            sql = sqls[-1]
            return sql

        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            return sql

        sqls = re.findall(r"\bSELECT\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            return sql

        sqls = re.findall(
            r"```sql\s*\n(.*?)```", llm_response, re.DOTALL | re.IGNORECASE
        )
        if sqls:
            sql = sqls[-1].strip()
            return sql

        sqls = re.findall(r"```(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1].strip()
            return sql

        return None

    def should_continue(self, state: AgentState):
        last_msg = state["messages"][-1]
        tool_calls = self.output_parser.invoke(last_msg)
        if tool_calls or self.extract_sql(last_msg.content):  # type: ignore
            return "tool"
        return "__end__"

    def retry_error(self, state: AgentState):
        tool_message = state["messages"][-1].content
        if (
            "error" in str(tool_message).lower()
            and state["n_trials"] < self.max_retries
        ) or self.ai_answer:
            return "llm"
        return "__end__"

    def compile_workflow(self, checkpointer: BaseCheckpointSaver | None = None):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("tool", self.execute_tool)

        graph.add_conditional_edges(
            "llm", self.should_continue, {"tool": "tool", "__end__": "__end__"}
        )
        graph.add_conditional_edges(
            "tool", self.retry_error, {"llm": "llm", "__end__": "__end__"}
        )
        graph.set_entry_point("llm")
        return graph.compile(checkpointer if self.ai_answer else None)
