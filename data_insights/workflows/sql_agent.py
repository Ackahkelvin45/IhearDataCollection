import os
import re
import json
import logging
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
    event,
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

logger = logging.getLogger(__name__)

# Default statement_timeout (seconds) for the NL->SQL agent engine when settings
# are unavailable. Overridden by AI_INSIGHT["DATABASE"]["STATEMENT_TIMEOUT_SECONDS"].
DEFAULT_SQL_STATEMENT_TIMEOUT_SECONDS = 15

# Columns whose raw values must never reach the LLM (Phase 3). These hold raw
# filesystem paths / object-storage keys (e.g. DO Spaces) for the audio files in
# the allowed data tables (data_noisedataset / data_recording /
# data_cleanspeechdataset). They are stripped from BOTH the CREATE TABLE schema
# and the sample-row preview injected into {table_info}, so the agent never sees
# the storage location. Org-wide aggregate analytics is unaffected — only this
# storage-path column is masked, not which rows are aggregated.
SENSITIVE_COLUMN_NAMES = frozenset({"audio"})


def _resolve_readonly_db_credentials(db_user: str, db_password: str) -> tuple[str, str]:
    """Return dedicated read-only credentials for the NL->SQL agent engine.

    Prefers AI_INSIGHT["DATABASE"]["READONLY_USER"/"READONLY_PASSWORD"] (env:
    AI_INSIGHT_DB_READONLY_USER / AI_INSIGHT_DB_READONLY_PASSWORD). If unset,
    falls back to the read-WRITE app credentials with a loud warning — the
    engine is still forced read-only at the session/transaction level below, but
    a dedicated least-privilege role is defense-in-depth and strongly preferred.

    To create a least-privilege read-only role in Postgres, run (as superuser):

        CREATE ROLE ai_insight_ro LOGIN PASSWORD '<strong-password>';
        GRANT CONNECT ON DATABASE iheardatadb TO ai_insight_ro;
        GRANT USAGE ON SCHEMA public TO ai_insight_ro;
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO ai_insight_ro;
        -- Auto-grant SELECT on future tables created by the app owner:
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT ON TABLES TO ai_insight_ro;
        -- Optionally make the role itself read-only at the role level:
        ALTER ROLE ai_insight_ro SET default_transaction_read_only = on;

    Then set AI_INSIGHT_DB_READONLY_USER / AI_INSIGHT_DB_READONLY_PASSWORD.
    """
    ro_user = ro_password = ""
    try:
        from django.conf import settings

        db_cfg = getattr(settings, "AI_INSIGHT", {}).get("DATABASE", {})
        ro_user = db_cfg.get("READONLY_USER") or ""
        ro_password = db_cfg.get("READONLY_PASSWORD") or ""
    except Exception:
        pass

    # Allow env fallback even if settings are unavailable.
    ro_user = ro_user or os.getenv("AI_INSIGHT_DB_READONLY_USER", "")
    ro_password = ro_password or os.getenv("AI_INSIGHT_DB_READONLY_PASSWORD", "")

    if ro_user and ro_password:
        return ro_user, ro_password

    logger.warning(
        "NL->SQL agent: no dedicated read-only DB credentials configured "
        "(AI_INSIGHT_DB_READONLY_USER/AI_INSIGHT_DB_READONLY_PASSWORD); falling "
        "back to read-WRITE app credentials. The engine is still forced "
        "read-only (default_transaction_read_only=on) at the session level, but "
        "a dedicated least-privilege role is strongly recommended."
    )
    return db_user, db_password


def _get_statement_timeout_seconds() -> int:
    """Statement timeout (seconds) for the NL->SQL agent engine, from settings."""
    try:
        from django.conf import settings

        db_cfg = getattr(settings, "AI_INSIGHT", {}).get("DATABASE", {})
        return int(
            db_cfg.get(
                "STATEMENT_TIMEOUT_SECONDS", DEFAULT_SQL_STATEMENT_TIMEOUT_SECONDS
            )
        )
    except Exception:
        return int(
            os.getenv(
                "AI_INSIGHT_SQL_TIMEOUT_SECONDS",
                DEFAULT_SQL_STATEMENT_TIMEOUT_SECONDS,
            )
        )


def create_readonly_engine(
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: int,
    db_name: str,
) -> Engine:
    """Build a SQLAlchemy engine for the NL->SQL agent that is strictly read-only.

    On EVERY pooled connection (a "connect" event, so it also covers schema
    reflection and sample-row reads), this runs:
        SET default_transaction_read_only = on   -> Postgres rejects any
            INSERT/UPDATE/DELETE/DDL, even if the regex/SELECT-only guard is
            bypassed.
        SET statement_timeout = <ms>             -> runaway / cartesian queries
            are killed server-side.
    Org-wide READ analytics is intentional, so this engine may SELECT across all
    collectors — it is only writes that are blocked. The Django ORM connection
    and the LangGraph PostgresSaver checkpointer use SEPARATE connections and
    remain writable; this only affects the SQL-agent engine.
    """
    ro_user, ro_password = _resolve_readonly_db_credentials(db_user, db_password)
    timeout_ms = max(1, _get_statement_timeout_seconds()) * 1000

    engine = create_engine(
        f"postgresql://{ro_user}:{ro_password}@{db_host}:{db_port}/{db_name}"
    )

    @event.listens_for(engine, "connect")
    def _set_readonly_session(dbapi_connection, connection_record):
        # Enforce read-only + statement timeout at the session level for every
        # physical connection in the pool.
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SET default_transaction_read_only = on")
            cursor.execute(f"SET statement_timeout = {timeout_ms}")
        finally:
            cursor.close()

    return engine


# Use Docker DB ('db') when available, otherwise localhost for local dev
postgres_host = os.getenv("POSTGRES_HOST")
if postgres_host == "db":
    # We're in Docker, use the Docker DB
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    DB_HOST = "db"
    DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    DB_NAME = os.getenv("POSTGRES_DB", "iheardatadb")
else:
    # Local development - use local DB config
    USE_LOCAL = os.getenv("USE_SQLITE")
    if USE_LOCAL:
        DB_USER = os.getenv("LOCAL_POSTGRES_USER", "kelvin")
        DB_PASSWORD = os.getenv("LOCAL_POSTGRES_PASSWORD", "kelvin")
        DB_HOST = os.getenv("LOCAL_POSTGRES_HOST", "localhost")
        DB_PORT = int(os.getenv("LOCAL_POSTGRES_PORT", 5432))
        DB_NAME = os.getenv("LOCAL_POSTGRES_DB", "datacollection")
    else:
        DB_USER = os.getenv("POSTGRES_USER", "postgres")
        DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
        DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
        DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
        DB_NAME = os.getenv("POSTGRES_DB", "iheardatadb")


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

            for k, v in list(table.columns.items()):
                # Drop NullType columns (LangChain default) and any sensitive
                # column (e.g. `audio`, which holds a raw storage path / object
                # key) so it never reaches the schema OR the sample rows below.
                if type(v.type) is NullType or k in SENSITIVE_COLUMN_NAMES:
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

        # Select explicit columns (not `*`) so sensitive columns already
        # removed from the table metadata in get_table_info() — e.g. `audio`,
        # the raw storage path — are NOT pulled back by the sample-row preview.
        sample_columns = [
            col for col in table.columns if col.name not in SENSITIVE_COLUMN_NAMES
        ]
        if not sample_columns:
            return ""
        query = (
            select(*sample_columns)
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
        max_string_length: int = 60,
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

        # Strictly read-only engine: writes/DDL are rejected by Postgres and
        # runaway queries are killed by statement_timeout, independent of the
        # regex/SELECT-only guard (P2-1). Uses dedicated read-only credentials
        # when configured. Does NOT affect the Django ORM or the LangGraph
        # checkpointer, which use separate (writable) connections.
        engine = create_readonly_engine(db_user, db_password, db_host, db_port, db_name)

        self.llm = llm.bind_tools([PostgresSQLInput])
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.db = SQLDatabaseWrapper(
            engine,
            include_tables=include_tables,
            sample_rows_in_table_info=sample_rows_in_table_info,
            indexes_in_table_info=indexes_in_table_info,
            max_string_length=max_string_length,
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

        # NOTE: This regex/keyword guard is defense-in-depth ONLY. Regex cannot
        # reliably parse SQL (comments, string literals, nesting, dialect quirks
        # all create bypasses), so the PRIMARY defenses are the read-only engine
        # (default_transaction_read_only=on), the SELECT/WITH-only statement
        # check below, and statement_timeout (see P2-1/P2-2). This layer just
        # catches obvious abuse early and bounds result size.
        patterns = [
            r";\s*\w+\s*=",
            r"\bor\s+1\s*=\s*1\b",
            r"/\*.*?\*/",
            # Time-based / DoS functions (across dialects).
            r"\b(pg_sleep|pg_sleep_for|pg_sleep_until)\b",
            r"\b(pg_terminate_backend|pg_cancel_backend)\b",
            r"\b(exec|execute|xp_cmdshell)\b",
            r"\b(waitfor|delay)\b",
            r"\b(benchmark|sleep)\b.*?\(",
            r"\b(load_file|outfile|dumpfile)\b",
            # Postgres COPY (statement-initial only, to avoid flagging a column
            # named "copy") and large object / file access.
            r"^\s*copy\b",
            r"\b(lo_import|lo_export)\b",
        ]

        if not query:
            raise ValueError("Operation not allowed: Empty query")

        # Strip SQL comments BEFORE any validation so comment-based payloads
        # (e.g. "-- ..." or "/* ... */") cannot hide writes/DoS from the checks
        # below. Replaces the previously commented-out "--" regex rule.
        query = sqlparse.format(query, strip_comments=True).strip()
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

        # Reject multi-statement input (e.g. "SELECT ...; DROP ...") — only a
        # single SELECT/WITH statement is allowed.
        meaningful = [st for st in parsed if st.token_first(skip_cm=True) is not None]
        if len(meaningful) > 1:
            raise ValueError("Operation not allowed: Multiple statements")

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

        def validate_statement(
            statement: sqlparse.sql.Statement, top_level: bool = False
        ):
            # SELECT-only enforcement: the top-level statement must be a SELECT
            # or a WITH (CTE) — reject INSERT/UPDATE/DELETE/DDL etc. This is NOT
            # applied to recursively-validated parenthesis contents, since those
            # may legitimately be value lists (e.g. "IN (1, 2, 3)") or scalar
            # subqueries; writes hidden inside parentheses are still caught by
            # the UNSAFE_KEYWORDS scan below.
            if top_level:
                first_token = statement.token_first(skip_cm=True)
                if not first_token or first_token.value.upper() not in (
                    "SELECT",
                    "WITH",
                ):
                    raise ValueError("Only SELECT operations are allowed")

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

        # Validate every top-level statement, including "UNKNOWN" ones: an
        # unclassifiable statement is not a SELECT/WITH and must be rejected
        # rather than silently skipped (which was a bypass).
        for st in parsed:
            if st.token_first(skip_cm=True) is None:
                # Whitespace-only fragment (e.g. trailing tokens) — nothing to do.
                continue
            validate_statement(st, top_level=True)

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
                    AIMessage(
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
                    message_obj = AIMessage(content=error_msg)
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
                    rows = res
                    columns = (
                        list(rows[0].keys())
                        if rows and isinstance(rows[0], dict)
                        else []
                    )
                    payload = {
                        "rows": rows,
                        "columns": columns,
                        "row_count": len(rows),
                        "limit": self.top_k,
                        "offset": self.default_offset,
                        "has_more": len(rows) == self.top_k,
                        "query_kind": "sql",
                        "query_sql": query,
                    }
                    res = json.dumps(payload)

                message_obj = ToolMessage(
                    content=res,
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            else:
                message_obj = AIMessage(content=res)
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
                message_obj = AIMessage(content=error_msg)
            return {
                "messages": [message_obj],
                "n_trials": n_trials,
            }

    def extract_sql(self, llm_response: str) -> str | None:
        # NOTE: the CREATE TABLE AS extraction path was removed — the agent
        # engine is now strictly read-only and only SELECT/WITH statements pass
        # validation, so extracting DDL would only produce a guaranteed failure.
        # Normal analytics use plain SELECT / CTEs, which are handled below.
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
        if self.ai_answer and state.get("n_trials", 0) < self.max_retries:
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
