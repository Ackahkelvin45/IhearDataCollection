import uuid
from typing import Dict, List, Any, Literal, Optional
from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from loguru import logger

from ai_insight.workflow.prompt import SQL_SYSTEM_TEMPLATE
from core.models import (
    Customer,
    DormantAccount,
    BankAccount,
    CustomerEmail,
    CustomerTransaction,
    CustomerSegmentation,
)


from .sql_agent import TextToSQLAgent
from core.tasks import send_customer_email_with_attachments, send_templated_bulk_email
from ai_insight.models import QueryCacheModel


AI_CONFIG = getattr(settings, "AI_INSIGHT", {})
DB_CONFIG = AI_CONFIG.get("DATABASE", {})
AGENT_CONFIG = AI_CONFIG.get("AGENT", {})
SECURITY_CONFIG = AI_CONFIG.get("SECURITY", {})

DB_URI = (
    f"postgresql://{DB_CONFIG.get('USER', 'admin')}:"
    f"{DB_CONFIG.get('PASSWORD', 'localhost')}@"
    f"{DB_CONFIG.get('HOST', 'db')}:"
    f"{DB_CONFIG.get('PORT', 5432)}/"
    f"{DB_CONFIG.get('NAME', 'brainbox-crm')}"
)


class CustomerSearchInput(BaseModel):
    """Input for customer search tool"""

    filter_criteria: Optional[
        Dict[
            Literal[
                "name",
                "email",
                "phone",
                "customer_type",
                "sector",
                "industry",
                "gender",
            ],
            Any,
        ]
    ] = Field(default_factory=dict, description="Filter criteria for customers")
    limit: int = Field(default=50, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")
    include_behavior: bool = Field(
        default=False, description="Include customer behavior data"
    )


class DormantAccountSearchInput(BaseModel):
    days_inactive: Optional[int] = Field(
        default=None, description="Minimum days inactive"
    )
    min_balance: Optional[float] = Field(
        default=None, description="Minimum account balance"
    )
    max_balance: Optional[float] = Field(
        default=None, description="Maximum account balance"
    )
    limit: int = Field(default=50, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")


class EmailSendInput(BaseModel):
    customer_id: int = Field(description="Customer ID to send email to")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body content")
    to_email: Optional[str] = Field(
        default=None, description="Override recipient email"
    )


class BulkEmailInput(BaseModel):
    """Input for sending bulk emails"""

    query_id: str = Field(description="Query handle ID for recipient list")
    subject: str = Field(description="Email subject template")
    body: str = Field(description="Email body template with placeholders")
    send_immediately: bool = Field(
        default=False, description="Send immediately or queue"
    )


class CustomerDetailInput(BaseModel):
    """Input for getting detailed customer information"""

    customer_id: int = Field(description="Customer ID to retrieve details for")
    include_accounts: bool = Field(default=True, description="Include bank accounts")
    include_transactions: bool = Field(
        default=False, description="Include recent transactions"
    )
    include_behavior: bool = Field(
        default=True, description="Include behavior analysis"
    )


# Tool implementations
class CustomerSearchTool(BaseTool):
    name: str = "search_customers"
    description: str = """Search for customers based on criteria. Returns a query handle for large result sets.
    Use this to find customers by name, email, phone, customer type, etc."""

    def _run(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        include_behavior: bool = False,
    ) -> Dict[str, Any]:
        try:
            filter_criteria = filter_criteria or {}

            # Build queryset
            queryset = Customer.objects.all()

            # Apply filters
            if "name" in filter_criteria:
                queryset = queryset.filter(
                    Q(full_name__icontains=filter_criteria["name"])
                    | Q(short_name__icontains=filter_criteria["name"])
                )

            if "email" in filter_criteria:
                queryset = queryset.filter(
                    customer_email__icontains=filter_criteria["email"]
                )

            if "phone" in filter_criteria:
                queryset = queryset.filter(
                    mobile_number__icontains=filter_criteria["phone"]
                )

            if "customer_type" in filter_criteria:
                queryset = queryset.filter(
                    customer_type=filter_criteria["customer_type"]
                )

            if "sector" in filter_criteria:
                queryset = queryset.filter(sector__icontains=filter_criteria["sector"])

            if "industry" in filter_criteria:
                queryset = queryset.filter(
                    industry__icontains=filter_criteria["industry"]
                )

            # Get total count
            total_count = queryset.count()

            # If large result set, create query handle
            if total_count > 100:
                query_id = f"customers_{uuid.uuid4().hex[:12]}"

                # Create cache entry
                cache_entry = QueryCacheModel.objects.create(
                    query_id=query_id,
                    query_type="customer_search",
                    query_sql=str(queryset.query),
                    result_count=total_count,
                    created_by_id=1,  # TODO: Get from context
                    metadata={
                        "filter_criteria": filter_criteria,
                        "include_behavior": include_behavior,
                    },
                )

                # Return handle with sample
                sample_customers = list(
                    queryset[:5].values(
                        "id",
                        "full_name",
                        "customer_email",
                        "mobile_number",
                        "customer_type",
                        "sector",
                        "industry",
                    )
                )

                return {
                    "query_id": query_id,
                    "total_count": total_count,
                    "sample_data": sample_customers,
                    "message": f'Found {total_count} customers. Use query_id "{query_id}" for bulk operations.',
                }

            else:
                # Return data directly for small result sets
                customers = queryset[offset : offset + limit]

                result_data = []
                for customer in customers:
                    customer_data = {
                        "id": customer.id,
                        "full_name": customer.full_name,
                        "customer_number": customer.customer_number,
                        "email": customer.customer_email,
                        "phone": customer.mobile_number,
                        "customer_type": customer.customer_type,
                        "sector": customer.sector,
                        "industry": customer.industry,
                        "relationship_officer": customer.relationship_officer,
                    }

                    if include_behavior and hasattr(customer, "behaviors"):
                        behavior = customer.behaviors
                        customer_data["behavior"] = {
                            "avg_balance": float(behavior.average_account_balance),
                            "transaction_count": behavior.number_of_transactions,
                            "min_transaction": float(behavior.min_transaction_amount),
                            "max_transaction": float(behavior.max_transaction_amount),
                        }

                    result_data.append(customer_data)

                return {
                    "customers": result_data,
                    "total_count": total_count,
                    "returned_count": len(result_data),
                }

        except Exception as e:
            logger.error(f"Error in customer search: {str(e)}")
            return {"error": f"Customer search failed: {str(e)}"}


class DormantAccountSearchTool(BaseTool):
    name: str = "search_dormant_accounts"
    description: str = """Search for dormant accounts based on criteria. Returns query handle for large datasets.
    Use this to find accounts by inactivity period, balance range, etc."""

    def _run(
        self,
        days_inactive: Optional[int] = None,
        min_balance: Optional[float] = None,
        max_balance: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        try:
            # Build queryset
            queryset = DormantAccount.objects.select_related("customer")

            # Apply filters
            if days_inactive:
                queryset = queryset.filter(inactive_period__gte=str(days_inactive))

            if min_balance:
                # Convert working_balance to float for comparison
                queryset = queryset.extra(
                    where=["CAST(working_balance AS DECIMAL) >= %s"],
                    params=[min_balance],
                )

            if max_balance:
                queryset = queryset.extra(
                    where=["CAST(working_balance AS DECIMAL) <= %s"],
                    params=[max_balance],
                )

            total_count = queryset.count()

            # Create query handle for large datasets
            if total_count > 100:
                query_id = f"dormant_{uuid.uuid4().hex[:12]}"

                QueryCacheModel.objects.create(
                    query_id=query_id,
                    query_type="dormant_accounts",
                    query_sql=str(queryset.query),
                    result_count=total_count,
                    created_by_id=1,  # TODO: Get from context
                    metadata={
                        "days_inactive": days_inactive,
                        "min_balance": min_balance,
                        "max_balance": max_balance,
                    },
                )

                # Return sample
                sample_accounts = []
                for account in queryset[:5]:
                    sample_accounts.append(
                        {
                            "id": account.id,
                            "customer_name": account.customer.full_name,
                            "account_number": account.account_number,
                            "working_balance": account.working_balance,
                            "inactive_period": account.inactive_period,
                            "currency": account.currency,
                        }
                    )

                return {
                    "query_id": query_id,
                    "total_count": total_count,
                    "sample_data": sample_accounts,
                    "message": f'Found {total_count} dormant accounts. Use query_id "{query_id}" for bulk operations.',
                }

            else:
                # Return data directly
                accounts = queryset[offset : offset + limit]
                result_data = []

                for account in accounts:
                    result_data.append(
                        {
                            "id": account.id,
                            "customer_id": account.customer.id,
                            "customer_name": account.customer.full_name,
                            "customer_email": account.customer.customer_email,
                            "account_number": account.account_number,
                            "working_balance": account.working_balance,
                            "inactive_period": account.inactive_period,
                            "currency": account.currency,
                            "last_credit_date": account.date_last_cr_cust,
                            "last_debit_date": account.amnt_last_dr_cust,
                        }
                    )

                return {
                    "dormant_accounts": result_data,
                    "total_count": total_count,
                    "returned_count": len(result_data),
                }

        except Exception as e:
            logger.error(f"Error in dormant account search: {str(e)}")
            return {"error": f"Dormant account search failed: {str(e)}"}


class CustomerDetailTool(BaseTool):
    name: str = "get_customer_details"
    description: str = (
        """Get detailed information about a specific customer including accounts, behavior, and recent activity."""
    )

    def _run(
        self,
        customer_id: int,
        include_accounts: bool = True,
        include_transactions: bool = False,
        include_behavior: bool = True,
    ) -> Dict[str, Any]:
        try:
            customer = Customer.objects.get(id=customer_id)

            # Basic customer info
            customer_data = {
                "id": customer.id,
                "full_name": customer.full_name,
                "short_name": customer.short_name,
                "customer_number": customer.customer_number,
                "email": customer.customer_email,
                "phone": customer.mobile_number,
                "customer_type": customer.customer_type,
                "nationality": customer.nationality,
                "industry": customer.industry,
                "sector": customer.sector,
                "relationship_officer": customer.relationship_officer,
                "date_of_birth": customer.date_of_birth,
                "age": customer.age,
                "gender": customer.gender,
            }

            # Include bank accounts
            if include_accounts:
                accounts = []
                for account in customer.bank_accounts.all():
                    accounts.append(
                        {
                            "id": account.id,
                            "account_number": account.account_number,
                            "account_name": account.account_name,
                            "account_category": account.account_category,
                            "currency": account.currency,
                            "opening_date": account.opening_date,
                        }
                    )
                customer_data["bank_accounts"] = accounts

            # Include behavior analysis
            if include_behavior and hasattr(customer, "behaviors"):
                behavior = customer.behaviors
                customer_data["behavior_analysis"] = {
                    "average_account_balance": float(behavior.average_account_balance),
                    "number_of_transactions": behavior.number_of_transactions,
                    "min_transaction_amount": float(behavior.min_transaction_amount),
                    "max_transaction_amount": float(behavior.max_transaction_amount),
                }

            # Include recent transactions
            if include_transactions:
                recent_transactions = []
                transactions = CustomerTransaction.objects.filter(
                    Q(debit_account__customer=customer)
                    | Q(credit_account__customer=customer)
                ).order_by("-date_created")[:10]

                for txn in transactions:
                    recent_transactions.append(
                        {
                            "transaction_ref": txn.transaction_ref,
                            "narration": txn.transaction_narration,
                            "debit_amount": float(txn.debit_amount),
                            "credit_amount": float(txn.credit_amount),
                            "closing_balance": float(txn.closing_balance),
                            "booking_date": txn.booking_date,
                            "value_date": txn.value_date,
                        }
                    )
                customer_data["recent_transactions"] = recent_transactions

            # Dormant account status
            dormant_accounts = customer.dormant_accounts.all()
            if dormant_accounts.exists():
                customer_data["dormant_status"] = {
                    "has_dormant_accounts": True,
                    "dormant_account_count": dormant_accounts.count(),
                    "dormant_accounts": [
                        {
                            "account_number": acc.account_number,
                            "inactive_period": acc.inactive_period,
                            "working_balance": acc.working_balance,
                        }
                        for acc in dormant_accounts
                    ],
                }
            else:
                customer_data["dormant_status"] = {"has_dormant_accounts": False}

            return customer_data

        except Customer.DoesNotExist:
            return {"error": f"Customer with ID {customer_id} not found"}
        except Exception as e:
            logger.error(f"Error getting customer details: {str(e)}")
            return {"error": f"Failed to get customer details: {str(e)}"}


class SendEmailTool(BaseTool):
    name: str = "send_customer_email"
    description: str = (
        """Send an email to a specific customer. Use this for individual email communications."""
    )

    def _run(
        self, customer_id: int, subject: str, body: str, to_email: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            customer = Customer.objects.get(id=customer_id)

            # Use provided email or customer's default email
            email_address = to_email or customer.customer_email
            if not email_address:
                return {"error": "No email address available for this customer"}

            # Create email record
            email_obj = CustomerEmail.objects.create(
                customer=customer, subject=subject, body=body, to_email=email_address
            )

            # Send email asynchronously
            send_customer_email_with_attachments.delay(email_obj.id)

            return {
                "success": True,
                "email_id": email_obj.id,
                "message": f"Email queued for delivery to {customer.full_name} ({email_address})",
            }

        except Customer.DoesNotExist:
            return {"error": f"Customer with ID {customer_id} not found"}
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return {"error": f"Failed to send email: {str(e)}"}


class BulkEmailTool(BaseTool):
    name: str = "send_bulk_email"
    description: str = (
        """Send bulk emails using a query handle. The system will handle template processing and delivery."""
    )

    def _run(
        self, query_id: str, subject: str, body: str, send_immediately: bool = False
    ) -> Dict[str, Any]:
        try:
            # Get query cache entry
            cache_entry = QueryCacheModel.objects.get(query_id=query_id)

            if send_immediately:
                # Use the new templated bulk email task
                task_result = send_templated_bulk_email.delay(
                    query_id=query_id,
                    subject_template=subject,
                    body_template=body,
                    created_by_id=cache_entry.created_by_id,
                )

                return {
                    "success": True,
                    "task_id": task_result.id,
                    "message": f"Bulk email campaign started for {cache_entry.result_count} recipients. Task ID: {task_result.id}",
                }
            else:
                # Queue the task for later processing
                task_result = send_templated_bulk_email.apply_async(
                    kwargs={
                        "query_id": query_id,
                        "subject_template": subject,
                        "body_template": body,
                        "created_by_id": cache_entry.created_by_id,
                    },
                    countdown=60,  # Start in 1 minute
                )

                return {
                    "success": True,
                    "task_id": task_result.id,
                    "message": f"Bulk email campaign queued for {cache_entry.result_count} recipients. Will start in 1 minute. Task ID: {task_result.id}",
                }

        except QueryCacheModel.DoesNotExist:
            return {"error": f"Query handle {query_id} not found or expired"}
        except Exception as e:
            logger.error(f"Error in bulk email: {str(e)}")
            return {"error": f"Bulk email failed: {str(e)}"}


class QueryStatsTool(BaseTool):
    name: str = "get_query_stats"
    description: str = (
        """Get statistics and summary information about a query handle."""
    )

    def _run(self, query_id: str) -> Dict[str, Any]:
        try:
            cache_entry = QueryCacheModel.objects.get(query_id=query_id)

            return {
                "query_id": query_id,
                "query_type": cache_entry.query_type,
                "total_count": cache_entry.result_count,
                "created_at": cache_entry.created_at.isoformat(),
                "expires_at": cache_entry.expires_at.isoformat(),
                "metadata": cache_entry.metadata,
                "is_expired": timezone.now() > cache_entry.expires_at,
            }

        except QueryCacheModel.DoesNotExist:
            return {"error": f"Query handle {query_id} not found or expired"}


class CustomerSegmentationTool(BaseTool):
    name: str = "search_by_segmentation"
    description: str = """Search customers using predefined segmentation criteria."""

    def _run(
        self, segmentation_id: int, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        try:
            segmentation = CustomerSegmentation.objects.get(id=segmentation_id)

            # Build queryset based on segmentation criteria
            queryset = Customer.objects.filter(
                behaviors__average_account_balance__gte=segmentation.average_account_balance,
                behaviors__number_of_transactions__gte=segmentation.number_of_transactions,
                behaviors__min_transaction_amount__gte=segmentation.min_transaction_amount,
                behaviors__max_transaction_amount__lte=segmentation.max_transaction_amount,
            )

            # Apply gender filter if specified
            if segmentation.gender:
                genders = [
                    g.strip() for g in segmentation.gender.split(",") if g.strip()
                ]
                queryset = queryset.filter(gender__in=genders)

            total_count = queryset.count()

            # Create query handle for large result sets
            if total_count > 100:
                query_id = f"segment_{uuid.uuid4().hex[:12]}"

                QueryCacheModel.objects.create(
                    query_id=query_id,
                    query_type="segmentation_search",
                    query_sql=str(queryset.query),
                    result_count=total_count,
                    created_by_id=1,  # TODO: Get from context
                    metadata={
                        "segmentation_id": segmentation_id,
                        "segmentation_name": segmentation.segmentation_name,
                    },
                )

                return {
                    "query_id": query_id,
                    "total_count": total_count,
                    "segmentation_name": segmentation.segmentation_name,
                    "message": f'Found {total_count} customers matching segmentation "{segmentation.segmentation_name}". Use query_id "{query_id}" for bulk operations.',
                }

            else:
                # Return data directly for small result sets
                customers = queryset[offset : offset + limit]
                result_data = []

                for customer in customers:
                    result_data.append(
                        {
                            "id": customer.id,
                            "full_name": customer.full_name,
                            "customer_email": customer.customer_email,
                            "mobile_number": customer.mobile_number,
                            "customer_type": customer.customer_type,
                            "sector": customer.sector,
                        }
                    )

                return {
                    "customers": result_data,
                    "total_count": total_count,
                    "segmentation_name": segmentation.segmentation_name,
                    "returned_count": len(result_data),
                }

        except CustomerSegmentation.DoesNotExist:
            return {"error": f"Segmentation with ID {segmentation_id} not found"}
        except Exception as e:
            logger.error(f"Error in segmentation search: {str(e)}")
            return {"error": f"Segmentation search failed: {str(e)}"}


class LikelyDormantAccountTool(BaseTool):
    name: str = "predict_dormant_accounts"
    description: str = """Predict customer bank accounts that are likely to become dormant. "
        "Returns a query handle for large result sets. Use this tool to identify "
        "accounts whose last transaction is older than a specified number of "
        "days but that are *not* yet marked as dormant."""

    def _run(
        self,
        days_without_transaction: int = 60,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Find accounts with no transactions for the given number of days."""
        try:
            from django.utils import timezone
            from datetime import timedelta

            cutoff = timezone.now() - timedelta(days=days_without_transaction)
            cutoff_str = cutoff.strftime("%Y%m%d")

            from django.db.models import Max

            latest_txn = CustomerTransaction.objects.values(
                "debit_account_id"
            ).annotate(last_txn=Max("booking_date"))

            dormant_ids = DormantAccount.objects.values_list(
                "account_number", flat=True
            )

            candidate_qs = BankAccount.objects.exclude(
                account_number__in=dormant_ids
            ).filter(
                id__in=[
                    row["debit_account_id"]
                    for row in latest_txn
                    if row["last_txn"] < cutoff_str
                ]
            )

            total_count = candidate_qs.count()

            if total_count > 100:
                query_id = f"likely_dormant_{uuid.uuid4().hex[:12]}"
                QueryCacheModel.objects.create(
                    query_id=query_id,
                    query_type="likely_dormant_accounts",
                    query_sql=str(candidate_qs.query),
                    result_count=total_count,
                    created_by_id=1,  # TODO: context user
                    metadata={"days_without_transaction": days_without_transaction},
                )

                sample = list(
                    candidate_qs[:5].values(
                        "id",
                        "account_number",
                        "customer__full_name",
                        "opening_date",
                        "currency",
                    )
                )

                return {
                    "query_id": query_id,
                    "total_count": total_count,
                    "sample_data": sample,
                    "message": (
                        f"Found {total_count} accounts with no transactions for {days_without_transaction} days. "
                        f"Use query_id '{query_id}' for bulk operations."
                    ),
                }
            else:
                data = []
                for acc in candidate_qs[offset : offset + limit]:
                    data.append(
                        {
                            "id": acc.id,
                            "account_number": acc.account_number,
                            "customer_name": acc.customer.full_name,
                            "opening_date": acc.opening_date,
                            "currency": acc.currency,
                        }
                    )

                return {
                    "likely_dormant_accounts": data,
                    "total_count": total_count,
                    "returned_count": len(data),
                }

        except Exception as e:
            logger.error(f"Error predicting dormant accounts: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}


llm = ChatOpenAI(model=AGENT_CONFIG.get("MODEL", "gpt-4"))

allowed_tables = SECURITY_CONFIG.get("DEFAULT_ALLOWED_TABLES", [])


class DataAnalysisInput(BaseModel):
    query: str = Field(description="The natural language query to analyst")


class DataAnalysisTool(BaseTool):
    name: str = "ai_powered_data_analysis_tool"
    description: str = (
        "Generic data analysis tool executes AI-powered exploration and querying of the database. Use this to ask a queric question if the user's message cannot be answered with any of the other tools directly"
    )
    agent: Any = None
    top_k: int = 10
    args_schema: ArgsSchema | None = DataAnalysisInput

    @model_validator(mode="before")
    def add_agent(cls, data):
        top_k = data.get("top_k", 10)
        data["agent"] = TextToSQLAgent(
            llm=llm,
            system_prompt=SQL_SYSTEM_TEMPLATE,
            include_tables=allowed_tables,
            top_k=top_k,
            ai_answer=False,
        ).compile_workflow()
        return data

    def _run(
        self,
        query: str,
        **kwargs,
    ):
        try:
            response = self.agent.invoke({"messages": [HumanMessage(content=query)]})
        except Exception as e:
            logger.error(f"Error in data analysis tool: {e}")
            return {"message": "Error in data analysis tool"}
        msg = response["messages"][-1].content
        if "no results found" in str(msg).lower():
            return {"message": "No results found"}
        if "error" in str(msg).lower():
            return {"message": "Error in data analysis tool"}
        return msg


AGENT_TOOLS = [
    CustomerSearchTool(),
    DormantAccountSearchTool(),
    CustomerDetailTool(),
    SendEmailTool(),
    BulkEmailTool(),
    QueryStatsTool(),
    CustomerSegmentationTool(),
    LikelyDormantAccountTool(),
    DataAnalysisTool(),
]


def get_tool_by_name(tool_name: str) -> Optional[BaseTool]:
    """Get a tool by its name"""
    for tool in AGENT_TOOLS:
        if tool.name == tool_name:
            return tool
    return None


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Get all tool schemas for LLM binding"""
    schemas = []
    for tool in AGENT_TOOLS:
        schemas.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.schema() if tool.args_schema else {},
            }
        )
    return schemas
