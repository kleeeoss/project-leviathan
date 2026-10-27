import asyncio
import os
import random
import json
import aioboto3
import collections
import string
from enum import Enum
from typing import List, Any, Optional, Dict
from dotenv import load_dotenv
from google import genai
from genai_client import get_vertex_client
from models import SCHEMA_REGISTRY

load_dotenv()

# LocalStack Configuration
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "project-6d3877ae-baae-4159-a77")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")
def _get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


MODEL_INPUT_PRICE_PER_M = _get_env_float("MODEL_INPUT_PRICE_PER_M", 0.30)
MODEL_OUTPUT_PRICE_PER_M = _get_env_float("MODEL_OUTPUT_PRICE_PER_M", 2.50)


def calculate_cost(prompt_tokens: Optional[int], candidates_tokens: Optional[int]) -> Optional[float]:
    if prompt_tokens is None or candidates_tokens is None:
        return None
    input_rate = _get_env_float("MODEL_INPUT_PRICE_PER_M", MODEL_INPUT_PRICE_PER_M)
    output_rate = _get_env_float("MODEL_OUTPUT_PRICE_PER_M", MODEL_OUTPUT_PRICE_PER_M)
    cost = (prompt_tokens / 1_000_000.0) * input_rate + (candidates_tokens / 1_000_000.0) * output_rate
    return round(cost, 6)


class TaskStatus(Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskNode:
    def __init__(
        self,
        task_id: str,
        prompt: str,
        dependencies: Optional[List[str]] = None,
        output_schema: Optional[str] = None
    ):
        self.task_id = task_id
        self.prompt = prompt
        self.dependencies = dependencies if dependencies is not None else []
        self.output_schema = output_schema
        self.status = TaskStatus.PENDING
        self.result: Any = None
        self.usage: Optional[dict] = None


class ResultWrapper(dict):
    """
    Dictionary wrapper supporting attribute-style dot notation in format strings
    (e.g., {fetch_aws.summary}) alongside fallback stringification ({fetch_aws}).
    """
    def __getattr__(self, name: str):
        if name in self:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, ResultWrapper):
                return ResultWrapper(val)
            return val
        raise AttributeError(f"Field '{name}' not found in task result.")

    def __str__(self):
        if "summary" in self:
            return str(self["summary"])
        return json.dumps(self)


# ==========================================
# ☁️ REAL EMULATED AWS INFRASTRUCTURE
# ==========================================

class AWSQueueManager:
    """Manages connection to emulated SQS queues."""

    def __init__(self):
        self.session = aioboto3.Session(
            aws_access_key_id="mock_key",
            aws_secret_access_key="mock_secret",
            region_name=AWS_REGION
        )
        self.task_queue_url = None
        self.response_queue_url = None

    async def initialize_queues(self):
        """Connects to existing physical infrastructure provisioned by Terraform, with auto-fallback."""
        async with self.session.resource("sqs", endpoint_url=LOCALSTACK_ENDPOINT) as sqs:
            try:
                task_queue = await sqs.get_queue_by_name(QueueName="leviathan-prod-task-queue")
                response_queue = await sqs.get_queue_by_name(QueueName="leviathan-prod-response-queue")
                self.task_queue_url = task_queue.url
                self.response_queue_url = response_queue.url
                print(f"🔗 [AWS INFRA] Successfully linked to Terraform Task Queue: {self.task_queue_url}")
                print(f"🔗 [AWS INFRA] Successfully linked to Terraform Response Queue: {self.response_queue_url}")
            except Exception as e:
                print(f"⚠️ [AWS INFRA] Terraform queues not found ({e}). Auto-provisioning queues in LocalStack...")
                task_queue = await sqs.create_queue(QueueName="leviathan-prod-task-queue")
                response_queue = await sqs.create_queue(QueueName="leviathan-prod-response-queue")
                self.task_queue_url = task_queue.url
                self.response_queue_url = response_queue.url
                print(f"🔗 [AWS INFRA] Auto-provisioned Task Queue: {self.task_queue_url}")
                print(f"🔗 [AWS INFRA] Auto-provisioned Response Queue: {self.response_queue_url}")


class LambdaWorker:
    """Simulates an AWS Lambda function polling an actual SQS queue via SDK."""

    def __init__(self, worker_id: int, queue_manager: AWSQueueManager, client: Optional[genai.Client] = None):
        self.worker_id = f"Lambda-{worker_id}"
        self.qm = queue_manager
        self.client = client

    async def start_polling(self):
        print(f"🟢 [{self.worker_id}] Polling real LocalStack SQS endpoint...")

        async with self.qm.session.client("sqs", endpoint_url=LOCALSTACK_ENDPOINT) as sqs:
            while True:
                response = await sqs.receive_message(
                    QueueUrl=self.qm.task_queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=1
                )

                if "Messages" in response:
                    for message in response["Messages"]:
                        try:
                            body = json.loads(message["Body"])
                            print(f"⚡ [{self.worker_id}] Pulled Task '{body['task_id']}' from real SQS!")

                            result_payload = await self._invoke_llm(body)

                            await sqs.send_message(
                                QueueUrl=self.qm.response_queue_url,
                                MessageBody=json.dumps(result_payload)
                            )

                            await sqs.delete_message(
                                QueueUrl=self.qm.task_queue_url,
                                ReceiptHandle=message["ReceiptHandle"]
                            )
                        except Exception as e:
                            print(f"⚠️ [{self.worker_id}] Transient error processing task message: {e}. Message retained in queue.")
                await asyncio.sleep(0.5)

    async def _invoke_llm(self, task_data: dict) -> dict:
        client = self.client or get_vertex_client()
        if client is None:
            return {
                "task_id": task_data["task_id"],
                "status": TaskStatus.FAILED.value,
                "result": None,
                "error": "Vertex AI Client unavailable (offline or missing credentials)",
                "usage": {
                    "status": "UNKNOWN",
                    "prompt_tokens": None,
                    "candidates_tokens": None,
                    "total_tokens": None,
                    "cost_usd": None
                }
            }

        schema_name = task_data.get("output_schema")
        schema_cls = SCHEMA_REGISTRY.get(schema_name) if schema_name else None

        max_retries = 3
        cum_prompt_tokens = 0
        cum_candidates_tokens = 0
        cum_total_tokens = 0
        usage_is_unknown = False
        model_id = os.getenv("GEMINI_MODEL_ID", GEMINI_MODEL_ID)

        for attempt in range(1, max_retries + 1):
            try:
                call_config = {}
                if schema_cls:
                    call_config["response_mime_type"] = "application/json"
                    call_config["response_schema"] = schema_cls

                response = await client.aio.models.generate_content(
                    model=model_id,
                    contents=task_data["prompt"],
                    config=call_config if call_config else None
                )

                # Token accounting across retries
                if response.usage_metadata is not None:
                    p_tok = response.usage_metadata.prompt_token_count or 0
                    c_tok = response.usage_metadata.candidates_token_count or 0
                    t_tok = response.usage_metadata.total_token_count or (p_tok + c_tok)
                    cum_prompt_tokens += p_tok
                    cum_candidates_tokens += c_tok
                    cum_total_tokens += t_tok
                else:
                    usage_is_unknown = True

                raw_text = response.text.strip() if response.text else ""

                # Structured validation if schema is present
                if schema_cls:
                    validated_obj = schema_cls.model_validate_json(raw_text)
                    result_val = validated_obj.model_dump()
                else:
                    result_val = raw_text

                if usage_is_unknown:
                    usage_payload = {
                        "status": "UNKNOWN",
                        "prompt_tokens": None,
                        "candidates_tokens": None,
                        "total_tokens": None,
                        "cost_usd": None
                    }
                else:
                    usage_payload = {
                        "status": "RECORDED",
                        "prompt_tokens": cum_prompt_tokens,
                        "candidates_tokens": cum_candidates_tokens,
                        "total_tokens": cum_total_tokens,
                        "cost_usd": calculate_cost(cum_prompt_tokens, cum_candidates_tokens)
                    }

                return {
                    "task_id": task_data["task_id"],
                    "status": TaskStatus.COMPLETED.value,
                    "result": result_val,
                    "error": None,
                    "usage": usage_payload
                }

            except Exception as e:
                print(f"⚠️ [{self.worker_id}] Transient Error on '{task_data['task_id']}' (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    if usage_is_unknown or cum_total_tokens == 0:
                        usage_payload = {
                            "status": "UNKNOWN",
                            "prompt_tokens": None,
                            "candidates_tokens": None,
                            "total_tokens": None,
                            "cost_usd": None
                        }
                    else:
                        usage_payload = {
                            "status": "RECORDED",
                            "prompt_tokens": cum_prompt_tokens,
                            "candidates_tokens": cum_candidates_tokens,
                            "total_tokens": cum_total_tokens,
                            "cost_usd": calculate_cost(cum_prompt_tokens, cum_candidates_tokens)
                        }
                    return {
                        "task_id": task_data["task_id"],
                        "status": TaskStatus.FAILED.value,
                        "result": None,
                        "error": str(e),
                        "usage": usage_payload
                    }
                await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 1.0))


# ==========================================
# 🧠 THE DISTRIBUTED ORCHESTRATOR
# ==========================================

class DAGOrchestrator:
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self.qm = AWSQueueManager()
        self.completed_count = 0
        self.failed_count = 0
        self.cancelled_count = 0
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.has_unknown_usage = False

    def add_node(self, node: TaskNode):
        self.nodes[node.task_id] = node

    def load_from_json(self, filepath: str):
        with open(filepath, "r") as file:
            data = json.load(file)
        for task_data in data.get("tasks", []):
            self.add_node(TaskNode(
                task_id=task_data["id"],
                prompt=task_data["prompt"],
                dependencies=task_data.get("dependencies", []),
                output_schema=task_data.get("output_schema")
            ))
        print(f"📄 [ENGINE] Loaded {len(self.nodes)} tasks from JSON.")

    def validate_graph(self) -> List[str]:
        """
        Validates graph acyclicity and dependency consistency using Kahn's algorithm (O(V+E)).
        Also validates that output_schema exists in SCHEMA_REGISTRY and prompt template
        variables/attributes match declared dependencies and schema definitions.
        Returns topological ordering of task IDs if valid.
        Raises ValueError if a cycle, unknown dependency, schema mismatch, or template error is detected.
        """
        # 1. Check schemas, dependencies, and prompt template variables
        for task_id, node in self.nodes.items():
            if node.output_schema and node.output_schema not in SCHEMA_REGISTRY:
                raise ValueError(f"Task '{task_id}' specifies unknown output_schema '{node.output_schema}'")

            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Task '{task_id}' depends on unknown task '{dep}'")

            formatter = string.Formatter()
            for _, field_name, _, _ in formatter.parse(node.prompt):
                if field_name is not None and field_name != "":
                    base_dep = field_name.split(".")[0]
                    if base_dep not in node.dependencies:
                        raise ValueError(
                            f"Task '{task_id}' prompt template variable '{field_name}' "
                            f"not found in declared dependencies: {node.dependencies}"
                        )
                    if "." in field_name:
                        parts = field_name.split(".")
                        attr_name = parts[1]
                        dep_node = self.nodes[base_dep]
                        if dep_node.output_schema:
                            schema_cls = SCHEMA_REGISTRY.get(dep_node.output_schema)
                            if schema_cls and attr_name not in schema_cls.model_fields:
                                raise ValueError(
                                    f"Task '{task_id}' template variable '{field_name}' references attribute '{attr_name}' "
                                    f"which does not exist in schema '{dep_node.output_schema}' "
                                    f"(valid fields: {list(schema_cls.model_fields.keys())})"
                                )
                        else:
                            raise ValueError(
                                f"Task '{task_id}' template variable '{field_name}' references attribute '{attr_name}', "
                                f"but dependency '{base_dep}' does not declare an output_schema."
                            )

        # 2. Kahn's Algorithm for cycle detection (O(V + E) using deque)
        in_degree = {task_id: len(node.dependencies) for task_id, node in self.nodes.items()}
        adj_list = collections.defaultdict(list)
        for task_id, node in self.nodes.items():
            for dep in node.dependencies:
                adj_list[dep].append(task_id)

        queue = collections.deque([task_id for task_id, deg in in_degree.items() if deg == 0])
        topological_order = []

        while queue:
            curr = queue.popleft()
            topological_order.append(curr)
            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topological_order) != len(self.nodes):
            raise ValueError("Cycle detected in DAG")

        return topological_order

    def _handle_failure(self, failed_task_id: str, error_msg: str, shared_in_degree: dict, adj_list: dict):
        """
        Handles terminal task failure by marking the failed node and performing BFS
        cascade cancellation of all transitive downstream descendants while leaving
        orthogonal, independent DAG branches untouched.
        """
        failed_node = self.nodes[failed_task_id]
        if failed_node.status == TaskStatus.FAILED:
            return

        failed_node.status = TaskStatus.FAILED
        failed_node.result = None
        self.failed_count += 1
        print(f"❌ [ENGINE] Marked Task '{failed_task_id}' as FAILED: {error_msg}")

        # BFS cascade cancellation of all transitive downstream descendants
        queue = collections.deque(adj_list.get(failed_task_id, []))
        visited = set(queue)

        while queue:
            descendant_id = queue.popleft()
            descendant = self.nodes[descendant_id]
            if descendant.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
                descendant.status = TaskStatus.CANCELLED
                self.cancelled_count += 1
                print(f"🚫 [ENGINE] Cascade Cancelled downstream task '{descendant_id}' due to ancestor failure of '{failed_task_id}'.")
            for child in adj_list.get(descendant_id, []):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

    async def _publish_ready_tasks(self, shared_in_degree: dict, sqs_client, adj_list: dict):
        """Publishes tasks directly into the LocalStack SQS queue via AWS SDK."""
        for task_id, degree in list(shared_in_degree.items()):
            node = self.nodes[task_id]
            if degree == 0 and node.status == TaskStatus.PENDING:
                context = {}
                for dep in node.dependencies:
                    dep_res = self.nodes[dep].result
                    if isinstance(dep_res, dict):
                        context[dep] = ResultWrapper(dep_res)
                    else:
                        context[dep] = dep_res
                try:
                    resolved_prompt = node.prompt.format(**context)
                    payload = {
                        "task_id": task_id,
                        "prompt": resolved_prompt,
                        "output_schema": node.output_schema
                    }

                    # Send via AWS SDK
                    await sqs_client.send_message(
                        QueueUrl=self.qm.task_queue_url,
                        MessageBody=json.dumps(payload)
                    )

                    node.status = TaskStatus.QUEUED
                    print(f"📥 [ENGINE] Successfully pushed payload for '{task_id}' into AWS SQS.")
                except (KeyError, AttributeError) as e:
                    print(f"❌ [ENGINE] Formatting Error on '{task_id}': missing template variable/attribute {e}")
                    self._handle_failure(task_id, f"Missing template variable or attribute: {e}", shared_in_degree, adj_list)

    async def _process_response_message(
        self,
        message: dict,
        shared_in_degree: dict,
        adj_list: dict,
        sqs_client
    ) -> bool:
        """
        Processes a single SQS message from the response queue.
        Distinguishes transient processing errors from terminal task results.
        Returns True if message was acknowledged (terminal result, duplicate, or unroutable message),
        or False if message had a transient processing error (retained in SQS for redelivery).
        """
        try:
            body = json.loads(message["Body"])
            task_id = body["task_id"]
            status = TaskStatus(body["status"])
            result = body.get("result")
            error = body.get("error")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"⚠️ [ENGINE] Transient processing error on received message: {exc}. Retaining in SQS.")
            return False

        # If task_id does not belong to this DAG, acknowledge and delete to prevent queue poisoning
        if task_id not in self.nodes:
            print(f"⚠️ [ENGINE] Unknown task_id '{task_id}' in response queue. Acknowledging to prevent queue poisoning.")
            await sqs_client.delete_message(
                QueueUrl=self.qm.response_queue_url,
                ReceiptHandle=message["ReceiptHandle"]
            )
            return True

        node = self.nodes[task_id]

        # Idempotency guard: If node is already in a terminal state, acknowledge duplicate delivery
        # Prevents double-counting FinOps tokens or executing duplicate completions
        if node.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            print(f"⚠️ [ENGINE] Duplicate response received for task '{task_id}' in state {node.status.value}. Acknowledging.")
            await sqs_client.delete_message(
                QueueUrl=self.qm.response_queue_url,
                ReceiptHandle=message["ReceiptHandle"]
            )
            return True

        # Process FinOps token & cost accounting from first delivery
        usage = body.get("usage")
        node.usage = usage
        if usage and usage.get("status") == "RECORDED":
            self.total_prompt_tokens += usage.get("prompt_tokens") or 0
            self.total_candidates_tokens += usage.get("candidates_tokens") or 0
            self.total_tokens += usage.get("total_tokens") or 0
            self.total_cost_usd += usage.get("cost_usd") or 0.0
        elif usage and usage.get("status") == "UNKNOWN":
            self.has_unknown_usage = True
        else:
            self.has_unknown_usage = True
            node.usage = {
                "status": "UNKNOWN",
                "prompt_tokens": None,
                "candidates_tokens": None,
                "total_tokens": None,
                "cost_usd": None
            }

        try:
            if status == TaskStatus.COMPLETED:
                node.status = TaskStatus.COMPLETED
                node.result = result
                self.completed_count += 1
                print(f"✅ [ENGINE] Context Loopback: Processed '{node.task_id}' from Response SQS.")
                for neighbor in adj_list.get(node.task_id, []):
                    shared_in_degree[neighbor] -= 1

                # Publish newly unlocked tasks
                await self._publish_ready_tasks(shared_in_degree, sqs_client, adj_list)
            elif status == TaskStatus.FAILED:
                print(f"❌ [ENGINE] Terminal Failure reported on Node '{node.task_id}': {error}")
                self._handle_failure(node.task_id, error or "Task failed", shared_in_degree, adj_list)
            else:
                print(f"⚠️ [ENGINE] Unexpected task status '{status}' for '{node.task_id}'")
        finally:
            await sqs_client.delete_message(
                QueueUrl=self.qm.response_queue_url,
                ReceiptHandle=message["ReceiptHandle"]
            )
        return True

    def get_finops_summary(self) -> dict:
        """Returns cumulative FinOps cost and token tracking across all DAG tasks."""
        return {
            "status": "UNKNOWN" if self.has_unknown_usage else "RECORDED",
            "total_prompt_tokens": None if self.has_unknown_usage else self.total_prompt_tokens,
            "total_candidates_tokens": None if self.has_unknown_usage else self.total_candidates_tokens,
            "total_tokens": None if self.has_unknown_usage else self.total_tokens,
            "total_cost_usd": None if self.has_unknown_usage else round(self.total_cost_usd, 6),
            "pricing_rates": {
                "input_per_m": _get_env_float("MODEL_INPUT_PRICE_PER_M", MODEL_INPUT_PRICE_PER_M),
                "output_per_m": _get_env_float("MODEL_OUTPUT_PRICE_PER_M", MODEL_OUTPUT_PRICE_PER_M)
            },
            "tasks": {task_id: node.usage for task_id, node in self.nodes.items() if node.usage}
        }

    async def run(self):
        print("🚀 [ENGINE] Initializing LocalStack Cloud-Native Runtime...")

        # 0. Pre-Flight Graph Validation
        self.validate_graph()

        # 1. Spin up queues inside LocalStack container
        await self.qm.initialize_queues()

        # 2. Start AWS Lambda Worker Pool emulation
        workers = [LambdaWorker(i, self.qm) for i in range(1, 4)]
        worker_tasks = [asyncio.create_task(w.start_polling()) for w in workers]

        # 3. Setup Graph Tracking
        shared_in_degree = {task_id: len(t.dependencies) for task_id, t in self.nodes.items()}
        adj_list = {task_id: [] for task_id in self.nodes}
        for task_id, task in self.nodes.items():
            for dep in task.dependencies:
                adj_list[dep].append(task_id)

        target_count = len(self.nodes)

        # 4. Open Orchestrator Client connection to poll for completions
        async with self.qm.session.client("sqs", endpoint_url=LOCALSTACK_ENDPOINT) as sqs:
            await self._publish_ready_tasks(shared_in_degree, sqs, adj_list)

            while (self.completed_count + self.failed_count + self.cancelled_count) < target_count:
                response = await sqs.receive_message(
                    QueueUrl=self.qm.response_queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=1
                )

                if "Messages" in response:
                    for message in response["Messages"]:
                        await self._process_response_message(message, shared_in_degree, adj_list, sqs)

                await asyncio.sleep(0.5)

        for wt in worker_tasks:
            wt.cancel()

        finops = self.get_finops_summary()
        print("\n==========================================")
        print("📊 [ENGINE] LEVIATHAN FINOPS EXECUTION SUMMARY")
        print("==========================================")
        if finops["status"] == "RECORDED":
            print(f"💰 Total Prompt Tokens     : {finops['total_prompt_tokens']}")
            print(f"💰 Total Completion Tokens : {finops['total_candidates_tokens']}")
            print(f"💰 Total Token Spend       : {finops['total_tokens']}")
            print(f"💵 Total Cloud Cost (USD)  : ${finops['total_cost_usd']:.6f}")
        else:
            print("⚠️ Usage Status: UNKNOWN (Missing usage metadata; cost not claimed as $0.00)")
        print("==========================================\n")
        print("🏁 [ENGINE] Cloud-Native pipeline finished execution. Dismantling simulated resources.")


if __name__ == "__main__":
    orchestrator = DAGOrchestrator()
    orchestrator.load_from_json("workflow.json")
    asyncio.run(orchestrator.run())
