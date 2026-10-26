import asyncio
import os
import random
import json
import aioboto3
import collections
import string
from google import genai
from dotenv import load_dotenv
from enum import Enum
from typing import List, Any, Optional, Dict

load_dotenv()

# LocalStack Configuration
LOCALSTACK_ENDPOINT = "http://localhost:4566"
AWS_REGION = "us-east-1"


class TaskStatus(Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskNode:
    def __init__(self, task_id: str, prompt: str, dependencies: Optional[List[str]] = None):
        self.task_id = task_id
        self.prompt = prompt
        self.dependencies = dependencies if dependencies is not None else []
        self.status = TaskStatus.PENDING
        self.result: Any = None


# ==========================================
# ☁️ REAL EMULATED AWS INFRASTRUCTURE
# ==========================================

class AWSQueueManager:
    """Manages creation and interaction with actual emulated SQS queues."""

    def __init__(self):
        # Setup an asynchronous AWS Session using fake credentials for LocalStack
        self.session = aioboto3.Session(
            aws_access_key_id="mock_key",
            aws_secret_access_key="mock_secret",
            region_name=AWS_REGION
        )
        self.task_queue_url = None
        self.response_queue_url = None

    async def initialize_queues(self):
        """Connects to existing physical infrastructure provisioned by Terraform."""
        async with self.session.resource("sqs", endpoint_url=LOCALSTACK_ENDPOINT) as sqs:
            try:
                # We no longer CREATE queues. We FETCH the ones Terraform built.
                task_queue = await sqs.get_queue_by_name(QueueName="leviathan-prod-task-queue")
                response_queue = await sqs.get_queue_by_name(QueueName="leviathan-prod-response-queue")

                self.task_queue_url = task_queue.url
                self.response_queue_url = response_queue.url
                print(f"🔗 [AWS INFRA] Successfully linked to Terraform Task Queue: {self.task_queue_url}")
                print(f"🔗 [AWS INFRA] Successfully linked to Terraform Response Queue: {self.response_queue_url}")
            except Exception as e:
                print(f"❌ [AWS INFRA] Failed to find Terraform queues. Did you run 'terraform apply'? Error: {e}")
                raise e


class LambdaWorker:
    """Simulates an AWS Lambda function polling an actual SQS queue via SDK."""

    def __init__(self, worker_id: int, queue_manager: AWSQueueManager):
        self.worker_id = f"Lambda-{worker_id}"
        self.qm = queue_manager

    async def start_polling(self):
        print(f"🟢 [{self.worker_id}] Polling real LocalStack SQS endpoint...")

        # Open an async client connection to SQS
        async with self.qm.session.client("sqs", endpoint_url=LOCALSTACK_ENDPOINT) as sqs:
            while True:
                # Poll for 1 message at a time
                response = await sqs.receive_message(
                    QueueUrl=self.qm.task_queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=1  # Long polling
                )

                if "Messages" in response:
                    for message in response["Messages"]:
                        try:
                            body = json.loads(message["Body"])
                            print(f"⚡ [{self.worker_id}] Pulled Task '{body['task_id']}' from real SQS!")

                            # Process task via AI
                            result_payload = await self._invoke_llm(body)

                            # Push result to response queue
                            await sqs.send_message(
                                QueueUrl=self.qm.response_queue_url,
                                MessageBody=json.dumps(result_payload)
                            )

                            # Delete message from task queue so no one else processes it
                            await sqs.delete_message(
                                QueueUrl=self.qm.task_queue_url,
                                ReceiptHandle=message["ReceiptHandle"]
                            )
                        except Exception as e:
                            # Transient processing error: do not ack/delete message so visibility timeout allows redelivery
                            print(f"⚠️ [{self.worker_id}] Transient error processing task message: {e}. Message retained in queue.")
                await asyncio.sleep(0.5)

    async def _invoke_llm(self, task_data: dict) -> dict:
        local_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = await local_client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=task_data["prompt"]
                )
                return {
                    "task_id": task_data["task_id"],
                    "status": TaskStatus.COMPLETED.value,  # <-- .value prevents JSON crash
                    "result": response.text.strip(),
                    "error": None
                }
            except Exception as e:
                print(f"⚠️ [{self.worker_id}] Transient Error on '{task_data['task_id']}': {e}")
                if attempt == max_retries:
                    return {
                        "task_id": task_data["task_id"],
                        "status": TaskStatus.FAILED.value,  # <-- .value prevents JSON crash
                        "result": None,
                        "error": str(e)
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

    def add_node(self, node: TaskNode):
        self.nodes[node.task_id] = node

    def load_from_json(self, filepath: str):
        with open(filepath, "r") as file:
            data = json.load(file)
        for task_data in data.get("tasks", []):
            self.add_node(TaskNode(
                task_id=task_data["id"],
                prompt=task_data["prompt"],
                dependencies=task_data.get("dependencies", [])
            ))
        print(f"📄 [ENGINE] Loaded {len(self.nodes)} tasks from JSON.")

    def validate_graph(self) -> List[str]:
        """
        Validates graph acyclicity and dependency consistency using Kahn's algorithm (O(V+E)).
        Also validates that prompt template variables match declared dependencies.
        Returns topological ordering of task IDs if valid.
        Raises ValueError if a cycle, unknown dependency, or template mismatch is detected.
        """
        # 1. Check for unknown dependencies and prompt template variable mismatches
        for task_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Task '{task_id}' depends on unknown task '{dep}'")

            formatter = string.Formatter()
            for _, field_name, _, _ in formatter.parse(node.prompt):
                if field_name is not None and field_name != "":
                    if field_name not in node.dependencies:
                        raise ValueError(
                            f"Task '{task_id}' prompt template variable '{field_name}' "
                            f"not found in declared dependencies: {node.dependencies}"
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
                context = {dep: self.nodes[dep].result for dep in node.dependencies}
                try:
                    resolved_prompt = node.prompt.format(**context)
                    payload = {"task_id": task_id, "prompt": resolved_prompt}

                    # Send via AWS SDK
                    await sqs_client.send_message(
                        QueueUrl=self.qm.task_queue_url,
                        MessageBody=json.dumps(payload)
                    )

                    node.status = TaskStatus.QUEUED
                    print(f"📥 [ENGINE] Successfully pushed payload for '{task_id}' into AWS SQS.")
                except KeyError as e:
                    print(f"❌ [ENGINE] Formatting Error on '{task_id}': missing template variable {e}")
                    self._handle_failure(task_id, f"Missing template variable: {e}", shared_in_degree, adj_list)

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
            # Transient processing error: do NOT delete message;
            # leave in SQS queue for visibility-timeout redelivery
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
        if node.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            print(f"⚠️ [ENGINE] Duplicate response received for task '{task_id}' in state {node.status.value}. Acknowledging.")
            await sqs_client.delete_message(
                QueueUrl=self.qm.response_queue_url,
                ReceiptHandle=message["ReceiptHandle"]
            )
            return True

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
            # Acknowledge and delete terminal task response from queue
            await sqs_client.delete_message(
                QueueUrl=self.qm.response_queue_url,
                ReceiptHandle=message["ReceiptHandle"]
            )
        return True

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
            # Seed the initial tasks into SQS
            await self._publish_ready_tasks(shared_in_degree, sqs, adj_list)

            while (self.completed_count + self.failed_count + self.cancelled_count) < target_count:
                # Poll the Response queue for results from Lambdas
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
        print("🏁 [ENGINE] Cloud-Native pipeline finished execution. Dismantling simulated resources.")


if __name__ == "__main__":
    orchestrator = DAGOrchestrator()
    orchestrator.load_from_json("workflow.json")
    asyncio.run(orchestrator.run())
