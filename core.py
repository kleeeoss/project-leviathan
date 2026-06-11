import asyncio
import os
import random
import json
import aioboto3, boto3
import uuid
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
                await asyncio.sleep(0.5)

    async def _invoke_llm(self, task_data: dict) -> dict:
        local_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = await local_client.aio.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=task_data['prompt']
                )
                return {
                    "task_id": task_data['task_id'],
                    "status": TaskStatus.COMPLETED.value,  # <-- .value prevents JSON crash
                    "result": response.text.strip(),
                    "error": None
                }
            except Exception as e:
                print(f"⚠️ [{self.worker_id}] Transient Error on '{task_data['task_id']}': {e}")
                if attempt == max_retries:
                    return {
                        "task_id": task_data['task_id'],
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

    def add_node(self, node: TaskNode):
        self.nodes[node.task_id] = node

    def load_from_json(self, filepath: str):
        with open(filepath, 'r') as file:
            data = json.load(file)
        for task_data in data.get('tasks', []):
            self.add_node(TaskNode(
                task_id=task_data['id'],
                prompt=task_data['prompt'],
                dependencies=task_data.get('dependencies', [])
            ))
        print(f"📄 [ENGINE] Loaded {len(self.nodes)} tasks from JSON.")

    async def _publish_ready_tasks(self, shared_in_degree: dict, sqs_client):
        """Publishes tasks directly into the LocalStack SQS queue via AWS SDK."""
        for task_id, degree in shared_in_degree.items():
            node = self.nodes[task_id]
            if degree == 0 and node.status == TaskStatus.PENDING:
                context = {dep: self.nodes[dep].result for dep in node.dependencies}
                try:
                    resolved_prompt = node.prompt.format(**context)
                    payload = {"task_id": task_id, "prompt": resolved_prompt}

                    # Core change: Send via AWS SDK
                    await sqs_client.send_message(
                        QueueUrl=self.qm.task_queue_url,
                        MessageBody=json.dumps(payload)
                    )

                    node.status = TaskStatus.QUEUED
                    print(f"📥 [ENGINE] Successfully pushed payload for '{task_id}' into AWS SQS.")
                except KeyError as e:
                    print(f"❌ [ENGINE] Formatting Error on '{task_id}': missing template variable {e}")
                    node.status = TaskStatus.FAILED

    async def run(self):
        print("🚀 [ENGINE] Initializing LocalStack Cloud-Native Runtime...")

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

        completed_count = 0
        target_count = len(self.nodes)

        # 4. Open Orchestrator Client connection to poll for completions
        async with self.qm.session.client("sqs", endpoint_url=LOCALSTACK_ENDPOINT) as sqs:
            # Seed the initial tasks into SQS
            await self._publish_ready_tasks(shared_in_degree, sqs)

            while completed_count < target_count:
                # Poll the Response queue for results from Lambdas
                response = await sqs.receive_message(
                    QueueUrl=self.qm.response_queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=1
                )

                if "Messages" in response:
                    for message in response["Messages"]:
                        body = json.loads(message["Body"])
                        node = self.nodes[body['task_id']]

                        # Convert the JSON string back into a Python Enum!
                        node.status = TaskStatus(body['status'])
                        node.result = body['result']
                        completed_count += 1

                        if node.status == TaskStatus.COMPLETED:
                            print(f"✅ [ENGINE] Context Loopback: Processed '{node.task_id}' from Response SQS.")
                            for neighbor in adj_list[node.task_id]:
                                shared_in_degree[neighbor] -= 1

                            # Publish newly unlocked tasks
                            await self._publish_ready_tasks(shared_in_degree, sqs)
                        else:
                            print(f"❌ [ENGINE] Fatal Failure on Node '{node.task_id}': {body['error']}")
                            break

                        # Delete read message from response queue
                        await sqs.delete_message(
                            QueueUrl=self.qm.response_queue_url,
                            ReceiptHandle=message["ReceiptHandle"]
                        )
                await asyncio.sleep(0.5)

        for wt in worker_tasks:
            wt.cancel()
        print("🏁 [ENGINE] Cloud-Native pipeline finished execution. Dismantling simulated resources.")


if __name__ == "__main__":
    orchestrator = DAGOrchestrator()
    orchestrator.load_from_json("workflow.json")
    asyncio.run(orchestrator.run())