import pytest
import json
from unittest.mock import AsyncMock
from core import DAGOrchestrator, TaskNode, TaskStatus


def test_valid_acyclic_dag_validation():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("fetch_aws", "Summary of AWS", []))
    orch.add_node(TaskNode("fetch_gcp", "Summary of GCP", []))
    orch.add_node(TaskNode("compare", "Compare {fetch_aws} and {fetch_gcp}", ["fetch_aws", "fetch_gcp"]))

    order = orch.validate_graph()
    assert len(order) == 3
    assert order.index("fetch_aws") < order.index("compare")
    assert order.index("fetch_gcp") < order.index("compare")


def test_cycle_detection():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("A", "Do A", ["C"]))
    orch.add_node(TaskNode("B", "Do B", ["A"]))
    orch.add_node(TaskNode("C", "Do C", ["B"]))

    with pytest.raises(ValueError, match="Cycle detected in DAG"):
        orch.validate_graph()


def test_unknown_dependency_detection():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("A", "Do A", []))
    orch.add_node(TaskNode("B", "Do B", ["A", "non_existent_task"]))

    with pytest.raises(ValueError, match="depends on unknown task 'non_existent_task'"):
        orch.validate_graph()


def test_template_variable_mismatch():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("A", "Do A", []))
    orch.add_node(TaskNode("B", "Process {missing_dep}", ["A"]))

    with pytest.raises(ValueError, match="not found in declared dependencies"):
        orch.validate_graph()


def test_branch_failure_isolation_parallel_branches():
    """
    Verifies that when task A fails in branch (A -> B),
    downstream task B is cascade-cancelled, but independent branch (C -> D)
    remains completely untouched and executes to completion.
    """
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("A", "Do A", []))
    orch.add_node(TaskNode("B", "Do B: {A}", ["A"]))
    orch.add_node(TaskNode("C", "Do C", []))
    orch.add_node(TaskNode("D", "Do D: {C}", ["C"]))

    shared_in_degree = {task_id: len(t.dependencies) for task_id, t in orch.nodes.items()}
    adj_list = {task_id: [] for task_id in orch.nodes}
    for task_id, task in orch.nodes.items():
        for dep in task.dependencies:
            adj_list[dep].append(task_id)

    # Simulate failure of task A
    orch._handle_failure("A", "Simulated LLM rate limit", shared_in_degree, adj_list)

    assert orch.nodes["A"].status == TaskStatus.FAILED
    assert orch.nodes["B"].status == TaskStatus.CANCELLED
    assert orch.failed_count == 1
    assert orch.cancelled_count == 1

    # Verify independent branch (C -> D) is untouched and still PENDING
    assert orch.nodes["C"].status == TaskStatus.PENDING
    assert orch.nodes["D"].status == TaskStatus.PENDING

    # Now simulate successful completion of C
    orch.nodes["C"].status = TaskStatus.COMPLETED
    orch.nodes["C"].result = "Result C"
    orch.completed_count += 1
    for neighbor in adj_list["C"]:
        shared_in_degree[neighbor] -= 1

    # D is now unlocked (in-degree 0) and completes
    assert shared_in_degree["D"] == 0
    orch.nodes["D"].status = TaskStatus.COMPLETED
    orch.nodes["D"].result = "Result D"
    orch.completed_count += 1

    # Total check: completed(2) + failed(1) + cancelled(1) == target(4)
    target_count = len(orch.nodes)
    assert orch.completed_count + orch.failed_count + orch.cancelled_count == target_count


def test_diamond_graph_cascade_cancellation():
    """
    Diamond DAG:
        A -> B -> D
        A -> C -> D
    Failure of root node A must cancel all transitive descendants B, C, D.
    """
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("A", "Root", []))
    orch.add_node(TaskNode("B", "Left branch: {A}", ["A"]))
    orch.add_node(TaskNode("C", "Right branch: {A}", ["A"]))
    orch.add_node(TaskNode("D", "Join: {B} and {C}", ["B", "C"]))

    shared_in_degree = {task_id: len(t.dependencies) for task_id, t in orch.nodes.items()}
    adj_list = {task_id: [] for task_id in orch.nodes}
    for task_id, task in orch.nodes.items():
        for dep in task.dependencies:
            adj_list[dep].append(task_id)

    orch._handle_failure("A", "Root task exploded", shared_in_degree, adj_list)

    assert orch.nodes["A"].status == TaskStatus.FAILED
    assert orch.nodes["B"].status == TaskStatus.CANCELLED
    assert orch.nodes["C"].status == TaskStatus.CANCELLED
    assert orch.nodes["D"].status == TaskStatus.CANCELLED

    assert orch.failed_count == 1
    assert orch.cancelled_count == 3
    assert orch.completed_count == 0
    assert orch.failed_count + orch.cancelled_count + orch.completed_count == len(orch.nodes)


@pytest.mark.asyncio
async def test_publish_ready_tasks_handles_formatting_keyerror():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("A", "Root", []))
    orch.add_node(TaskNode("B", "Process {non_existent_key}", []))

    mock_sqs = AsyncMock()
    shared_in_degree = {"B": 0}
    adj_list = {"B": []}

    await orch._publish_ready_tasks(shared_in_degree, mock_sqs, adj_list)

    assert orch.nodes["B"].status == TaskStatus.FAILED
    assert orch.failed_count == 1
    mock_sqs.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_sqs_message_handling_transient_error_retains_message():
    """
    Verifies that malformed message JSON is treated as a transient error:
    the message is NOT deleted from SQS so visibility timeout allows redelivery.
    """
    orch = DAGOrchestrator()
    orch.qm.response_queue_url = "https://sqs.mock/response-queue"
    orch.add_node(TaskNode("A", "Do A", []))

    shared_in_degree = {"A": 0}
    adj_list = {"A": []}
    mock_sqs = AsyncMock()

    malformed_msg = {
        "Body": "INVALID_JSON_CORRUPTED{",
        "ReceiptHandle": "handle_malformed_1"
    }

    handled = await orch._process_response_message(malformed_msg, shared_in_degree, adj_list, mock_sqs)

    assert handled is False
    mock_sqs.delete_message.assert_not_called()


@pytest.mark.asyncio
async def test_sqs_message_handling_terminal_failure_deletes_and_cancels():
    """
    Verifies that worker-reported terminal failures trigger cascade cancellation
    and delete the message from the SQS response queue.
    """
    orch = DAGOrchestrator()
    orch.qm.response_queue_url = "https://sqs.mock/response-queue"
    orch.add_node(TaskNode("A", "Do A", []))
    orch.add_node(TaskNode("B", "Do B: {A}", ["A"]))

    shared_in_degree = {"A": 0, "B": 1}
    adj_list = {"A": ["B"], "B": []}
    mock_sqs = AsyncMock()

    terminal_fail_msg = {
        "Body": json.dumps({
            "task_id": "A",
            "status": TaskStatus.FAILED.value,
            "result": None,
            "error": "Poison pill: invalid model parameters"
        }),
        "ReceiptHandle": "handle_terminal_fail_2"
    }

    handled = await orch._process_response_message(terminal_fail_msg, shared_in_degree, adj_list, mock_sqs)

    assert handled is True
    assert orch.nodes["A"].status == TaskStatus.FAILED
    assert orch.nodes["B"].status == TaskStatus.CANCELLED
    assert orch.failed_count == 1
    assert orch.cancelled_count == 1
    mock_sqs.delete_message.assert_called_once_with(
        QueueUrl="https://sqs.mock/response-queue",
        ReceiptHandle="handle_terminal_fail_2"
    )


@pytest.mark.asyncio
async def test_sqs_message_handling_terminal_success_advances_dag():
    """
    Verifies that successful task results update state, unlock dependent tasks,
    and delete the message from the SQS response queue.
    """
    orch = DAGOrchestrator()
    orch.qm.response_queue_url = "https://sqs.mock/response-queue"
    orch.qm.task_queue_url = "https://sqs.mock/task-queue"
    orch.add_node(TaskNode("A", "Do A", []))
    orch.add_node(TaskNode("B", "Do B: {A}", ["A"]))

    shared_in_degree = {"A": 0, "B": 1}
    adj_list = {"A": ["B"], "B": []}
    mock_sqs = AsyncMock()

    success_msg = {
        "Body": json.dumps({
            "task_id": "A",
            "status": TaskStatus.COMPLETED.value,
            "result": "Output from task A",
            "error": None
        }),
        "ReceiptHandle": "handle_success_3"
    }

    handled = await orch._process_response_message(success_msg, shared_in_degree, adj_list, mock_sqs)

    assert handled is True
    assert orch.nodes["A"].status == TaskStatus.COMPLETED
    assert orch.nodes["A"].result == "Output from task A"
    assert orch.completed_count == 1
    assert shared_in_degree["B"] == 0
    # B was unlocked and published
    assert orch.nodes["B"].status == TaskStatus.QUEUED
    mock_sqs.send_message.assert_called_once()
    mock_sqs.delete_message.assert_called_once_with(
        QueueUrl="https://sqs.mock/response-queue",
        ReceiptHandle="handle_success_3"
    )


@pytest.mark.asyncio
async def test_sqs_message_handling_duplicate_response_is_idempotent():
    """
    Verifies that duplicate SQS deliveries for an already-completed task
    acknowledge the message without double-incrementing counters or in-degrees.
    """
    orch = DAGOrchestrator()
    orch.qm.response_queue_url = "https://sqs.mock/response-queue"
    orch.qm.task_queue_url = "https://sqs.mock/task-queue"
    orch.add_node(TaskNode("A", "Do A", []))
    orch.add_node(TaskNode("B", "Do B: {A}", ["A"]))

    shared_in_degree = {"A": 0, "B": 1}
    adj_list = {"A": ["B"], "B": []}
    mock_sqs = AsyncMock()

    success_msg = {
        "Body": json.dumps({
            "task_id": "A",
            "status": TaskStatus.COMPLETED.value,
            "result": "Output from task A",
            "error": None
        }),
        "ReceiptHandle": "handle_success_4"
    }

    # First delivery
    await orch._process_response_message(success_msg, shared_in_degree, adj_list, mock_sqs)
    assert orch.completed_count == 1
    assert shared_in_degree["B"] == 0

    # Second (duplicate) delivery
    mock_sqs.reset_mock()
    handled = await orch._process_response_message(success_msg, shared_in_degree, adj_list, mock_sqs)

    assert handled is True
    # Counters must NOT increment again
    assert orch.completed_count == 1
    # In-degree must NOT drop below 0
    assert shared_in_degree["B"] == 0
    # Duplicate message must still be deleted from queue
    mock_sqs.delete_message.assert_called_once_with(
        QueueUrl="https://sqs.mock/response-queue",
        ReceiptHandle="handle_success_4"
    )


@pytest.mark.asyncio
async def test_sqs_message_handling_unknown_task_is_acknowledged():
    """
    Verifies that an unroutable message with unknown task_id is acknowledged
    to prevent indefinite queue head-of-line blocking.
    """
    orch = DAGOrchestrator()
    orch.qm.response_queue_url = "https://sqs.mock/response-queue"
    orch.add_node(TaskNode("A", "Do A", []))

    shared_in_degree = {"A": 0}
    adj_list = {"A": []}
    mock_sqs = AsyncMock()

    alien_msg = {
        "Body": json.dumps({
            "task_id": "alien_task_999",
            "status": TaskStatus.COMPLETED.value,
            "result": "Who am I?",
            "error": None
        }),
        "ReceiptHandle": "handle_alien_5"
    }

    handled = await orch._process_response_message(alien_msg, shared_in_degree, adj_list, mock_sqs)

    assert handled is True
    mock_sqs.delete_message.assert_called_once_with(
        QueueUrl="https://sqs.mock/response-queue",
        ReceiptHandle="handle_alien_5"
    )
