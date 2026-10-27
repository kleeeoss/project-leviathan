import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core import (
    DAGOrchestrator,
    TaskNode,
    TaskStatus,
    LambdaWorker,
    ResultWrapper,
    calculate_cost
)
from models import TechSummary, ComparisonOutput, SCHEMA_REGISTRY


# ==========================================
# 1. GRAPH VALIDATION & SCHEMA CONTRACT TESTS
# ==========================================

def test_validate_graph_with_valid_structured_contracts():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("fetch_aws", "Summary AWS", [], output_schema="TechSummary"))
    orch.add_node(TaskNode("fetch_gcp", "Summary GCP", [], output_schema="TechSummary"))
    orch.add_node(TaskNode(
        "compare",
        "Compare: {fetch_aws.summary} vs {fetch_gcp.summary}",
        ["fetch_aws", "fetch_gcp"],
        output_schema="ComparisonOutput"
    ))

    order = orch.validate_graph()
    assert len(order) == 3
    assert order.index("compare") > order.index("fetch_aws")
    assert order.index("compare") > order.index("fetch_gcp")


def test_validate_graph_rejects_unknown_schema():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("task1", "Run task", [], output_schema="NonExistentSchema"))

    with pytest.raises(ValueError, match="specifies unknown output_schema 'NonExistentSchema'"):
        orch.validate_graph()


def test_validate_graph_rejects_invalid_schema_attribute():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("fetch_aws", "Summary AWS", [], output_schema="TechSummary"))
    orch.add_node(TaskNode(
        "compare",
        "Read invalid field: {fetch_aws.invalid_attribute_name}",
        ["fetch_aws"]
    ))

    with pytest.raises(ValueError, match="references attribute 'invalid_attribute_name' which does not exist in schema 'TechSummary'"):
        orch.validate_graph()


def test_validate_graph_rejects_attribute_on_untyped_dependency():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("node_a", "Raw prompt", []))
    orch.add_node(TaskNode("node_b", "Follow up: {node_a.summary}", ["node_a"]))

    with pytest.raises(ValueError, match="does not declare an output_schema"):
        orch.validate_graph()


def test_validate_graph_backward_compatibility_untyped_nodes():
    orch = DAGOrchestrator()
    orch.add_node(TaskNode("node_a", "Raw prompt", []))
    orch.add_node(TaskNode("node_b", "Follow up: {node_a}", ["node_a"]))

    order = orch.validate_graph()
    assert order == ["node_a", "node_b"]


# ==========================================
# 2. PROMPT RESOLUTION & RESULT WRAPPER TESTS
# ==========================================

def test_result_wrapper_attribute_and_string_access():
    data = {
        "service_name": "AWS Lambda",
        "summary": "Serverless event-driven compute.",
        "key_characteristics": ["Scalable", "Stateless"]
    }
    wrapper = ResultWrapper(data)

    # Dot notation access via getattr
    assert wrapper.summary == "Serverless event-driven compute."
    assert wrapper.service_name == "AWS Lambda"

    # Formatting with attribute
    formatted_attr = "Summary is: {node.summary}".format(node=wrapper)
    assert formatted_attr == "Summary is: Serverless event-driven compute."

    # Formatting with top-level object
    formatted_obj = "Full object: {node}".format(node=wrapper)
    assert "Serverless event-driven compute." in formatted_obj

    # Missing attribute raises AttributeError
    with pytest.raises(AttributeError, match="Field 'non_existent' not found"):
        _ = wrapper.non_existent


@pytest.mark.asyncio
async def test_publish_ready_tasks_resolves_schema_attributes_and_payload():
    orch = DAGOrchestrator()
    orch.qm.task_queue_url = "https://sqs.mock/task-queue"

    node_a = TaskNode("fetch_aws", "Summary AWS", [], output_schema="TechSummary")
    node_a.status = TaskStatus.COMPLETED
    node_a.result = {
        "service_name": "AWS Lambda",
        "summary": "Lambda runs code in response to events.",
        "key_characteristics": ["Serverless", "Event-driven"]
    }
    orch.add_node(node_a)

    node_b = TaskNode("compare", "AWS said: {fetch_aws.summary}", ["fetch_aws"], output_schema="ComparisonOutput")
    orch.add_node(node_b)

    mock_sqs = AsyncMock()
    shared_in_degree = {"compare": 0}
    adj_list = {"fetch_aws": ["compare"], "compare": []}

    await orch._publish_ready_tasks(shared_in_degree, mock_sqs, adj_list)

    assert node_b.status == TaskStatus.QUEUED
    mock_sqs.send_message.assert_called_once()
    call_kwargs = mock_sqs.send_message.call_args.kwargs
    sent_body = json.loads(call_kwargs["MessageBody"])

    assert sent_body["task_id"] == "compare"
    assert sent_body["prompt"] == "AWS said: Lambda runs code in response to events."
    assert sent_body["output_schema"] == "ComparisonOutput"


# ==========================================
# 3. WORKER INVOCATION, PYDANTIC VALIDATION & RETRIES
# ==========================================

@pytest.mark.asyncio
async def test_invoke_llm_structured_success():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "service_name": "AWS Lambda",
        "summary": "Serverless event-driven compute.",
        "key_characteristics": ["Auto-scaling", "Pay-per-use"]
    })
    mock_resp.usage_metadata.prompt_token_count = 50
    mock_resp.usage_metadata.candidates_token_count = 60
    mock_resp.usage_metadata.total_token_count = 110

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    worker = LambdaWorker(1, MagicMock(), client=mock_client)
    task_data = {
        "task_id": "fetch_aws",
        "prompt": "Write summary of AWS Lambda",
        "output_schema": "TechSummary"
    }

    result = await worker._invoke_llm(task_data)

    assert result["status"] == TaskStatus.COMPLETED.value
    assert isinstance(result["result"], dict)
    assert result["result"]["service_name"] == "AWS Lambda"
    assert result["usage"]["status"] == "RECORDED"
    assert result["usage"]["prompt_tokens"] == 50
    assert result["usage"]["candidates_tokens"] == 60
    assert result["usage"]["total_tokens"] == 110
    assert result["usage"]["cost_usd"] is not None


@pytest.mark.asyncio
async def test_invoke_llm_structured_validation_failure_retries_and_fails():
    mock_client = MagicMock()
    # Returns invalid JSON missing required fields across all retries
    mock_resp = MagicMock()
    mock_resp.text = '{"malformed": "missing required keys"}'
    mock_resp.usage_metadata.prompt_token_count = 40
    mock_resp.usage_metadata.candidates_token_count = 20
    mock_resp.usage_metadata.total_token_count = 60

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    worker = LambdaWorker(1, MagicMock(), client=mock_client)
    task_data = {
        "task_id": "fetch_aws",
        "prompt": "Write summary",
        "output_schema": "TechSummary"
    }

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await worker._invoke_llm(task_data)

    assert result["status"] == TaskStatus.FAILED.value
    assert result["result"] is None
    assert "validation error" in result["error"].lower()
    # Verifies cumulative tokens accumulated across all 3 attempts
    assert mock_client.aio.models.generate_content.call_count == 3
    assert result["usage"]["prompt_tokens"] == 120
    assert result["usage"]["candidates_tokens"] == 60
    assert result["usage"]["total_tokens"] == 180


@pytest.mark.asyncio
async def test_invoke_llm_missing_usage_metadata_flagged_as_unknown():
    """
    FinOps invariant: if usage_metadata is None or missing from the API response,
    explicitly flag status as UNKNOWN with None tokens and cost to avoid false $0.00 claims.
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Plain text response."
    mock_resp.usage_metadata = None  # Missing usage metadata

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    worker = LambdaWorker(1, MagicMock(), client=mock_client)
    task_data = {
        "task_id": "raw_task",
        "prompt": "Say hello"
    }

    result = await worker._invoke_llm(task_data)

    assert result["status"] == TaskStatus.COMPLETED.value
    assert result["usage"]["status"] == "UNKNOWN"
    assert result["usage"]["prompt_tokens"] is None
    assert result["usage"]["candidates_tokens"] is None
    assert result["usage"]["total_tokens"] is None
    assert result["usage"]["cost_usd"] is None


# ==========================================
# 4. FINOPS METERING & IDEMPOTENCY IN ORCHESTRATOR
# ==========================================

@pytest.mark.asyncio
async def test_orchestrator_finops_aggregation_and_idempotency():
    orch = DAGOrchestrator()
    orch.qm.response_queue_url = "https://sqs.mock/response-queue"
    orch.qm.task_queue_url = "https://sqs.mock/task-queue"

    orch.add_node(TaskNode("task_1", "Do 1", []))
    shared_in_degree = {"task_1": 0}
    adj_list = {"task_1": []}
    mock_sqs = AsyncMock()

    msg = {
        "Body": json.dumps({
            "task_id": "task_1",
            "status": TaskStatus.COMPLETED.value,
            "result": "Done 1",
            "error": None,
            "usage": {
                "status": "RECORDED",
                "prompt_tokens": 100,
                "candidates_tokens": 50,
                "total_tokens": 150,
                "cost_usd": 0.000155
            }
        }),
        "ReceiptHandle": "handle_1"
    }

    # First delivery
    await orch._process_response_message(msg, shared_in_degree, adj_list, mock_sqs)
    assert orch.completed_count == 1
    assert orch.total_prompt_tokens == 100
    assert orch.total_candidates_tokens == 50
    assert orch.total_tokens == 150
    assert orch.total_cost_usd == pytest.approx(0.000155, rel=1e-5)

    # Second delivery (duplicate SQS retry)
    mock_sqs.reset_mock()
    await orch._process_response_message(msg, shared_in_degree, adj_list, mock_sqs)

    # FinOps counters MUST NOT be incremented twice
    assert orch.completed_count == 1
    assert orch.total_prompt_tokens == 100
    assert orch.total_candidates_tokens == 50
    assert orch.total_tokens == 150
    assert orch.total_cost_usd == pytest.approx(0.000155, rel=1e-5)
    mock_sqs.delete_message.assert_called_once_with(
        QueueUrl="https://sqs.mock/response-queue",
        ReceiptHandle="handle_1"
    )


@pytest.mark.asyncio
async def test_orchestrator_propagates_unknown_finops_status():
    orch = DAGOrchestrator()
    orch.qm.response_queue_url = "https://sqs.mock/response-queue"
    orch.add_node(TaskNode("task_u", "Prompt", []))

    msg = {
        "Body": json.dumps({
            "task_id": "task_u",
            "status": TaskStatus.COMPLETED.value,
            "result": "Done",
            "error": None,
            "usage": {
                "status": "UNKNOWN",
                "prompt_tokens": None,
                "candidates_tokens": None,
                "total_tokens": None,
                "cost_usd": None
            }
        }),
        "ReceiptHandle": "handle_u"
    }

    mock_sqs = AsyncMock()
    await orch._process_response_message(msg, {"task_u": 0}, {"task_u": []}, mock_sqs)

    summary = orch.get_finops_summary()
    assert summary["status"] == "UNKNOWN"
    assert summary["total_cost_usd"] is None
    assert summary["total_tokens"] is None


@pytest.mark.asyncio
async def test_orchestrator_handles_missing_usage_in_sqs_message():
    orch = DAGOrchestrator()
    orch.qm.response_queue_url = "https://sqs.mock/response-queue"
    orch.add_node(TaskNode("task_m", "Prompt", []))

    # Message with missing usage key entirely
    msg = {
        "Body": json.dumps({
            "task_id": "task_m",
            "status": TaskStatus.COMPLETED.value,
            "result": "Done without usage key",
            "error": None
        }),
        "ReceiptHandle": "handle_m"
    }

    mock_sqs = AsyncMock()
    await orch._process_response_message(msg, {"task_m": 0}, {"task_m": []}, mock_sqs)

    summary = orch.get_finops_summary()
    assert summary["status"] == "UNKNOWN"
    assert summary["total_cost_usd"] is None
    assert summary["total_tokens"] is None
    assert orch.nodes["task_m"].usage["status"] == "UNKNOWN"


def test_calculate_cost_configurable_rates():
    with patch.dict("os.environ", {"MODEL_INPUT_PRICE_PER_M": "1.00", "MODEL_OUTPUT_PRICE_PER_M": "5.00"}):
        # 1M prompt ($1.00) + 1M output ($5.00) = $6.00
        cost = calculate_cost(1_000_000, 1_000_000)
        assert cost == 6.00

    # Default rates: $0.30 input, $2.50 output per 1M
    with patch.dict("os.environ", {"MODEL_INPUT_PRICE_PER_M": "0.30", "MODEL_OUTPUT_PRICE_PER_M": "2.50"}):
        cost_default = calculate_cost(100_000, 20_000)
        # 0.1 * 0.30 = 0.03, 0.02 * 2.50 = 0.05 => 0.08
        assert cost_default == 0.08

    # Empty or malformed rates fall back safely to defaults
    with patch.dict("os.environ", {"MODEL_INPUT_PRICE_PER_M": "", "MODEL_OUTPUT_PRICE_PER_M": "invalid_rate"}):
        cost_fallback = calculate_cost(100_000, 20_000)
        assert cost_fallback == 0.08

    # Missing tokens returns None
    assert calculate_cost(None, 100) is None
    assert calculate_cost(100, None) is None
