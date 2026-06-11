# Leviathan: Distributed Cloud-Native DAG Orchestrator 🌊

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Terraform](https://img.shields.io/badge/Terraform-IaC-8A2BE2.svg)
![AWS SQS](https://img.shields.io/badge/AWS-SQS-FF9900.svg)
![Docker|118](https://img.shields.io/badge/Docker-LocalStack-2496ED.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg)

**Leviathan** is a production-grade, asynchronous Multi-Agent workflow engine built in Python. It demonstrates advanced enterprise system design by executing Directed Acyclic Graphs (DAG) concurrently across a decoupled, serverless AWS architecture.

---

## 🏗️ System Architecture

Leviathan completely decouples the **Orchestrator** (The Brain) from the **AI Execution Workers** (The Muscle) using an event-driven message broker pattern, managed via Infrastructure as Code (IaC).

```mermaid
graph TD
    subgraph Control Plane
        O[DAG Orchestrator]
    end

    subgraph AWS Emulation Layer LocalStack
        TQ[(SQS: Task Queue)]
        RQ[(SQS: Response Queue)]
    end

    subgraph Compute Plane
        L1[Lambda Worker 1]
        L2[Lambda Worker 2]
        L3[Lambda Worker 3]
    end

    subgraph External APIs
        G[Google Gemini 2.5 API]
    end

    O -->|1. Publish Unlocked Tasks| TQ
    TQ -->|2. Async Polling| L1
    TQ -->|2. Async Polling| L2
    TQ -->|2. Async Polling| L3
    
    L1 <-->|3. LLM Inference| G
    L2 <-->|3. LLM Inference| G
    L3 <-->|3. LLM Inference| G

    L1 -->|4. Push Results| RQ
    L2 -->|4. Push Results| RQ
    L3 -->|4. Push Results| RQ

    RQ -->|5. Acknowledge & Resolve DAG| O
````

### Core Infrastructure Pillars

- **Infrastructure as Code (IaC):** The cloud data center is explicitly provisioned using **Terraform** (`main.tf`), ensuring immutable, reproducible, and self-documenting deployments.
    
- **Message Broker:** Employs **AWS SQS** (via LocalStack) to manage isolated Task and Response queues, preventing data loss and ghost messages during worker failures via Visibility Timeouts.
    
- **Worker Pool:** Asynchronous workers simulate horizontally scaled **AWS Lambda** functions, processing isolated AI workloads concurrently via the `aioboto3` asynchronous SDK.
    
- **CI/CD Pipeline:** Fully automated **GitHub Actions** workflows validating Python code quality (`flake8`) and Terraform syntax (`terraform validate`) on every push.
    

## 🧠 Algorithmic Foundations

Leviathan is built on strict mathematical and distributed systems foundations to ensure deterministic execution.

### 1. Topological Sorting (Deadlock Prevention)

Before any execution begins, the engine maps the workflow schema to a Directed Acyclic Graph. It utilizes **Kahn’s Algorithm** to calculate in-degrees and dynamically unlock nodes, guaranteeing no cyclic dependencies or infinite loops exist.

$$\mathcal{O}(|V| + |E|)$$

_Where $\mathbf{V}$ represents the task nodes and $\mathbf{E}$ represents the dependency edges._

### 2. Transient Fault Tolerance (Resiliency)

Cloud environments are inherently unstable. The worker nodes implement **Exponential Backoff with Jitter** to elegantly absorb and recover from upstream HTTP `429 (Rate Limit)` and `503 (Unavailable)` anomalies without cascading system failure.

$$T_{\text{wait}} = \left( B \times 2^n \right) + \text{Jitter}(0.1, 1.0)$$

_Where $\mathbf{B}$ is the base wait time in seconds, and $\mathbf{n}$ is the current retry attempt._

### 3. Dynamic Context Passing (Data Flow)

Downstream nodes dynamically map, format, and ingest unstructured textual outputs from multi-parent prerequisite nodes. If Node C depends on Nodes A and B, the engine automatically resolves `{A}` and `{B}` string formatting bindings prior to SQS queuing.

## Execution Instructions

### Prerequisites

- **Python 3.12+**
    
- **Docker Desktop** (for LocalStack AWS Emulation)
    
- **HashiCorp Terraform**
    
- **Google GenAI API Key**
    

### 1. Environment Setup

Clone the repository and install the required Python packages:

```
git clone [https://github.com/yourusername/project-leviathan.git](https://github.com/yourusername/project-leviathan.git)
cd project-leviathan
pip install -r requirements.txt
```

Create a `.env` file in the root directory to store your credentials safely:

```
GEMINI_API_KEY="your_api_key_here"
```

### 2. Infrastructure Provisioning (IaC)

Boot the LocalStack container to emulate the AWS environment locally:

```
docker compose up -d
```

Navigate to the infrastructure directory and provision the physical SQS queues via Terraform:

```
cd infrastructure
terraform init
terraform apply -auto-approve
cd ..
```

### 3. Run the Distributed Orchestrator

Execute the core engine. It will automatically load `workflow.json`, validate the graph, and begin distributing workloads across the emulated cloud.

```
python core.py
```

## 📂 Project Structure

```
leviathan/
├── .github/workflows/       # CI/CD Pipelines
├── infrastructure/          # Terraform IaC definitions
│   └── main.tf              # Main configuration for terraform
├── core.py                  # Distributed DAG Engine & Lambda Simulator
├── docker-compose.yml       # LocalStack container configuration
├── workflow.json            # Dynamic DAG schema input
├── requirements.txt         # Python dependencies
└── .gitignore               # To prevent pushing files like .env
```
