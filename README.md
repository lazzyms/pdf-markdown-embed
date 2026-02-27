# PDF -> Markdown -> Embedding

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PDF -> Markdown -> Embedding** is a robust, open-source document and image processing pipeline. It leverages state-of-the-art tools like `docling`, `easyocr`, and `langchain` to extract, process, and convert various document formats (PDF, DOCX, Images) into Markdown. It seamlessly integrates with **pgvector** for vector storage and **MinIO** for object storage, making it an ideal backend for Retrieval-Augmented Generation (RAG) and LLM-powered applications.

---

## Features

- **Multi-format Document Processing**: Extract text and structure from PDFs, DOCX, and images using `unstructured` and `docling`.
- **OCR Capabilities**: Built-in Optical Character Recognition using `easyocr` and `torch`.
- **Vector Storage**: Store and query document embeddings efficiently using PostgreSQL with the `pgvector` extension.
- **Object Storage**: Reliable file and asset storage using self-hosted MinIO.
- **LLM Integration**: Ready-to-use LangChain and Ollama integrations for advanced text processing and embedding generation.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv) (Astral's blazing-fast Python package manager)

To install `uv`, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Getting Started

Follow these steps to set up the project locally.

### 1. Environment Setup

Create a `.env` file in the root directory to configure your database, storage, and application settings.

```bash
cp .env.example .env
```

_If `.env.example` is not present, create a `.env` file with the following essential variables:_

```env
# PostgreSQL / pgvector settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=vector_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5433

# MinIO settings
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=your_secure_password
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=your_secure_password
```

### 2. Run Docker Containers

The project relies on PostgreSQL (with pgvector) and MinIO. Start the required infrastructure using Docker Compose.

First, ensure the external Docker network exists:

```bash
docker network create app-network
```

Then, spin up the containers in the background:

```bash
docker compose up -d
```

_Note: This will build the custom `pgvector` image and start the MinIO server. MinIO console will be available at `http://localhost:9001`._

### 3. Install Dependencies

We use `uv` for fast and deterministic dependency management. Create a virtual environment and install the dependencies defined in `pyproject.toml`.

```bash
# Create a virtual environment and sync dependencies
uv sync
```

Alternatively, if you prefer to manage the virtual environment manually:

```bash
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install -e .
```

### 4. Run the Application

Once the containers are healthy and dependencies are installed, you can run the main processing pipeline:

```bash
# If using uv sync
uv run python src/main.py

# If virtual environment is activated
python src/main.py
```

---

## Project Structure

```text
md-converter/
├── docker-compose.yml      # Infrastructure orchestration
├── Dockerfile.pgvector     # Custom PostgreSQL + pgvector image
├── pyproject.toml          # Python dependencies and project metadata
├── src/
│   ├── main.py             # Application entry point
│   ├── config/             # Configuration and environment settings
│   ├── models/             # LLM factories and model definitions
│   ├── processing/         # Document and image processing logic
│   ├── storage/            # MinIO client and Vector Store integrations
│   └── utils/              # Logging and helper utilities
└── temp_files/             # Temporary directory for processing artifacts
```

---

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
