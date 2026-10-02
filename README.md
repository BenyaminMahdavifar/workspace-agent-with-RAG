# Full Agent

An interactive Python coding agent for OpenAI-compatible Chat Completions APIs. It provides a command-line interface, workspace file operations, development command execution, generated Python tools, Markdown error knowledge, and local embedding-based retrieval augmented generation (RAG).

## Features

- Works with services implementing the OpenAI Chat Completions API.
- Uses a local Hugging Face embedding model for semantic retrieval.
- Stores errors and their resolutions as Markdown knowledge entries.
- Allows the agent to create, run, and repair generated Python tools. Generated tools cannot create other tools.
- Provides built-in workspace file operations and development command execution.
- Requires explicit approval for file access outside the workspace and commands outside the development-check allowlist.

## Configuration

The application reads these environment variables and interactively prompts for required values that are not set:

| Variable | Description |
| --- | --- |
| `BASE_URL` | Base URL for an OpenAI-compatible API |
| `API_KEY` | Chat service API key; entered without echo when prompted |
| `MODEL` | Chat model identifier |
| `HF_API` | Optional Hugging Face token; used to authenticate before loading the embedding model |
| `EMBEDDING_MODEL` | Hugging Face model identifier for local embeddings |

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. It is loaded locally and used to index and retrieve Markdown knowledge; no embedding endpoint from the chat service is required. `HF_API` is optional for public models and required for gated or private models.

`API_KEY` and `HF_API` are not stored in application configuration or knowledge files. Hugging Face may persist a supplied token in its own user cache, depending on the library's behavior. Set `AGENT_HOME` to change the agent data directory; the default is `~/.full-agent`.

## Install and Run

Install dependencies and run from the project directory:

```powershell
uv sync --all-extras
uv run full-agent
```

Show available command-line options:

```powershell
uv run full-agent --help
```

By default, the current terminal directory is the workspace. To select another project directory:

```powershell
uv run full-agent --workspace E:\projects\my-app
```

The same commands work in Command Prompt (CMD), for example:

```cmd
uv sync --all-extras
uv run full-agent --workspace C:\projects\my-app
```

## CLI Commands

| Command | Behavior |
| --- | --- |
| `/help` | Show available commands |
| `/status` | Show the workspace, models, generated tool count, and conversation turn count |
| `/clear` | Clear conversation history while keeping stored knowledge and generated tools |
| `/exit` or `/quit` | Exit the agent |

The agent can list, read, and write workspace files and run development or build commands. File paths are resolved before access; access outside the active workspace requires explicit approval.

Commands are executed as argument arrays with the shell disabled, so shell pipes and command chaining are not supported. Recognized test and build commands run without approval. Other commands, destructive options, paths outside the workspace, and operations that may affect remote services require explicit approval. Only `y` or `yes` approves an action; an empty response, any other response, or end-of-input denies it. The CLI reports progress while retrieving knowledge and working with files or commands. Command execution has a timeout, and large outputs are truncated.

The workspace boundary is enforced for direct file operations. Project tests and build scripts run with the operating-system permissions of the current user; they are not isolated in an operating-system sandbox and may access files outside the workspace. Approval prompts do not replace sandboxing.

## Generated Tools

The model uses an internal JSON protocol to request tool creation, execution, or repair. Only the agent can process these requests; generated Python tools do not have an API or permission to create or register other tools. A generated tool must define `run(input_data)`. Static validation, an import allowlist, restricted built-ins, input/output size limits, and a subprocess timeout are applied. When a tool is invalid or fails during execution, the agent records the error, can repair the tool, and validates or retests it before reuse.

**Security limitation:** AST validation and a subprocess do not provide a secure sandbox against hostile code or interpreter/operating-system vulnerabilities, and no operating-system memory limit is enforced. Run untrusted code in a separate VM or operating-system sandbox.

## Error Knowledge and RAG

Each detected error is first saved as a Markdown file with an `unresolved` status. After it is successfully resolved, the solution is added to that same file. Markdown files are indexed with the local embedding model and relevant entries are retrieved for later requests. Embeddings are stored in a local SQLite database and refreshed when a Markdown file changes.

## Tests

Run the test suite with:

```powershell
uv run pytest
```
