# Agent Orchestration

Agents are LLM-powered systems that can take actions by calling external tools in a loop until a task is complete.

## ReAct Pattern

The most common agent pattern: Reason, Act, Observe, Repeat.

1. **Reason**: The LLM thinks about what to do next.
2. **Act**: It calls a tool (search, API, database query, etc.).
3. **Observe**: The tool result is fed back to the LLM.
4. **Repeat**: The LLM decides whether to call another tool or return a final answer.

## Tool Calling

Modern LLM APIs support structured tool calling:

1. Define tools with a name, description, and JSON schema for inputs.
2. The model returns a `tool_use` block with the tool name and arguments.
3. Your code executes the tool and returns a `tool_result`.
4. The model continues with the result.

## Guardrails

Production agents need safety limits:

- **Max steps**: Prevent infinite loops (typically 5-10 steps).
- **Max cost/tokens**: Budget ceiling per request.
- **Tool allowlist**: Only expose necessary tools.
- **Output validation**: Check outputs against schemas or content policies.

## Frameworks

- **LangGraph**: State-machine approach. Nodes are functions, edges are transitions. More predictable than free-form loops.
- **CrewAI**: Multi-agent collaboration framework.
- **AutoGen**: Microsoft's multi-agent framework.

## Why State Machines Beat Free-Form Loops

Free-form agent loops are unpredictable and hard to debug. State machines (like LangGraph) make transitions explicit, enable better error handling, and produce reproducible workflows.
