from anthropic import Anthropic

client = Anthropic()

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the internal knowledge base for relevant information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
]


def _search_knowledge_base(query: str) -> str:
    from app.db import retrieve

    results = retrieve(query)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(
        f"[{r['doc_id']}]: {r['content']}" for r in results
    )


HANDLERS = {
    "search_knowledge_base": lambda **kwargs: _search_knowledge_base(**kwargs),
}


def run_agent(
    user_msg: str,
    system: str = "You are a helpful assistant with access to a knowledge base.",
    max_steps: int = 5,
) -> str:
    messages = [{"role": "user", "content": user_msg}]

    for _ in range(max_steps):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            text_blocks = [b for b in resp.content if hasattr(b, "text")]
            return text_blocks[0].text if text_blocks else ""

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": resp.content})

        results = []
        for tu in tool_uses:
            handler = HANDLERS.get(tu.name)
            if handler is None:
                output = f"Error: unknown tool '{tu.name}'"
            else:
                output = handler(**tu.input)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": output,
                }
            )
        messages.append({"role": "user", "content": results})

    return "Agent reached maximum step limit."
