from typing import List, Dict, Any, Optional, Union, NamedTuple

class Message(NamedTuple):
    """Internal representation of a message."""
    role: str
    content: str

class ParsedRequest(NamedTuple):
    """
    Structured representation of an LLM request.
    payload: The original full request body (to preserve other parameters).
    messages: The extracted and standardized list of messages.
    provider: The identified provider (e.g., 'openai', 'anthropic').
    """
    payload: Dict[str, Any]
    messages: List[Message]
    provider: str

def parse_openai_messages(messages: List[Dict[str, Any]]) -> List[Message]:
    """
    Parses OpenAI messages array into a list of Message namedtuples.
    """
    parsed = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parsed.append(Message(role=role, content=content))
    return parsed

def serialize_openai_messages(messages: List[Message]) -> List[Dict[str, str]]:
    """
    Serializes a list of Message namedtuples back to OpenAI format.
    """
    return [{"role": m.role, "content": m.content} for m in messages]

def parse_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Message]:
    """
    Parses Anthropic messages array into a list of Message namedtuples.
    """
    parsed = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content_raw = msg.get("content", "")
        
        if isinstance(content_raw, list):
            text_parts = []
            for block in content_raw:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "".join(text_parts)
        else:
            content = str(content_raw)
            
        parsed.append(Message(role=role, content=content))
    return parsed

def serialize_anthropic_messages(messages: List[Message], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Serializes standardized messages to Anthropic format.
    """
    payload = {}
    if system_prompt:
        payload['system'] = system_prompt
    
    anthropic_msgs = [
        {'role': m.role, 'content': m.content}
        for m in messages if m.role != 'system'
    ]
    payload['messages'] = anthropic_msgs
    return payload

def parse_request(payload: Dict[str, Any], host: str) -> Optional[ParsedRequest]:
    """
    Dispatcher that identifies the provider and parses the request payload.
    """
    if not payload or not isinstance(payload, dict):
        return None

    # Identify provider by host or payload structure
    if "openai" in host or "api.openai.com" in host:
        provider = "openai"
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            return None
        return ParsedRequest(
            payload=payload,
            messages=parse_openai_messages(messages),
            provider=provider
        )
    
    elif "anthropic" in host or "api.anthropic.com" in host:
        provider = "anthropic"
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            return None
        return ParsedRequest(
            payload=payload,
            messages=parse_anthropic_messages(messages),
            provider=provider
        )
    
    # Fallback: try to guess by payload structure
    if "messages" in payload and isinstance(payload["messages"], list):
        # Default to OpenAI-like parsing if we can't determine host
        return ParsedRequest(
            payload=payload,
            messages=parse_openai_messages(payload["messages"]),
            provider="generic"
        )

    return None

def serialize_request(parsed: ParsedRequest) -> Dict[str, Any]:
    """
    Reconstructs the full request payload using the modified messages.
    """
    payload = parsed.payload.copy()
    
    if parsed.provider == "openai" or parsed.provider == "generic":
        payload["messages"] = serialize_openai_messages(parsed.messages)
    
    elif parsed.provider == "anthropic":
        # Extract system prompt from messages if present
        system_prompt = None
        for m in parsed.messages:
            if m.role == "system":
                system_prompt = m.content
                break
        
        anthropic_payload = serialize_anthropic_messages(parsed.messages, system_prompt)
        payload.update(anthropic_payload)
        
    return payload
