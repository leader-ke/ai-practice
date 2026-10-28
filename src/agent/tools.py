tickets = {"1234567890": "open", "1234567891": "closed", "1234567892": "on hold"}


def lookup_ticket(ticket_id):
    if ticket_id in tickets:
        status = tickets.get(ticket_id)
        return f"Ticket status is: {status}", False
    else:
        return f"Error: Ticket id {ticket_id} not found.", True


TOOL_MAPPING = {"lookup_ticket": lookup_ticket}

tools = [
    {
        "name": "lookup_ticket",
        "description": "Look up a ticket by its ID and return its status",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
    }
]


def run_tool(name, tool_input):
    """Run one tool by name and return (message, is_error). It never raises."""
    tool = TOOL_MAPPING.get(name)
    if tool is None:
        return f"Error: Tool '{name}' not implemented.", True
    if not isinstance(tool_input, dict):
        return f"Error: input for tool '{name}' must be an object.", True
    try:
        return tool(**tool_input)
    except TypeError as e:
        return f"Error: invalid input for tool '{name}': {e}", True
    except Exception as e:
        return f"System Error executing tool '{name}': {e}", True
