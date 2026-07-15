shared_context = {

    "employee_id": "EMP1001",

    "department": "IT",

    "current_ticket": "INC1002"

}

# Agent 1

shared_context[
    "policy"
] = "Leave Policy"

print(
    "Agent 1 Updated Context:"
)

print(shared_context)

# Agent 2

print(
    "\nAgent 2 Access:"
)

print(
    shared_context["policy"]
)