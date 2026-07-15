SENSITIVE_ACTIONS = [

    "delete",
    "terminate",
    "disable",
    "reset account",
    "salary correction",
    "access escalation"

]

def requires_approval(query):

    return any(
        action in query.lower()
        for action in SENSITIVE_ACTIONS
    )
