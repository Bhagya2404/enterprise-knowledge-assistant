sensitive_actions = [

    "delete",
    "terminate",
    "salary correction",
    "access escalation",
    "reset account"

]

query = input("Request: ")

if any(
    action in query.lower()
    for action in sensitive_actions
):

    print("\n⚠ APPROVAL REQUIRED")
    print(
        "Request sent to manager."
    )

else:

    print("\nApproved")