import re

from governance import requires_approval
from calculator_tool import calculate
from ticket_tool import lookup_ticket


def route_query(query):

    # Governance

    if requires_approval(query):

        return {

            "type": "approval",

            "response":
            """
Human Approval Required

Sensitive action detected.

Request routed for manager approval.
"""
        }

    # Calculator

    if query.lower().startswith("calculate"):

        expression = (

            query.lower()

            .replace(
                "calculate",
                ""
            )

            .strip()

        )

        return {

            "type":"calculator",

            "response":
            calculate(
                expression
            )
        }

    # Ticket

    ticket = re.findall(

        r"INC\d+",

        query.upper()

    )

    if ticket:

        data = lookup_ticket(
            ticket[0]
        )

        return {

            "type":"ticket",

            "response":
            f"""
Ticket ID : {ticket[0]}

Title : {data['title']}

Status : {data['status']}
"""
        }

    return {

        "type":"rag"

    }