from calculator_tool import calculate
from ticket_tool import lookup_ticket
import re

question = input("Ask: ")

print("Received:", question)

# Calculator
if "calculate" in question.lower():

    expression = question.lower().replace(
        "calculate", ""
    ).strip()

    print("Result:", calculate(expression))

# Ticket Lookup
elif "INC" in question.upper():

    ticket = re.findall(
        r"INC\d+",
        question.upper()
    )

    print("Detected Ticket:", ticket)

    if ticket:
        print("Ticket:", lookup_ticket(ticket[0]))
    else:
        print("Ticket ID not found")

# RAG
else:

    print("This will go to the RAG system")