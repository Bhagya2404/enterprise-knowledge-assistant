tickets = {

    "INC1001": {
        "title":"VPN Issue",
        "status":"Open"
    },

    "INC1002": {
        "title":"Password Reset",
        "status":"Closed"
    },

    "INC1003": {
        "title":"Laptop Replacement",
        "status":"In Progress"
    }

}

def lookup_ticket(ticket_id):

    return tickets.get(
        ticket_id,
        {
            "title":"Not Found",
            "status":"Unknown"
        }
    )