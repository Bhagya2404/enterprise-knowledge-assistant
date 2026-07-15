memory = {}

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    if "employee id" in user_input.lower():

        emp_id = user_input.split()[-1]

        memory["employee_id"] = emp_id

        print("Assistant: Employee ID saved.")

    elif "my id" in user_input.lower():

        print(
            "Assistant:",
            memory.get(
                "employee_id",
                "No Employee ID stored."
            )
        )

    else:

        print(
            "Assistant:",
            "I remember:",
            memory
        )