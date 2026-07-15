def calculate(expression):

    try:

        return str(
            eval(expression)
        )

    except:

        return "Invalid Calculation"