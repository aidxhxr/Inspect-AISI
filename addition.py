from inspect_ai.tool import tool

@tool 
def add():
    async def execute(x: int, y: int):
        """
        Add two numbers.

        Args:
            x: first number to add,
            y: second number to add 
        
        Returns: 
            The sum of two numbers
        """
        return x + y 
    return execute