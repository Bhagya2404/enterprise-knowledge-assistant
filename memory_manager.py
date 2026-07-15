class MemoryManager:

    def __init__(self):

        self.memory = {}

    def save(self, key, value):

        self.memory[key] = value

    def get(self, key):

        return self.memory.get(key)

    def get_all(self):

        return self.memory