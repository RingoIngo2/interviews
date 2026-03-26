from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass


class ReadFileTool(Tool):
    def name(self) -> str:
        return "read_file"

    def run(self, path: str) -> str:
        with open(path) as f:
            return f.read()


class AppendFileTool(Tool):
    def name(self) -> str:
        return "append_file"

    def run(self, path: str, content: str) -> str:
        with open(path, "a") as f:
            f.write(content)
        return "Appended successfully"


import call_llm


class Agent:
    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name(): tool for tool in tools}
        self.system_prompt = "You are a helpful assistant for completing documents. Use the tools to do your job."
        self.messages = []

    def converse(self) -> dict:
        response = call_llm(self.system_prompt, self.messages, self.tools.keys())
        text = response.get("text") or f"Call tool {response['tool_call']['name']}"
        self.messages.append({"role": "assistant", "text": text})
        return text

    def run(self, prompt: str):
        self.messages.append({"role": "user", "text": prompt})
        while True:
            response = self.converse()
            tool_call = response.get("tool_call")
            if not tool_call:
                return

            tool_name = tool_call["name"]
            args = tool_call["input"]

            tool = self.tools.get(tool_name)
            if not tool:
                print(f"Unknown tool: {tool_name}")
                return

            result = tool.run(**args)
            self.messages.append({"role": "user", "text": f"Ran tool '{tool_name}' with args {args}. Result is {result}"})

if __name__ == "__main__":
    read_file_functionality = ReadFileFunctionality()
    append_file_functionality = AppendFileFunctionality()
    agent = Agent([read_file_functionality, append_file_functionality])
    agent.run("I started to draft a document. Please finish the document for me. The path to the file is /path/to/file.txt")


class Agent:
    def __init__(self, tools: list[Tool]):
     pass


    def run(self, prompt: str):
       pass