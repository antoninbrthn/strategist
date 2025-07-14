import json
import os


class Conversation:
    def __init__(self, convo_file):
        self.convo_file = convo_file
        self.messages = []
        self._load_or_create_conversation()

    def _load_or_create_conversation(self):
        if os.path.exists(self.convo_file):
            self._load_conversation()
        else:
            pass  # don't create empty folder
            # self._save_conversation()

    def _load_conversation(self):
        with open(self.convo_file, "r") as file:
            self.messages = json.load(file)

    def _save_conversation(self):
        os.makedirs(os.path.dirname(self.convo_file), exist_ok=True)
        with open(self.convo_file, "w") as file:
            json.dump(self.messages, file, indent=4)

    def add_message(self, role, content, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})
        self._save_conversation()
