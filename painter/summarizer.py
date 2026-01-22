import ollama
from deep_translator import GoogleTranslator
from pathlib import Path


class LocalSummarizer:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

        try:
            local_models = ollama.list()
        except Exception:
            raise ConnectionError(
                "Ollama service not running. Run 'ollama serve' in a terminal."
            )

        installed_models = [m["model"] for m in local_models["models"]]
        if (
            self.model_name not in installed_models
            and f"{self.model_name}:latest" not in installed_models
        ):
            print(f"Model '{self.model_name}' not found locally.")
            print(f"Please run: ollama pull {self.model_name}")
            raise ValueError(f"Model {self.model_name} is not downloaded.")

    def translate(self, text, source, target="en"):
        return GoogleTranslator(source=source, target=target).translate(text)

    def summarize(self, text, system_prompt, lang="en"):
        if isinstance(system_prompt, (str, Path)):
            p = Path(system_prompt)
            if p.exists() and p.is_file():
                system_instruction = p.read_text(encoding="utf-8").strip()
            else:
                system_instruction = str(system_prompt).strip()
        else:
            raise TypeError("system_prompt must be a string or a path to a file")

        if lang.lower() != "en":
            text = self.translate(text, lang)

        response = ollama.generate(
            model=self.model_name,
            system=system_instruction,
            prompt=text,
            options={
                "num_ctx": 4096,
                "temperature": 0.3,
                "top_p": 0.9,
            },
        )

        return response["response"].strip()
