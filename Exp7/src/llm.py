import os

from huggingface_hub import InferenceClient

from .prompts import build_domain_prompt


class BankingChatbot:
    def __init__(self, config):
        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Set it using: export HF_TOKEN='your_token'"
            )

        self.config = config

        self.client = InferenceClient(
            api_key=token,
            provider="auto",
        )

        self.model = config["model"]["name"]

    def answer(self, question, domain_information, history=None):
        """
        Generate a domain-specific banking response using
        prompt-based domain knowledge.

        The model itself is NOT fine-tuned or modified.
        """

        prompt = build_domain_prompt(
            question,
            domain_information,
        )

        # Include a small amount of recent conversation context.
        # This provides conversational continuity while keeping
        # the current question as the main request.
        if history:
            recent = history[-6:]

            conversation = "\n".join(
                f'{message["role"].upper()}: {message["content"]}'
                for message in recent
            )

            prompt = (
                "RECENT CONVERSATION:\n"
                f"{conversation}\n\n"
                "CURRENT REQUEST:\n"
                f"{prompt}"
            )

        try:
            response = self.client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                max_tokens=int(
                    self.config["model"]["max_new_tokens"]
                ),
                temperature=float(
                    self.config["model"]["temperature"]
                ),
            )

            answer = response.choices[0].message.content

            if not answer:
                return (
                    "I was unable to generate a response. "
                    "Please try asking the question again."
                )

            return str(answer).strip()

        except Exception as exc:
            raise RuntimeError(
                f"Unable to generate a response: {exc}"
            ) from exc