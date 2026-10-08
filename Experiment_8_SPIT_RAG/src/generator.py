import os
import requests

SYSTEM_PROMPT = """You are a careful SPIT-domain question-answering assistant. Answer ONLY using the supplied retrieved context. If the context does not contain enough evidence, say: 'I could not find this information in the available knowledge base.' Do not invent dates, policies, contacts, fees, marks, attendance, or institutional facts. Mention the source filename(s) used. The knowledge base may contain demonstration/mock documents; do not present mock content as official SPIT policy."""

def build_prompt(question: str, chunks: list[dict]) -> str:
    evidence = "\n\n".join(f"[Source: {c['source']} | Section: {c.get('page_section','')} | Similarity: {c.get('score',0):.3f}]\n{c['text']}" for c in chunks)
    return f"RETRIEVED CONTEXT:\n{evidence or '[No evidence retrieved]'}\n\nQUESTION: {question}\n\nAnswer with concise, evidence-grounded information and source filenames. If the evidence is insufficient, explicitly say so."

def generate_answer(question: str, chunks: list[dict], config: dict) -> str:
    llm = config["llm"]
    prompt = build_prompt(question, chunks)
    if llm.get("provider", "ollama") == "ollama":
        url = llm.get("base_url", "http://localhost:11434").rstrip("/") + "/api/chat"
        response = requests.post(url, json={"model": llm.get("model", "llama3.2:3b"), "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "options": {"temperature": llm.get("temperature", 0.1)}}, timeout=180)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    if llm.get("provider") == "openai_compatible":
        key = os.getenv(llm.get("api_key_env", "OPENAI_API_KEY"), "")
        if not key:
            raise RuntimeError(f"Missing API key environment variable: {llm.get('api_key_env', 'OPENAI_API_KEY')}")
        base = llm.get("base_url", "https://api.openai.com/v1").rstrip("/")
        response = requests.post(base + "/chat/completions", headers={"Authorization": f"Bearer {key}"},
            json={"model": llm.get("model", "gpt-4o-mini"), "temperature": llm.get("temperature", 0.1),
                  "max_tokens": llm.get("max_tokens", 600), "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}, timeout=180)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    raise ValueError("llm.provider must be 'ollama' or 'openai_compatible'")
