import streamlit as st
from src.config import load_config
from src.domain import load_domain_information
from src.llm import BankingChatbot
from src.prompts import build_basic_prompt, build_domain_prompt

config = load_config()
domain = load_domain_information()

st.set_page_config(
    page_title=config["app"]["title"],
    page_icon=config["app"]["page_icon"],
    layout="centered",
)

st.title(f'{config["app"]["page_icon"]} {config["app"]["title"]}')
st.caption("Experiment 7 — Prompting + Domain Information + Pre-trained LLM")

with st.sidebar:
    st.subheader("Experiment")
    st.write("**Domain:** Banking Customer Support")
    st.write("**Approach:** Prompt-based chatbot")
    st.write("**Model:**", config["model"]["name"])
    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Example questions")
    examples = [
        "How do I reset my banking password?",
        "What should I do if my debit card is lost?",
        "Why was my card payment declined?",
        "How can I get my account statement?",
        "Can I add a beneficiary?",
    ]
    for q in examples:
        st.caption("• " + q)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a banking customer-support question...")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    try:
        bot = BankingChatbot(config)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                answer = bot.answer(
                    question,
                    domain,
                    st.session_state.messages[:-1],
                )
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as exc:
        st.error(f"Unable to generate a response: {exc}")
        st.info("Check HF_TOKEN, model availability, internet access, and the selected model in config.yaml.")
