# Experiment 7 — Domain-Specific Generative AI Chatbot

**Course:** Generative AI (PE III) — CE413  
**Experiment:** 7  
**Domain:** Banking Customer Support  
**Framework:** Streamlit  
**LLM access:** Hugging Face Inference API  
**Architecture:** Prompting + domain information + pre-trained LLM

## Important
This experiment intentionally uses the **same banking customer-support domain** as the previous LoRA/Fine-Tuning experiment. The model is **not fine-tuned again**. Domain knowledge is supplied in the prompt, as required by the lab manual.

The Hugging Face token is never stored in source code.

## Project structure

```text
EXP7_Domain_Specific_GenAI_Chatbot/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── data/
│   ├── domain_information.yaml
│   └── evaluation_questions.yaml
└── src/
    ├── __init__.py
    ├── config.py
    ├── domain.py
    ├── prompts.py
    ├── llm.py
    └── evaluation.py
```

## Setup with uv

```bash
uv add -r requirements.txt
```

Set the Hugging Face token in the environment:

```bash
export HF_TOKEN="your_huggingface_token"
```

Then run:

```bash
uv run streamlit run app.py
```

Do **not** commit the token to GitHub.

## What the application implements

- Pre-trained instruction/chat LLM
- Banking-specific domain information in the prompt
- System-style instructions to reduce unsupported claims
- Streamlit user interface
- Conversation history
- Clear-chat control
- Example questions
- Prompt-engineering comparison: basic vs domain-specific prompt
- Additional domain facts for the second required modification
- A 10-question evaluation set:
  - 4 direct domain questions
  - 2 paraphrased questions
  - 2 multi-part questions
  - 2 out-of-domain questions

## Domain information

The supplied banking information covers common customer-support scenarios such as:
- password/PIN reset
- lost or stolen debit cards
- ATM cash issues
- unauthorized transactions
- account statements
- beneficiaries
- declined payments
- locked accounts
- transfers
- transaction notifications

These are synthetic educational policies for the experiment and are not real bank policies.

## Required experimentation

### Modification 1 — Prompt Engineering

The application can compare:
1. Basic prompt: asks the model to answer the question.
2. Domain-specific prompt: supplies banking information and explicit grounding instructions.

### Modification 2 — Additional Domain Information

Five additional synthetic facts are included in `data/domain_information.yaml`. The UI can test questions about them.

## Evaluation

Run:

```bash
uv run python -m src.evaluation
```

The script generates a prompt-based evaluation report template. Actual model responses are produced through the same Hugging Face inference service used by the application.

Because API/model responses can vary, the final correctness and hallucination judgments should be completed from the actual outputs. These results should be recorded before preparing the final lab documentation.

## Security

Use Streamlit secrets or an environment variable for `HF_TOKEN`. Never put:

```text
HF_TOKEN = "hf_..."
```

in source code or commit it to GitHub.

## CPU/GPU note

This implementation performs model inference through the Hugging Face inference service, matching the lab manual. Therefore a local PyTorch installation/GPU is **not required**. PyTorch is intentionally not added to `requirements.txt` because it would add a large unnecessary local dependency without being used by this architecture.

## Relation to other experiments

**Previous experiment:** LoRA/Fine-Tuning → model adaptation  
**Experiment 7:** Prompt + domain information → application  
**Next experiment:** RAG → retrieval + generation

Experiment 7 is not a RAG system: there is no retriever, embeddings, or vector database.


Experiment 7 — Prompting + Domain Information + Pre-trained LLM

Can I add a beneficiary?

Yes. You can add a beneficiary through the online banking beneficiary‑management section. After logging into your online banking account, navigate to the beneficiary management area, enter the required beneficiary details, and complete any additional verification that may be requested. If you encounter any issues, you can contact customer support for assistance.

How can I reset my banking password?

You can reset your banking password directly from the online‑banking login page:

Go to the bank’s online‑banking login screen.
Click “Forgot password?” (or a similar password‑reset link).
Follow the prompts—usually you’ll need to confirm your registered mobile number or email address.
You’ll receive a one‑time verification code; enter it to verify your identity.
Choose a new password that meets the required security criteria and confirm it.
If you’re unable to complete the reset because your account is locked after multiple failed login attempts, use the account‑recovery process (often a “Unlock

How do I report an unauthorized transaction?

To report an unauthorized transaction:

Act immediately – As soon as you notice the transaction, block the affected card or restrict access to the account. You can do this quickly through the banking app (or by contacting customer support).

Contact customer support – Call the bank’s support line or use the secure messaging feature in the online‑banking portal. Provide them with:

The details of the unauthorized transaction (date, amount, merchant, etc.).
Any reference number shown on the transaction record.
Confirmation that you have already
How long does a domestic bank transfer take?

The supplied domain information does not include a specific timeframe for how long a domestic bank transfer takes.

If you need an exact estimate, please refer to your bank’s terms and conditions or contact customer support directly. When making a transfer, remember to verify the recipient’s account details carefully before confirming the transaction. If you encounter any issues after the transfer is initiated, you should report the transaction to customer support and provide the transaction reference and relevant date/time details.

What is the process for adding someone as a beneficiary to my account?

Adding a Beneficiary

Log in to online banking – Use your username and password to access the portal.
Navigate to the Beneficiary‑Management section – This is usually found under “Transfers” or “Payments.”
Select “Add New Beneficiary.”
Enter the required details – Provide the beneficiary’s name, bank account number, sort code/IBAN, and any other information the form asks for.
Verify the information – Double‑check the
What should I do if I notice a transaction that I did not authorize?

If you see a transaction that you didn’t authorize, take these steps right away:

Temporarily block the card or account

Use the banking app to block the debit card, or if the issue is with the whole account, restrict access as the app allows.
**

How do I add a beneficiary, and what information do I need to provide?

Adding a Beneficiary

Log in to online banking – Use your username and password to access the portal.
Go to the Beneficiary‑Management section – Usually found under “Transfers” or “Payments.”
Select “Add New Beneficiary.”
Enter the required details – You will need to provide:
Beneficiary’s full name
Bank account number (or IBAN for international accounts)
Sort code (or equivalent
What should I do after noticing an unauthorized transaction, and how can I protect my account?

Steps to take after spotting an unauthorized transaction

Report it immediately – Contact customer‑support (or use the secure messaging feature in the banking app) and tell them about the transaction. Provide the transaction reference, date, time, and any other details you have.
Temporarily block the affected card or account –
If the transaction was made with your debit card, block the card right away through the banking app.
If you suspect the whole account may be compromised, use the app’s “lock account” option or ask support to restrict access temporarily.
What is the capital of France?

I’m sorry, but that question is outside the scope of the banking customer‑support information provided. If you have any banking‑related questions, I’d be happy to help!

Who is the CEO of Microsoft?

I’m sorry, but that question is outside the scope of the banking customer‑support information provided. If you have any banking‑related questions, I’d be happy to help!