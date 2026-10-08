# Demonstration document — NOT official SPIT information

This file is an illustrative, synthetic document included only to make the RAG pipeline testable before official documents are added. It does not state institutional policy or verified SPIT facts.

A Retrieval-Augmented Generation system has two main stages. First, it retrieves relevant text chunks from a controlled knowledge base using embeddings and similarity search. Second, it provides those chunks to a language model so that the generated answer can be grounded in evidence. The system should show retrieved source names and similarity scores. If the relevant evidence is absent, the assistant should say that the information is unavailable rather than inventing an answer.

The experiment investigates the effect of the number of retrieved chunks, called top-k. Small top-k values can omit relevant evidence. Larger top-k values can introduce irrelevant information, so increasing k does not always improve answer quality. Chunk size and overlap also affect whether a retrieved passage contains enough context. Source metadata is retained to support auditability and citation.
