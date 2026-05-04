# RAG and Fine-Tuning

RAG means retrieval-augmented generation. In RAG, the application searches a
knowledge base for relevant chunks and passes those chunks into the model prompt.
The model does not permanently learn the information; it uses the retrieved
context for the current answer.

Use RAG when:
- the knowledge changes often;
- the data is factual documentation, product information, notes, manuals, FAQs,
  policies, or project-specific context;
- the system must cite or inspect source material;
- the model is small and cannot memorize everything;
- updates must be cheap and fast.

Fine-tuning means changing model weights with training examples. It is better
for style, classification behavior, specialized formats, tool-use patterns, or
domain-specific phrasing. It is usually not the best way to store large factual
knowledge because changing or deleting facts requires another training cycle.

A practical AI system can use both:
- RAG for facts and documents;
- prompt rules for behavior;
- fine-tuning for stable output patterns after enough examples are collected.

For Napi on 2 GB RAM, RAG is the safest first choice. Store many small chunks on
SSD, search with SQLite FTS5, and insert only the top matching chunks into the
system prompt.

