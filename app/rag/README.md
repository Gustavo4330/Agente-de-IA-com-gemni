# app/rag

O assistente responde com base nos documentos desta pasta, e não só no prompt.

- `docs/`: a base de conhecimento da Nimbus Software (empresa fictícia). Cada seção `##` vira um trecho pesquisável.
  Para ensinar algo novo ao bot, basta editar ou adicionar um `.md` aqui e reiniciar a API.
- `documents.py`: lê os arquivos e divide em trechos.
- `embeddings.py`: transforma texto em vetores com o Gemini.
- `retriever.py`: guarda os vetores em memória, busca os trechos mais parecidos com a pergunta e anexa ao prompt.

Depois de mudar os documentos, rode `python -m evals.run_eval` para conferir se o bot continua acertando.
