# Nimbus Assistant

API de chatbot de atendimento em **Python + FastAPI**, com respostas do **Gemini** transmitidas token a token via **Server-Sent Events (SSE)**.
O assistente responde a partir dos documentos da empresa (**RAG**), **cita a fonte** e diz quando não sabe, e isso é **medido** por uma avaliação automática.

> Projeto de portfólio. A "Nimbus Software" é uma empresa **fictícia**, criada só para dar contexto ao assistente.

**Avaliação atual: 15/15 perguntas aprovadas.** A busca encontra o documento certo em 12/12, a resposta traz a informação esperada em 12/12 e cita a fonte em 12/12; nas 3 perguntas sem resposta nos documentos, o bot admite não saber e não inventa. Veja [`evals/results.md`](evals/results.md).

---

## O problema

Um chatbot de site precisa de mais do que "chamar um LLM":

- **Percepção de velocidade:** esperar a resposta inteira leva vários segundos, e o streaming mostra o texto conforme ele é gerado.
- **Segurança:** a chave da API não pode ir para o navegador, e o cliente não pode injetar instruções de sistema.
- **Custo e abuso:** entradas gigantes ou rajadas de requisições custam dinheiro; precisam ser barradas **antes** de chegar ao modelo.
- **Falhas reais:** o provedor pode limitar, demorar ou cair no meio de uma resposta, e o usuário precisa de uma mensagem clara em cada caso.
- **Respostas confiáveis:** o bot deve responder com fatos da empresa, não com o que o modelo "acha", e admitir quando não sabe.
- **Evolução:** trocar de provedor não deve exigir reescrever as rotas.

## Arquitetura

```
 Navegador (app/static/index.html)
   │  POST /chat  {"messages": [...histórico...]}
   ▼
┌──────────────────────────── FastAPI ─────────────────────────────┐
│ middleware.py              request-id · access log · limite body │
│ CORS (allowlist)                                                 │
│                                                                  │
│ POST /chat (routes.py)                                           │
│   1. Pydantic (schemas.py)       estrutura, roles user/assistant │
│   2. Rate limit por IP           rate_limit.py                   │
│   3. validate_conversation       turnos, tamanho, última = user  │
│   4. RAG: busca 3 trechos        app/rag → anexa ao prompt       │
│   5. stream_with_timeouts        1º token · ociosidade · total   │
│   6. "prime" do 1º token         falha cedo → HTTP 429/502/504   │
│   7. StreamingResponse           event: token | error | done     │
└──────────────────────────────┬───────────────────────────────────┘
                               │ LLMProvider.stream(system_prompt + trechos, messages)
                   ┌───────────┴────────────┐
                   │ app/llm/base.py        │  interface + erros normalizados
                   ├────────────────────────┤
                   │ app/llm/gemini.py      │  google-genai (async streaming)
                   │ (tests) FakeProvider   │  roteirizado, sem rede
                   └────────────────────────┘
          app/prompts/system_prompt.md  →  prompt de sistema (só no servidor)
          app/rag/docs/*.md             →  base de conhecimento da empresa
```

### Como o RAG funciona

1. **Na inicialização**, os arquivos de `app/rag/docs/` são divididos em trechos (um por seção `##`, 15 no total) e cada trecho vira um vetor com o `gemini-embedding-001`. Os vetores ficam em memória.
2. **A cada mensagem**, a pergunta do usuário também vira um vetor, e os 3 trechos mais parecidos (similaridade de cosseno) são anexados ao prompt de sistema com o nome do arquivo.
3. **O prompt instrui o modelo** a usar só esses trechos como fonte de fatos, a terminar com `Fonte: arquivo.md` e a dizer que não sabe quando a informação não estiver lá.

Se a busca falhar ou passar de `RAG_TIMEOUT`, o bot responde só com o prompt base; o chat nunca cai por causa do RAG.
Com poucos documentos, um banco vetorial seria peso desnecessário. Com milhares de trechos, a troca natural é por pgvector ou Chroma, mantendo a mesma classe `Retriever`.

### Estrutura

```
app/
  main.py            monta a aplicação (configuração, middleware, rotas)
  routes.py          as rotas: /, /health e /chat
  chat_service.py    limites, timeouts e formato SSE (sem dependência de HTTP)
  errors.py          formato único de erro {"error": {...}}
  schemas.py         modelos Pydantic de entrada e saída
  config.py          configurações via variáveis de ambiente (pydantic-settings)
  llm/               interface LLMProvider + implementação Gemini
  prompts/           prompt de sistema em arquivo
  middleware.py      request-id, access log, limite de tamanho do corpo
  logging_config.py  logs em JSON
  rate_limit.py      limite de requisições por IP (em memória)
  static/index.html  front mínimo que consome o SSE
  rag/
    docs/            documentos da empresa fictícia (a base de conhecimento)
    documents.py     divide os documentos em trechos
    embeddings.py    transforma texto em vetores (Gemini)
    retriever.py     busca os trechos mais parecidos e monta o contexto
evals/
  dataset.json       15 perguntas com o que a resposta deve (ou não) conter
  run_eval.py        faz as perguntas ao bot real e gera o relatório
  results.md         último resultado
tests/               pytest com LLM e embeddings falsos (sem internet, sem chave)
```

### Contrato do stream

`POST /chat` com `{"messages": [{"role": "user" | "assistant", "content": "..."}]}`. A última mensagem deve ser do usuário.

```
event: token
data: {"text":"Olá"}

event: token
data: {"text":", como posso ajudar?"}

event: done
data: {}
```

Se algo falhar **depois** que o stream começou, chega `event: error` com `{"code": "...", "message": "..."}` no lugar de `done`.
Falhas **antes** do primeiro token voltam como JSON com status HTTP real:

| Status | `error.code`                                   | Quando                                         |
|-------:|-----------------------------------------------|------------------------------------------------|
| 413    | `payload_too_large`                           | corpo maior que `MAX_BODY_BYTES`               |
| 422    | `invalid_request`                             | JSON inválido, role não permitido, campo extra |
| 422    | `too_many_turns` / `message_too_long` / `conversation_too_long` / `last_message_not_user` | limites de conversa |
| 429    | `rate_limited`                                | limite por IP ou limite do provedor            |
| 502    | `upstream_unavailable`                        | provedor fora / erro de rede                   |
| 503    | `not_configured`                              | chave ausente ou inválida                      |
| 504    | `timeout`                                     | primeiro token não chegou a tempo              |

## Como rodar localmente

Requisitos: Python 3.11+.

```bash
python -m venv .venv
```

```bash
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

```bash
cp .env.example .env    # depois coloque sua GEMINI_API_KEY no .env
```

```bash
uvicorn app.main:get_app --factory --reload
```

Abra http://localhost:8000 para o front de teste, ou http://localhost:8000/docs para o OpenAPI.
Link para teste com deploy feito via render: https://nimbus-assistant.onrender.com/

Teste o stream pelo terminal:

```bash
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Quais serviços vocês oferecem?"}]}'
```

Sem chave, a API sobe normalmente: `/health` responde e `/chat` devolve `503 not_configured`.

## Como rodar com Docker

```bash
docker build -t nimbus-assistant .
```

```bash
docker run --rm -p 8000:8000 --env-file .env nimbus-assistant
```

A imagem roda como usuário sem privilégios, tem `HEALTHCHECK` em `/health` e não contém o `.env` (ver `.dockerignore`). A chave entra só em tempo de execução.

## Testes

```bash
pytest
```

Os testes usam um `FakeProvider` roteirizado e não precisam de internet nem de chave. Eles cobrem:

- **Streaming:** ordem dos tokens, evento `done`, cabeçalhos SSE, quebras de linha e Unicode dentro dos tokens, resposta vazia.
- **Validação:** payloads malformados, role `system` vindo do cliente, campos extras, limites de tamanho/turnos/total, última mensagem do usuário, corpo grande demais, rate limit. Todos verificam que o LLM **não** foi chamado.
- **Erros:** falhas do provedor antes do 1º token (status HTTP) e no meio do stream (evento `error` depois dos tokens parciais), timeout de primeiro token, de ociosidade e total, app sem chave.
- **Privacidade dos logs:** os logs são JSON, têm `request_id` e **não** contêm o texto da mensagem, da resposta nem do prompt.
- **Gemini sem rede:** conversão de roles (`assistant` → `model`), mapeamento de erros do SDK e de rede, envio do `system_instruction`, com um cliente falso injetado.
- **RAG:** divisão dos documentos em trechos, ranking da busca, montagem do contexto, documentos chegando ao modelo, e o chat seguindo normalmente quando a busca falha ou demora. Usa um embedder falso baseado em palavras.
- **Regras da avaliação:** leitura da linha de fonte e as checagens de cada tipo de pergunta.

## Avaliação

Os testes acima garantem que o código funciona. A avaliação mede se o **bot responde bem**, usando o Gemini de verdade:

```bash
python -m evals.run_eval
```

São 15 perguntas em [`evals/dataset.json`](evals/dataset.json):

- **12 com resposta nos documentos.** Para cada uma checamos: a busca trouxe o documento certo? A resposta tem a informação esperada (ex.: `"90 dias"`)? Ela cita o arquivo certo?
- **3 sem resposta nos documentos** (CNPJ, número de funcionários, fundador). O certo é admitir que não sabe e **não inventar** o dado. Cada pergunta traz um padrão do dado inventado (ex.: um formato de CNPJ) que não pode aparecer na resposta.

O relatório vai para [`evals/results.md`](evals/results.md), com as respostas reprovadas transcritas para análise.
Falhas do provedor (ex.: limite de requisições do plano gratuito) são contadas à parte e não entram como erro do bot. O script espaça as chamadas para caber na cota gratuita, então leva cerca de 1 minuto.

As checagens são simples de propósito (palavras-chave e expressões regulares), fáceis de ler e sem custo extra. Uma evolução seria usar um segundo modelo como juiz (*LLM-as-a-judge*) para avaliar respostas mais abertas.

## Decisões técnicas

- **Servidor sem estado.** O histórico vem do cliente a cada requisição, o que permite escalar horizontalmente sem sessão. Em troca, os limites de turnos e de tamanho total precisam ser aplicados no servidor.
- **`system` é exclusivo do servidor.** O schema só aceita `user` e `assistant` (`extra="forbid"`), e o prompt de sistema fica num arquivo em `app/prompts/`, enviado como `system_instruction`. Assim o cliente não injeta instruções e o prompt pode ser revisado sem mexer no código.
- **Interface `LLMProvider` com erros normalizados.** As rotas não conhecem o SDK do Gemini. Cada provedor traduz seus erros para `LLMRateLimitError`, `LLMTimeoutError` etc., que já carregam status HTTP e mensagem pública. Trocar de provedor (ou usar um fake nos testes) é escrever uma classe nova.
- **Contrato SSE próprio** (`token` / `error` / `done`) em vez de repassar o formato do provedor. O front fica desacoplado do modelo, e erros no meio do stream têm uma forma explícita de chegar ao cliente.
- **"Prime" do primeiro token.** A rota espera o primeiro trecho antes de abrir o `StreamingResponse`. Rate limit, chave inválida e timeout inicial (os casos mais comuns) viram status HTTP de verdade em vez de um `200` seguido de erro.
- **Três timeouts** (primeiro token, ociosidade entre tokens, duração total) com `asyncio.wait_for`. O iterador do provedor é sempre fechado (`aclose`), inclusive quando o cliente desconecta, para não deixar conexões penduradas com o provedor.
- **Validação antes do LLM.** Tudo o que é barato de checar é checado antes de gastar tokens.
- **Logs estruturados sem conteúdo.** São registrados apenas metadados: turnos, caracteres, chunks, duração, resultado e `request_id`. Os loggers do `httpx` ficam em `WARNING` para não vazar URLs ou cabeçalhos. A chave é `SecretStr` e não aparece em `repr`.
- **Código simples de propósito.** Cada arquivo tem uma responsabilidade e cabe numa tela; o middleware usa o `@app.middleware("http")` padrão do FastAPI. Por isso, o `duration_ms` do access log mede o tempo até a resposta **começar**; a duração total do stream fica no evento `chat_finished`.
- **Sem dependências desnecessárias.** O rate limit e o formatter JSON usam só a biblioteca padrão.

### Limitações conhecidas

- O rate limit é em memória e por processo. Com várias réplicas, ou atrás de proxy (onde o IP visto é o do proxy), o certo é Redis e o cabeçalho `X-Forwarded-For` de um proxy confiável.
- O limite de corpo usa `Content-Length`. Requisições *chunked* sem esse cabeçalho só são barradas depois do parsing, pelos limites de conversa.
- O front de teste mostra texto puro (sem renderizar markdown), de propósito, para não interpretar HTML vindo do modelo.

- A busca usa só a **última mensagem** do usuário. Uma pergunta de continuação vaga ("e quanto custa?") pode trazer trechos menos relevantes.
- O índice é recriado a cada inicialização (15 trechos, uma chamada de embedding). Com muitos documentos, valeria salvar os vetores em disco.

## Próximos passos

1. **Deploy público** com um link "teste agora" e um GIF no topo deste README.
2. CI com GitHub Actions rodando `pytest` a cada mudança.
3. Mais perguntas na avaliação, incluindo conversas de vários turnos.
