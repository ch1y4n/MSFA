AgentProxy
==========

AgentProxy is a transparent FastAPI reverse proxy for observing unfiltered
special/control tokens in requests sent from an Agent to an open-model provider.

`agents_50_list.csv` records the 50 open-source agent projects used in the
dynamic evaluation. Project names and repository URLs are replaced with stable
anonymous identifiers for review; categories and disclosure statuses are retained.

It records `filter_miss` evidence when a raw target-model special token appears
in provider-bound semantic prompt text. Optional provider validation can later
probe whether the provider/tokenizer splits that token into normal pieces or
treats it as an effective special token.

## Development

This project is managed with `uv`.

```bash
uv run agentproxy serve --provider-base-url http://localhost:8000
uv run agentproxy serve --provider-base-url http://localhost:8000 --validate-provider
uv run agentproxy serve --config agentproxy.yaml
uv run agentproxy report --summary
uv run agentproxy fetch-tokenizer meta-llama/Llama-3.1-8B-Instruct
uv run pytest
```

Default rules live in `config/rules`.

For gated Hugging Face models, set `HF_TOKEN` in your shell before running
`fetch-tokenizer`. The token is read from the environment and is not stored by
AgentProxy.

## Configuration

`agentproxy serve` reads `agentproxy.yaml` by default when the file exists.
You can also pass `--config path/to/config.yaml`. CLI flags override the config
file, and `AGENTPROXY_PROVIDER_BASE_URL` overrides the file for the provider URL.

```yaml
provider_base_url: "http://localhost:8000"
host: "127.0.0.1"
port: 10018
rules_dir: "config/rules"
logs_dir: "logs"
log_response_body: true

provider_validation:
  enabled: false
  mode: "delta_probe"
  async: true
  max_tokens_per_request: 20
  concurrency: 1
  probe_max_tokens: 1
  disable_thinking: true
  extra_body: {}
  chat_completions_path: "/chat/completions"
  timeout_seconds: 60.0
```
