## Quick try of pgvector

Set your [jina key](https://jina.ai/api-dashboard/embedding) (free key is good enough) in `.env` as `TOKEN_JINA=<your key>`, then:

```bash
# Prepare a few embeddings
get_sample_embeddings_jina.sh
```

Start postgresql with [pgvector](https://github.com/pgvector/pgvector) extension:
```bash
# Through docker compose:
make up
make pg_ping

# When you're done
make down
```

Prepare the env with uv and launch the script:
```bash
make setup script
```

Should show the following:

<img src="https://github.com/user-attachments/assets/dc634d97-2bbe-4a68-98e9-7f94ecf07259" width="650em">
