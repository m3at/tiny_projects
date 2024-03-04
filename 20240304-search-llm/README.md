Experiments with [llamafile](https://github.com/Mozilla-Ocho/llamafile) for a simple search autocompletion mockup.

This is not a smart way to do completion: language models are already next word predictors! Even for typo correction, it could be handled with some backtracking. This is just for fun

---

Setup:

To run ggml/gguf models with llamafile:
```bash
# 32Mb
curl -L -o llamafile.exe https://github.com/Mozilla-Ocho/llamafile/releases/download/0.6.2/llamafile-0.6.2
chmod +x llamafile.exe
mkdir -p llamafile models
```

---

For quick tests TinyLlama is nice and (relatively) light. Does poorly at instruction following.
```bash
# 0.8Gb
curl -L -o llamafile/tinyllama.llamafile "https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile?download=true" && chmod +x tinyllama.llamafile
# For exploration
./llamafile/tinyllama.llamafile --server --nobrowser -ngl 9999 --parallel 2 --port 8080
```

Attempt at query completion (the format seems right, but not sure), best parames after a quick manual exploration and eyeballing the results:
```bash
# Note the twice escaped prompt, seems necessary
./llamafile/tinyllama.llamafile -ngl 9999 --n-predict 64 --no-display-prompt --escape --temp 0.6 --top-k 100 --top-p 0.7 --p-accept 0.0 -p "'$(cat prompts/tinyllama.txt)'" 2>/dev/null
```

Results: get the task with examples, but not great, for example for "ski gogle adutl" it rarely gets "ski goggles adult" and instead often ignore ski and does "ski google ads". When it gets "goggle", the rest is better. Might be solved with better prompts?

**TODO**: loop with an other model to tune the generation parameter :D

---

Phi-2: does a bit better (understands goggles), but harder to get it to stop and just give the completions. The ChatML format does help for this. Adding an explicit end word like "END" should at least help parsing.

```bash
wget -O models/phi-2.Q5_K_M.gguf "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q5_K_M.gguf?download=true"
./llamafile.exe -m models/phi-2.Q5_K_M.gguf -ngl 999 --n-predict 64 --escape --temp 0.8 --top-k 100 --top-p 0.95 --p-accept 0.0 -p "$(cat prompts/phi2.txt)" 2>/dev/null
```

---

CapybaraHermes-2.5-Mistral-7B. Much better, if of course slower

```bash
wget -O models/capybarahermes-2.5-mistral-7b.Q5_K_M.gguf "https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/resolve/main/capybarahermes-2.5-mistral-7b.Q5_K_M.gguf?download=true"
./llamafile.exe -m models/capybarahermes-2.5-mistral-7b.Q5_K_S.gguf  -ngl 999 --n-predict 64 --escape --mirostat 2 -p "$(cat prompts/chatml.txt)" 2>/dev/null
```
