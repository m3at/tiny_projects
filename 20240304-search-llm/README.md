Experiments with [llamafile](https://github.com/Mozilla-Ocho/llamafile) for a simple search autocompletion mockup.

This is not a smart way to do completion: language models are already next word predictors! Even for typo correction, it could be handled with some backtracking. This is just for fun

---

Setup:

To run ggml/gguf models with llamafile:
```bash
# 32Mb
curl -L -o llamafile.exe https://github.com/Mozilla-Ocho/llamafile/releases/download/0.6.2/llamafile-0.6.2
chmod +x llamafile.exe
mkdir -p llamafile models_gguf
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
## TinyLlama
./llamafile/tinyllama.llamafile -ngl 9999 --n-predict 64 --no-display-prompt --escape --temp 0.6 --top-k 100 --top-p 0.7 --p-accept 0.0 -p "'$(cat prompts/prompt_tinyllama.txt)'" 2>/dev/null
```

Results: get the task with examples, but not great, for example for "ski gogle adutl" it rarely gets "ski goggles adult" and instead often ignore ski and does "ski google ads". When it gets "goggle", the rest is better. Might be solved with better prompts?

**TODO**: loop with an other model to tune the generation parameter :D