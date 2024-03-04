Experiments with [llamafile](https://github.com/Mozilla-Ocho/llamafile) for a simple search autocompletion mockup.

This is not a smart way to do completion: language models are already next word predictors! Even for typo correction, it could be handled with some backtracking. A big limitation being that those models do not see characters, only tokens. This is just for fun

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

CapybaraHermes-2.5-Mistral-7B. Much better, if of course slower. [OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) seems similar, I can't tell apart the results. The slightly lighter `Q4_K_M` variants seem good enough too.

```bash
# wget -O models/capybarahermes-2.5-mistral-7b.Q5_K_M.gguf "https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/resolve/main/capybarahermes-2.5-mistral-7b.Q5_K_M.gguf?download=true"
wget -O models/openhermes-2.5-mistral-7b.Q5_K_M.gguf "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q5_K_M.gguf?download=true"
./llamafile.exe -m models/openhermes-2.5-mistral-7b.Q5_K_M.gguf -ngl 999 --n-predict 64 --no-display-prompt --escape --repeat-penalty 1.0 --no-penalize-nl --mirostat 2 -p "$(cat prompts/chatml.txt)" 2>/dev/null
```

---

LLaMa-2-7b, good results, but same inconvenience as Phi-2 for controllability

```bash
wget -O models/llama-2-7b.Q4_K_M.gguf "https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/resolve/main/capybarahermes-2.5-mistral-7b.Q5_K_M.gguf?download=true"
./llamafile.exe -m models/llama-2-7b.Q4_K_M.gguf -ngl 999 --n-predict 64 --escape --mirostat 2 -p "$(cat prompts/phi2.txt)" 2>/dev/null
```

---

[Beyonder-4x7B-v2](https://huggingface.co/mlabonne/Beyonder-4x7B-v2)

Somehow run on my M1 mac, nice!
```bash
wget -O models/beyonder-4x7b-v2.Q3_K_M.gguf "https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/resolve/main/capybarahermes-2.5-mistral-7b.Q5_K_M.gguf?download=true"
./llamafile.exe -m models/beyonder-4x7b-v2.Q3_K_M.gguf -ngl 999 --n-predict 64 --escape --mirostat 2 -p "$(cat prompts/chatml.txt)" 2>/dev/null
```

---

[Gemma-7B](https://huggingface.co/google/gemma-7b-it) [gguf](https://huggingface.co/google/gemma-7b-GGUF)

Needed to recompile llamafile with latest cosmocc ([issue](https://github.com/Mozilla-Ocho/llamafile/issues/269)) but now runs fine. 
```bash
wget -O models/gemma-7b-it.Q5_K_M.gguf "https://huggingface.co/second-state/Gemma-7b-it-GGUF/resolve/main/gemma-7b-it-Q5_K_M.gguf?download=true"
./llamafile.exe -m models/gemma-7b-it.Q5_K_M.gguf -ngl 999 --n-predict 64 --escape --ctx-size 0 --mirostat 2 -p "$(cat prompts/phi2.txt)" 2>/dev/null
```
The results are quite bad though. Might be something wrong with the default params or sampling, as the model is different enough and quite new? The result is _not_ totally non-sensical though, it just adds comments and "escape" the task. Like:
```
ski goggles 20-dollar price range.  (This query has a typo in the word "adutL")

Ski GOGGLES ARE NOT SUGGESTED BECAUSE THE QUERY HAS A TYPO AND DOESN'T MATCH ANY PRODUCTS AVAILABLE ON OUR MARKETPLACE

**Please provide me with an explanation of why
```

[Threatening the model](https://minimaxir.com/2024/02/chatgpt-tips-analysis/) does get rid of the meta comments characters, but the model still rambles instead of completing.

Instead, [RTFM](https://huggingface.co/google/gemma-7b-it/discussions/38#65d7b14adb51f7c160769fa1) (or at least online discussion, which is as close as we'll get), actually works. Who would have guessed!

```bash
# Acutally keemping the temperature high seems better for autocompletion purpose? More diversity
./llamafilerec -m models/gemma-7b-it.Q5_K_M.gguf -ngl 999 --no-display-prompt --n-predict 64 --escape --ctx-size 0 --temp 0.7 --repeat-penalty 1.0 --no-penalize-nl -p "$(cat prompts/gemma.txt)" 2>/dev/null
```

It works, but the model mainly stick to short words. Even at high temperatures somehow. Still decent. Not as good at counting to five compared to other 7B models (often returns 6 results).
