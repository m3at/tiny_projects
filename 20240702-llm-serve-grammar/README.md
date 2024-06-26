Example of local LLM serving with [llamafile](https://github.com/Mozilla-Ocho/llamafile), [Qwen2](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF) and some basic grammar constraint.

Prepare with:
```bash
# Get the llamafile server (<30Mb):
 curl -L -o ~/.local/bin/llamafile "https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.9/llamafile-0.8.9"
# Get a model (1.2Gb):
wget -O Qwen2-1.5B-Instruct.Q6_K.gguf "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q6_k.gguf?download=true"
# Start the server with (for example):
llamafile --model Qwen2-1.5B-Instruct.Q6_K.gguf --server --nobrowser --n-gpu-layers 999 --gpu APPLE -c 0 --parallel 1 --port 8080 --fast --flash-attn
```

Run (using [jq](https://github.com/jqlang/jq) to parse the output):
```bash
./post.sh "bread and meat inside" | jq '.choices[0].message.content'
# "hotdog"
./post.sh "dog" | jq '.choices[0].message.content'
# "not hotdog"
```

![hotdog not hotdog app, Silicon Valley show](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjFyYnR4cWpsMm05bDVibzh4dHdiYzZjcndmODEzNXdldDJuOXIyYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ohzdXIKl0BjNK2g3m/giphy.gif)


https://github.com/m3at/tiny_projects/assets/3440771/d660dfc7-cb8f-4132-b30c-d2de56a12af7

