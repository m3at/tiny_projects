#!/usr/bin/env python3

import json
from pathlib import Path


def main() -> None:
    t = Path("./prompt_chatml.txt").read_text()
    Path("./prompt_chatml.json").write_text(
        json.dumps(
            {
                "prompt": t,
                "n_predict": 64,
                "cache_prompt": True,
                # "cache_prompt": False,
                # "n_probs": 3,
                "mirostat": 2,
                "top_k": 40,  # default
                # "top_k": 10,
                # "temperature": 0.8,  # default
            }
        )
    )


if __name__ == "__main__":
    main()
