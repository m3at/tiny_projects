Score function names in solidity contract for likelyhood of being nonsense. Using the key inference parts from the [nostril code](https://github.com/casics/nostril).

Setup:
```bash
# Install solidity compiler
brew tap ethereum/ethereum
brew install solidity
# Jq
brew install jq
```

Run:
```bash
./score_lines.sh CONTRACT.sol
# Or equivalently
solc --ast-compact-json CONTRACT.sol | tail -n +5 | jq -r '.nodes[] | select(.nodes) | .nodes[] | select(.kind == "function") | .name' | python check_stdin.py
```

Example of output:
```txt
 9.3 getBcQyknxmojvto
 8.1 brcFfffactoryknxmojvto
 4.8 transferFrom
 3.3 factory
```
