#!/usr/bin/env python3

import re
import argparse
from pathlib import Path

import tree_sitter_solidity as tssolidity
from tree_sitter import Language, Parser, Tree

SOLIDITY = Language(tssolidity.language(), "solidity")


def get_top_comments(tree: Tree) -> list[str]:
    # _q = """
    # (contract_declaration
    #   name: (identifier) @function.def
    #   body: (contract_body) @function.block)
    # """
    # _q = """
    # (function_definition
    #   name: (identifier) @function.def
    #   )
    # """
    # _q = """
    # ((comment)+ @comment.documentation)
    # """
    # _q = """
    # ((comment)+ @comment.documentation
    #  #eq? @comment.parent.type "source_file")
    # """

    # Comments at the top-level
    _q = """
(source_file
  (comment) @comment.documentation)
    """

    query = SOLIDITY.query(_q)
    captures = query.captures(tree.root_node)

    comments = []
    c_style_comment = re.compile(r"(^// ?|^\s*/\*\*?\s*|\s?\*?\*/|^\s*\*\s?)")

    for x in captures[:4]:
        c = x[0]
        t = c.text.decode("utf-8").splitlines()
        
        if c.is_named:
            # print(t)
            for line in t:
                if "SPDX-License-Identifier" in line:
                    continue
                if "Submitted for verification at basescan.org" in line:
                    continue

                line = c_style_comment.sub("", line)

                if len(line) == 0:
                    continue

                comments.append(line)
            # print(c.parent.type)
        # else:
        #     print(">> ", t)

    return comments


def main(filenames: list[Path], maxlines: int):
    parser = Parser()
    parser.set_language(SOLIDITY)

    # links
    # pattern = re.compile(r"(https?://[^\s]+)")
    # pattern.findall(line)

    for f in filenames:
        print(f"~~~ {f} ~~~")
        tree = parser.parse(f.read_bytes())
        # print(tree)
        # print(tree.root_node)

        # https://sambacha.github.io/tree-sitter-solidity/
        # https://ethereum.org/en/developers/docs/standards/tokens/erc-20/

        comments = get_top_comments(tree)

        print("\n".join(comments))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regex for URLs in a text file, and check if each are valid and reachable",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filenames", nargs="+", type=Path)
    parser.add_argument("--maxlines", type=int, default=50)
    args = parser.parse_args()
    main(args.filenames, args.maxlines)
