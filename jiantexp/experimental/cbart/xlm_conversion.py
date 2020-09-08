import json

import jiant.utils.python.io as io


def convert_codes_to_merges(input_path, output_path):
    lines = io.read_file_lines(input_path, encoding="utf-8")
    converted_merges = []
    for i, line in enumerate(lines):
        assert line[-1] == "\n"
        tokens = line.split()
        if len(tokens) == 3:
            converted_merges.append(tokens[0] + " " + tokens[1])
        elif len(tokens) == 2:
            converted_merges.append(tokens[-2] + " " + tokens[-1])
        else:
            converted_merges.append(tokens[-1])
    io.write_file("\n".join(converted_merges) + "\n", output_path, encoding="utf-8")


def convert_vocab(input_path, output_path, num_special=10):
    target_size = 95000
    lines = io.read_file_lines(input_path)
    starting_vocab_ls = [
        '<s>',
        '</s>',
        '<pad>',
        '<unk>',
    ]
    for i in range(num_special):
        starting_vocab_ls.append(f'<special{i}>')
    vocab_ls = starting_vocab_ls[:]
    bad_ls = []
    seen_set = set()
    for i, line in enumerate(lines[:target_size]):
        filter_tokens = line.strip().split()
        seen_token = line.strip().split()[0]
        if len(filter_tokens) == 2 and (seen_token not in seen_set):
            seen_set.add(seen_token)
            tokens = line.split()
            token, _ = tokens
            if token.endswith("@@"):
                save_token = token[:-2]
            else:
                save_token = token + "</w>"
            vocab_ls.append(save_token)
        elif len(filter_tokens) == 1 or (seen_token in seen_set):
            bad_ls.append(i)
        else:
            raise RuntimeError()
    final_list = vocab_ls[:target_size]
    out_vocab = {word: i for i, word in enumerate(final_list)}
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out_vocab))