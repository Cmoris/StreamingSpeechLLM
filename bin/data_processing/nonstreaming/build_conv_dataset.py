import json
import os
import glob
import re
import argparse


def load_jsonl(filepath):
    """Load all JSON records from a JSONL file."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def convert_records_to_conversations(records):
    conversations = []
    for record in records[:-1]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": record["audio_path"],
                        "start": record["start"],
                        "end": record["end"],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": record["text"],
                    }
                ],
            },
        ]
        conversations.append(conversation)
    return conversations

def extract_conversation(records):
    """
    Extract all [start, end, char] entries from assistant text_stream content.
    Handles both list-of-dicts and single-dict record formats.
    """
    all_entries = []

    for record in records:
        items = record if isinstance(record, list) else [record]
        all_entries.append(items)
    return all_entries


def merge_pair(file_a, file_b, output_path):
    """Merge one _a / _b pair and write the result."""
    records_a = load_jsonl(file_a)
    records_b = load_jsonl(file_b)
    conversations_a = convert_records_to_conversations(records_a)
    conversations_b = convert_records_to_conversations(records_b)
    entries_a = extract_conversation(conversations_a)
    entries_b = extract_conversation(conversations_b)

    entries = entries_a + entries_b
    try:
        entries.sort(key=lambda x: x[0]["content"][0]["start"])
    except Exception as e:
        print(e)
        breakpoint()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            line = json.dumps(entry, ensure_ascii=True) + "\n"
            f.write(line)

    return len(entries_a), len(entries_b), len(entries)


def find_pairs(input_dir):
    """Return sorted list of (file_a, file_b, base_name) triples."""
    a_files = sorted(glob.glob(os.path.join(input_dir, '*_a.jsonl')))
    pairs = []
    for a_file in a_files:
        b_file = a_file.replace('_a.jsonl', '_b.jsonl')
        if os.path.exists(b_file):
            base = re.sub(r'_a\.jsonl$', '', os.path.basename(a_file))
            pairs.append((a_file, b_file, base))
        else:
            print(f"[WARN] No matching _b file for: {os.path.basename(a_file)}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Merge _a and _b channel JSONL files.')
    parser.add_argument('--input',  default='.', help='Directory containing *_a_after.jsonl files')
    parser.add_argument('--output', default='./merged', help='Output directory for merged files')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pairs = find_pairs(args.input)

    if not pairs:
        print(f'No _a/_b pairs found in: {args.input}')
        return

    print(f'Found {len(pairs)} pair(s). Merging...\n')
    for file_a, file_b, base in pairs:
        out = os.path.join(args.output, f'{base}.jsonl')
        n_a, n_b, n_total = merge_pair(file_a, file_b, out)
        print(f'  {base:20s}  A={n_a:4d}  B={n_b:4d}  merged={n_total:4d}  → {out}')

    print(f'\nDone. Merged files saved to: {args.output}')


if __name__ == '__main__':
    main()