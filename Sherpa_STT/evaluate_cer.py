"""Compute character error rate for a manually transcribed reference."""
import argparse
import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    # Keep Korean and alphanumerics; ignore punctuation and whitespace consistently.
    return ''.join(re.findall(r'[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]', text.lower()))


def edit_distance(ref: str, hyp: str) -> int:
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        curr = [i]
        for j, b in enumerate(hyp, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (a != b)))
        prev = curr
    return prev[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, help='Human transcript UTF-8 .txt')
    parser.add_argument('--session', default='output/session.json')
    args = parser.parse_args()
    reference = normalize(Path(args.reference).read_text(encoding='utf-8-sig'))
    doc = json.loads(Path(args.session).read_text(encoding='utf-8'))
    predicted = normalize(' '.join(e['payload']['text'] for e in doc['events']
                     if e['event_type'] in ('transcript_final', 'transcript_edited')))
    if not reference:
        raise SystemExit('Reference is empty after normalization')
    distance = edit_distance(reference, predicted)
    print(f'Reference length: {len(reference)} characters')
    print(f'Edit distance: {distance}')
    print(f'CER: {distance / len(reference) * 100:.2f}%')
    print('NOTE: This compares the full transcript, not utterance-aligned segments.')


if __name__ == '__main__':
    main()
