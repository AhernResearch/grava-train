#!/usr/bin/env python3
"""Select the next candidate or GRPO set using scored policy rollouts."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from grava_train.rl.selection import (
    GreedyRule,
    HighVarianceRule,
    aggregate_rollouts,
    export_selected_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-file', required=True, help='Canonical GRPO pool JSONL')
    parser.add_argument('--data-root', default=None, help='Root for relative dataset paths')
    parser.add_argument('--rollouts', nargs='+', required=True, help='Scored JSONL files/shards')
    parser.add_argument('--rule', choices=['greedy', 'high_variance'], required=True)
    parser.add_argument('--num-rollouts', type=int, default=8, help='Expected samples per case')
    parser.add_argument('--metric', choices=['pdm_score', 'cds', 'rl_score'], default='pdm_score')
    parser.add_argument('--greedy-threshold', type=float, default=0.9)
    parser.add_argument('--min-score-lt', type=float, default=0.8)
    parser.add_argument('--max-score-ge', type=float, default=0.9)
    parser.add_argument('--std-ge', type=float, default=0.1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.num_rollouts < 1:
        parser.error('--num-rollouts must be positive')
    source = (Path(args.data_root or '.').expanduser() / args.dataset_file).resolve(strict=True)
    paths = [Path(p).resolve(strict=True) for p in args.rollouts]
    output = Path(args.output).resolve()
    rule = (GreedyRule(args.greedy_threshold) if args.rule == 'greedy' else
            HighVarianceRule(args.min_score_lt, args.max_score_ge, args.std_ge))
    expected = 1 if args.rule == 'greedy' else args.num_rollouts
    cases = aggregate_rollouts(paths, expected, args.metric)
    statistics = [case.statistics() for case in cases.values()]
    selected = {row['id'] for row in statistics if rule.accepts(row)}
    # A greedy scan must cover the whole supplied pool; high-variance sampling
    # intentionally covers only the candidates exported by the greedy stage.
    with source.open() as stream:
        source_ids = {json.loads(line)['id'] for line in stream}
    if args.rule == 'greedy' and cases.keys() != source_ids:
        raise ValueError('Greedy rollout IDs must cover the supplied canonical pool exactly')
    if not cases.keys() <= source_ids:
        raise ValueError('Rollout IDs are absent from the supplied canonical pool')
    rows = export_selected_rows(source, output, selected)
    summary = {'input_cases': len(cases), 'selected_cases': rows,
               'num_rollouts': expected, 'metric': args.metric, 'rule': args.rule,
               'thresholds': asdict(rule)}
    manifest = {**summary, 'source_dataset': str(source),
                'rollouts': [{'path': str(p)} for p in paths],
                'output': str(output),
                'image_paths_relative_to': str(Path(args.data_root).expanduser().resolve() if args.data_root else source.parent)}
    output.with_suffix('.stats.json').write_text(json.dumps(summary, indent=2) + '\n')
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with output.with_suffix('.case_stats.jsonl').open('w') as stream:
        for row in sorted(statistics, key=lambda row: row['id']):
            stream.write(json.dumps({**row, 'selected': row['id'] in selected}) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
