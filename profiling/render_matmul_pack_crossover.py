# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import pathlib
import shlex


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render the matmul pack-once crossover notebook.')
    parser.add_argument('benchmark', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    return parser.parse_args()


def format_ratio(ratio):
    return (
        f"{ratio['median']:.3f}x "
        f"({ratio['q10']:.3f}..{ratio['q90']:.3f})"
    )


def format_seconds(value):
    return f'{value * 1e6:.3f}'


def format_strides(row):
    return '<br>'.join(
        f"{operand['name']}: shape={tuple(operand['shape'])}, "
        f"strides={tuple(operand['strides'])}"
        for operand in row['operands']
    )


def classify(ratio):
    if ratio['q90'] < 0.95:
        return 'pack-faster'
    if ratio['q10'] >= 1.05:
        return 'current-faster'
    if ratio['q10'] >= 0.95 and ratio['q90'] < 1.05:
        return 'parity'
    return 'inconclusive'


def render_environment(lines, metadata):
    lines.extend((
        '## Environment',
        '',
        f"- Revision: `{metadata['git_revision']}`.",
        f"- Dirty tree: `{str(metadata['git_dirty']).lower()}`.",
        f"- Platform: `{metadata['platform']}`.",
        f"- Machine: `{metadata['machine']}`.",
        f"- Python: `{metadata['python']}`.",
        f"- NumPy: `{metadata['numpy']}`.",
        f"- Threads: `{metadata['thread_count']}`.",
        f"- Samples per route: `{metadata['repeat']}`.",
        f"- Warmups per route: `{metadata['warmup']}`.",
        '- Matrix sides: `'
        + ', '.join(str(value) for value in metadata['sides'])
        + '`.',
        '- Batch sizes: `'
        + ', '.join(str(value) for value in metadata['batches'])
        + '`.',
        '',
    ))


def render_routes(lines):
    lines.extend((
        '## Routes',
        '',
        '- `current` uses automatic prototype dispatch.',
        '- `generic` forces signed-stride contraction.',
        '- `pack_once` packs only unsupported supplied operands, rebuilds',
        '  the plan, and must enter BLAS.',
        '- `prepacked` moves the same copy outside the timed call.',
        '- `direct` exists only when both operands already have direct',
        '  BLAS descriptors.',
        '- NumPy view and prepacked routes locate NumPy layout overhead.',
        '',
        'Every route includes output allocation.  `pack_once` includes',
        'validation, packing, plan reconstruction, and BLAS calls.',
        '',
    ))


def render_summary(lines, rows):
    counts = {
        'pack-faster': 0,
        'parity': 0,
        'inconclusive': 0,
        'current-faster': 0,
    }
    for row in rows:
        status = classify(
            row['ratios']['pack_once_over_current'])
        counts[status] += 1
    lines.extend((
        '## Pack-once versus current dispatch',
        '',
        '| Result | Cases |',
        '| --- | ---: |',
        f"| Pack faster | {counts['pack-faster']} |",
        f"| Parity | {counts['parity']} |",
        f"| Inconclusive | {counts['inconclusive']} |",
        f"| Current faster | {counts['current-faster']} |",
        '',
        'This aggregate is a screening result.  Dispatch thresholds must',
        'come from the detailed side and batch crossover below.',
        '',
    ))


def render_rows(lines, rows):
    lines.extend((
        '## Detailed crossover',
        '',
        '| Topology | Layout | S | B | Supplied operands | Number | '
        'Current us | Generic us | Pack us | Prepacked us | NumPy view us | '
        'Generic/current | Pack/current | Pack/prepacked | '
        'NumPy view/prepacked | Result |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | '
        '---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ))
    for row in rows:
        medians = row['medians_seconds']
        ratios = row['ratios']
        status = classify(ratios['pack_once_over_current'])
        lines.append(
            f"| `{row['topology']}` | `{row['layout']}` | "
            f"{row['side']} | {row['batch']} | "
            f"{format_strides(row)} | {row['number']} | "
            f"{format_seconds(medians['current'])} | "
            f"{format_seconds(medians['generic'])} | "
            f"{format_seconds(medians['pack_once'])} | "
            f"{format_seconds(medians['prepacked'])} | "
            f"{format_seconds(medians['numpy_view'])} | "
            f"{format_ratio(ratios['generic_over_current'])} | "
            f"{format_ratio(ratios['pack_once_over_current'])} | "
            f"{format_ratio(ratios['pack_once_over_prepacked'])} | "
            f"{format_ratio(ratios['numpy_view_over_prepacked'])} | "
            f"{status} |"
        )
    lines.append('')


def render_reproduction(lines, metadata):
    sides = ','.join(str(value) for value in metadata['sides'])
    batches = ','.join(str(value) for value in metadata['batches'])
    filters = ''.join(
        f" --filter {shlex.quote(value)}"
        for value in metadata['case_filters']
    )
    cpu = metadata.get('requested_cpu')
    cpu_option = '' if cpu is None else f' --cpu {cpu}'
    lines.extend((
        '## Reproduce',
        '',
        '```console',
        '$ source /path/to/devenv/scripts/init',
        '$ devenv use prime',
        '$ make BUILD_QT=OFF',
        '$ PYTHONPATH=.:profiling python3 \\',
        '    profiling/profile_matmul_pack_crossover.py \\',
        f'    --sides {sides} \\',
        f'    --batches {batches} \\',
        f"    --repeat {metadata['repeat']} \\",
        f"    --warmup {metadata['warmup']}"
        f"{cpu_option}{filters} \\",
        '    --output /tmp/matmul-pack-crossover.json',
        '$ PYTHONPATH=.:profiling python3 \\',
        '    profiling/render_matmul_pack_crossover.py \\',
        '    /tmp/matmul-pack-crossover.json \\',
        '    /tmp/matmul-pack-crossover.md',
        '```',
        '',
        'On macOS, omit `--cpu`.',
        '',
    ))


def main():
    args = parse_args()
    payload = json.loads(
        args.benchmark.read_text(encoding='utf-8'))
    metadata = payload['metadata']
    rows = payload['results']
    lines = ['# Matmul pack-once crossover', '']
    render_environment(lines, metadata)
    render_routes(lines)
    render_summary(lines, rows)
    render_rows(lines, rows)
    render_reproduction(lines, metadata)
    lines.append(
        '<!-- vim: set ft=markdown ff=unix fenc=utf8 '
        'et sw=2 ts=2 sts=2 tw=79: -->'
    )
    args.output.write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
