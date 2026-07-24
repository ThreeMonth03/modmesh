# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import pathlib


STATUSES = (
    'pack-faster',
    'parity',
    'inconclusive',
    'generic-faster',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render the rectangular vector packing notebook.')
    parser.add_argument('benchmark', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    return parser.parse_args()


def classify(ratio):
    if ratio['q90'] < 0.95:
        return 'pack-faster'
    if ratio['q10'] >= 1.05:
        return 'generic-faster'
    if ratio['q10'] >= 0.95 and ratio['q90'] < 1.05:
        return 'parity'
    return 'inconclusive'


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


def count_statuses(rows):
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        status = classify(
            row['ratios']['pack_once_over_generic'])
        counts[status] += 1
    return counts


def render_environment(lines, metadata):
    pairs = ', '.join(
        f'{inner_size}x{output_extent}'
        for inner_size, output_extent
        in metadata['dimension_pairs']
    )
    batches = ', '.join(
        str(value) for value in metadata['batches'])
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
        f"- Dimension pairs `(K, O)`: `{pairs}`.",
        f"- Batch sizes: `{batches}`.",
        '',
        '`O` is the output extent: `N` for `1D @ ND` and `M` for',
        '`ND @ 1D`.  The matrix core is always C layout so this notebook',
        'isolates vector packing from matrix packing.',
        '',
    ))


def render_policy(lines, metadata):
    current = metadata['current_policy']
    reuse = metadata['reuse_extension']
    lines.extend((
        '## Predicates under test',
        '',
        'The current portable predicate is:',
        '',
        '```text',
        f"core_work >= {current['minimum_core_work']}",
        f"batch >= {current['minimum_batches']}",
        '```',
        '',
        'The reuse-aware extension is:',
        '',
        '```text',
        f"core_work >= {reuse['minimum_core_work']}",
        'reuse_intensity = batch * output_extent',
        'reuse_intensity >= '
        f"{reuse['minimum_reuse_intensity']}",
        '```',
        '',
        'Because `core_work = K * output_extent`, the second condition is',
        'equivalent to comparing total contracted work with the `K` values',
        'copied by vector packing.',
        '',
        'The extension adds work to the current predicate.  It does not',
        'disable an existing pack-once selection.',
        '',
    ))


def render_summary(lines, rows):
    policies = (
        (
            'Current portable predicate',
            [row for row in rows
             if row['current_policy_selected']],
        ),
        (
            'Reuse-aware extension only',
            [row for row in rows
             if row['reuse_policy_selected']
             and not row['current_policy_selected']],
        ),
        (
            'Combined predicate',
            [row for row in rows
             if row['combined_policy_selected']],
        ),
        ('All measured rows', rows),
    )
    lines.extend((
        '## Pack-once versus generic contraction',
        '',
        '| Selection | Rows | Pack faster | Parity | Inconclusive | '
        'Generic faster |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ))
    for label, selected in policies:
        counts = count_statuses(selected)
        lines.append(
            f"| {label} | {len(selected)} | "
            f"{counts['pack-faster']} | {counts['parity']} | "
            f"{counts['inconclusive']} | "
            f"{counts['generic-faster']} |"
        )
    lines.extend((
        '',
        'An extension is acceptable only when every added rectangular',
        'factorization avoids a conclusive generic-faster result on both',
        'OpenBLAS and Accelerate.',
        '',
    ))


def render_rows(lines, rows):
    lines.extend((
        '## Detailed rectangular crossover',
        '',
        '| Topology | Vector layout | K | O | Core work | B | '
        'Reuse intensity | Current | Extension | Combined | '
        'Supplied operands | '
        'Number | Current us | Generic us | Pack us | Prepacked us | '
        'NumPy us | Pack/generic | Generic/current | Pack/current | '
        'NumPy/current | Result |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | '
        '--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | '
        '---: | ---: | ---: | ---: | --- |',
    ))
    for row in rows:
        medians = row['medians_seconds']
        ratios = row['ratios']
        status = classify(ratios['pack_once_over_generic'])
        current = 'pack' if row['current_policy_selected'] else 'generic'
        reuse = (
            'pack' if row['reuse_policy_selected'] else 'generic')
        combined = (
            'pack' if row['combined_policy_selected'] else 'generic')
        lines.append(
            f"| `{row['topology']}` | `{row['layout']}` | "
            f"{row['inner_size']} | {row['output_extent']} | "
            f"{row['core_work']} | {row['batch']} | "
            f"{row['reuse_intensity']} | {current} | {reuse} | "
            f"{combined} | "
            f"{format_strides(row)} | {row['number']} | "
            f"{format_seconds(medians['current'])} | "
            f"{format_seconds(medians['generic'])} | "
            f"{format_seconds(medians['pack_once'])} | "
            f"{format_seconds(medians['prepacked'])} | "
            f"{format_seconds(medians['numpy_view'])} | "
            f"{format_ratio(ratios['pack_once_over_generic'])} | "
            f"{format_ratio(ratios['generic_over_current'])} | "
            f"{format_ratio(ratios['pack_once_over_current'])} | "
            f"{format_ratio(ratios['numpy_view_over_current'])} | "
            f"{status} |"
        )
    lines.append('')


def render_reproduction(lines, metadata):
    pairs = ','.join(
        f'{inner_size}x{output_extent}'
        for inner_size, output_extent
        in metadata['dimension_pairs']
    )
    batches = ','.join(
        str(value) for value in metadata['batches'])
    cpu = metadata.get('requested_cpu')
    cpu_option = '' if cpu is None else f' --cpu {cpu}'
    lines.extend((
        '## Reproduce',
        '',
        '```console',
        '$ source /path/to/devenv/scripts/init',
        '$ devenv use prime',
        '$ make BUILD_QT=OFF \\',
        '    CMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)"',
        '$ PYTHONPATH=.:profiling python3 \\',
        '    profiling/profile_matmul_vector_pack_rectangular.py \\',
        f'    --dimension-pairs {pairs} \\',
        f'    --batches {batches} \\',
        f"    --repeat {metadata['repeat']} \\",
        f"    --warmup {metadata['warmup']}{cpu_option} \\",
        '    --output /tmp/matmul-vector-pack-rectangular.json',
        '$ PYTHONPATH=.:profiling python3 \\',
        '    profiling/render_matmul_vector_pack_rectangular.py \\',
        '    /tmp/matmul-vector-pack-rectangular.json \\',
        '    /tmp/matmul-vector-pack-rectangular.md',
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
    lines = ['# Rectangular vector packing crossover', '']
    render_environment(lines, metadata)
    render_policy(lines, metadata)
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
