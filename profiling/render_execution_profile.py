# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import pathlib
import shlex


LEGACY_STATUSES = (
    'improved',
    'parity',
    'regression',
    'legacy-incorrect',
    'new-only',
    'inconclusive',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render a detailed execution benchmark notebook.')
    parser.add_argument('input', type=pathlib.Path,
                        help='Benchmark JSON produced by the profiler.')
    parser.add_argument('output', type=pathlib.Path,
                        help='Markdown notebook to write.')
    return parser.parse_args()


def format_flags(operand):
    flags = ''
    if operand['c_contiguous']:
        flags += 'C'
    if operand['f_contiguous']:
        flags += 'F'
    if operand['negative_stride']:
        flags += 'N'
    if operand['zero_stride']:
        flags += 'Z'
    return flags or 'S'


def format_operand(operand):
    name = operand['name']
    if 'scalar' in operand:
        return f"{name}=scalar({operand['scalar']})"
    shape = ','.join(str(value) for value in operand['shape'])
    strides = ','.join(str(value) for value in operand['strides'])
    return (f'{name}: shape=({shape}), strides=({strides}), '
            f'flags={format_flags(operand)}')


def format_operands(operands):
    return '<br>'.join(format_operand(operand) for operand in operands)


def format_seconds(value):
    return 'n/a' if value is None else f'{value * 1000:.6f}'


def format_ratio(row, prefix):
    median = row.get(prefix)
    if median is None:
        return 'n/a'
    q10 = row[f'{prefix}_q10']
    q90 = row[f'{prefix}_q90']
    return f'{median:.3f}x ({q10:.3f}..{q90:.3f})'


def count_status(rows, status):
    return sum(row['status'] == status for row in rows)


def platform_name(metadata):
    system = metadata['platform'].lower()
    if system.startswith('macos'):
        return 'macOS'
    if system.startswith('linux'):
        return 'Linux'
    return metadata['machine']


def render_environment(lines, metadata):
    affinity = metadata.get('cpu_affinity')
    affinity_text = 'n/a' if affinity is None else ', '.join(
        str(cpu) for cpu in affinity)
    lines.extend((
        '## Recorded environment',
        '',
        f"- Code revision: `{metadata['git_revision']}`.",
        f"- Dirty tree: `{str(metadata['git_dirty']).lower()}`.",
        f"- Platform: `{metadata['platform']}`.",
        f"- Machine: `{metadata['machine']}`.",
        f"- Python: `{metadata['python']}`.",
        f"- NumPy: `{metadata['numpy']}`.",
        f"- Seed: `{metadata['seed']}`.",
        f"- Fixed cases: `{metadata['case_count']}`.",
        '- HPC matmul suite: '
        f"`{str(metadata.get('hpc_matmul', False)).lower()}`.",
        '- Optional slow matmul case: '
        f"`{str(metadata.get('hpc_matmul_slow', False)).lower()}`.",
        f"- Samples per route: `{metadata['repeat']}`.",
        f"- Warmups per route: `{metadata['warmup']}`.",
        f"- Threads: `{metadata['thread_count']}`.",
        f'- CPU affinity: `{affinity_text}`.',
        '',
        'The machine-readable JSON also records NumPy build configuration,',
        'extension linkage, thread-control variables, raw timing samples,',
        'paired ratios, and q10/q90 ratio quantiles.',
        '',
    ))


def render_reproduction(lines, metadata):
    cpu = metadata.get('requested_cpu')
    cpu_option = '' if cpu is None else f' --cpu {cpu}'
    revision = metadata['git_revision'][:8]
    command = [
        '$ python3 profiling/profile_execution_prototype.py \\',
        '    --benchmark-only \\',
    ]
    if metadata.get('hpc_matmul', False):
        command.append('    --hpc-matmul \\')
    if metadata.get('hpc_matmul_slow', False):
        command.append('    --hpc-matmul-slow \\')
    for case_filter in metadata.get('case_filters', []):
        command.append(f'    --filter {shlex.quote(case_filter)} \\')
    command.extend((
        f"    --repeat {metadata['repeat']} \\",
        f"    --warmup {metadata['warmup']}{cpu_option} \\",
        f'    --output /tmp/solvcon-execution-{revision}.json',
    ))
    lines.extend((
        '## Reproduction',
        '',
        '```console',
        '$ source /path/to/devenv/scripts/init',
        '$ devenv use prime',
        *command,
        '```',
        '',
        'The profiler silently checks every route against NumPy before',
        'timing. Mutable destinations are reset before every route and',
        'sample. Call order rotates within each paired sample.',
        '',
    ))


def render_reading_guide(lines):
    lines.extend((
        '## Reading the tables',
        '',
        '- Strides are measured in elements, not bytes.',
        '- `C` and `F` mean C- and F-contiguous.',
        '- `N` means at least one negative stride.',
        '- `S` means a general non-dense strided view.',
        '- `Z` means at least one zero stride.',
        '- A ratio greater than one means planned is faster.',
        '- Every ratio shows `median (q10..q90)` from paired samples.',
        '- Faster or slower requires the full interval to clear five',
        '  percent. A crossing interval is `inconclusive`.',
        '- Existing BLAS is `SimpleArray::matmul_blas()` and is measured',
        '  only for unbatched rank-1 and rank-2 operations.',
        '- A control is another planned route measured in the same rotating',
        '  sample. Its label identifies the compared layout.',
        '- `legacy-incorrect` means planned matched NumPy but legacy did',
        '  not. Incorrect legacy results are not timed.',
        '',
    ))


def render_inventory(lines, rows, families):
    lines.extend((
        '## Coverage inventory',
        '',
        '| Family | Cases | Improved | Parity | Regression | '
        'Legacy incorrect | New only | Inconclusive |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ))
    for family in families:
        family_rows = [row for row in rows if row['family'] == family]
        counts = [
            count_status(family_rows, status)
            for status in LEGACY_STATUSES
        ]
        lines.append(
            f'| {family} | {len(family_rows)} | {counts[0]} | '
            f'{counts[1]} | {counts[2]} | {counts[3]} | '
            f'{counts[4]} | {counts[5]} |')
    lines.append('')


def format_mib(value):
    return f'{value / 1024 / 1024:.1f}'


def format_workload(workload):
    if workload is None:
        return 'n/a'
    matrix = (
        f"{workload['rows']}x{workload['inner_size']}x"
        f"{workload['columns']}")
    macs = workload['multiply_accumulates'] / 1e9
    return (
        f"B={workload['batch_matrices']}, "
        f"R={workload['batch_rank']}, "
        f"MxKxN={matrix}<br>"
        f"MAC={macs:.3f}G, "
        f"logical input="
        f"{format_mib(workload['logical_input_bytes'])} MiB, "
        f"backing input="
        f"{format_mib(workload['backing_input_bytes'])} MiB, "
        f"expanded input="
        f"{format_mib(workload['expanded_input_bytes'])} MiB, "
        f"output={format_mib(workload['output_bytes'])} MiB")


def render_family(lines, family, rows):
    has_workload = any(row.get('workload') is not None for row in rows)
    has_blas = any(
        row.get('blas_seconds') is not None or
        row.get('blas_correct') is not None
        for row in rows)
    has_control = any(row.get('control_label') for row in rows)
    headers = ['Operation', 'Scenario']
    separators = ['---', '---']
    if has_workload:
        headers.append('Workload')
        separators.append('---')
    headers.extend((
        'Operands', 'Calls/sample', 'NumPy ms', 'Legacy ms'))
    separators.extend(('---', '---:', '---:', '---:'))
    if has_blas:
        headers.append('Existing BLAS ms')
        separators.append('---:')
    if has_control:
        headers.append('Control ms')
        separators.append('---:')
    headers.append('Planned ms')
    separators.append('---:')
    headers.append('Legacy/planned (q10..q90)')
    separators.append('---:')
    if has_blas:
        headers.append('BLAS/planned (q10..q90)')
        separators.append('---:')
    if has_control:
        headers.append('Control/planned (q10..q90)')
        separators.append('---:')
    headers.extend((
        'NumPy/planned (q10..q90)', 'Legacy status'))
    separators.extend(('---:', '---'))
    if has_blas:
        headers.append('Planned vs BLAS')
        separators.append('---')
    if has_control:
        headers.append('Planned vs control')
        separators.append('---')
    headers.append('Planned vs NumPy')
    separators.append('---')

    lines.extend((
        f'### {family}',
        '',
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(separators) + ' |',
    ))
    for row in rows:
        cells = [row['operation'], row['layout']]
        if has_workload:
            cells.append(format_workload(row.get('workload')))
        cells.extend((
            format_operands(row['operands']),
            str(row['number']),
            format_seconds(row['numpy_seconds']),
            format_seconds(row['legacy_seconds']),
        ))
        if has_blas:
            cells.append(format_seconds(row.get('blas_seconds')))
        if has_control:
            control = format_seconds(row.get('control_seconds'))
            if row.get('control_label'):
                control += f" ({row['control_label']})"
            cells.append(control)
        cells.extend((
            format_seconds(row['planned_seconds']),
            format_ratio(row, 'legacy_over_planned'),
        ))
        if has_blas:
            cells.append(format_ratio(row, 'blas_over_planned'))
        if has_control:
            cells.append(format_ratio(row, 'control_over_planned'))
        cells.extend((
            format_ratio(row, 'numpy_over_planned'),
            row['status'],
        ))
        if has_blas:
            cells.append(row.get('planned_vs_blas', 'not-measured'))
        if has_control:
            cells.append(row.get('planned_vs_control', 'not-measured'))
        cells.append(row['planned_vs_numpy'])
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')


def mismatch_summary(error):
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    for line in lines:
        if line.startswith('Mismatched elements:'):
            return line.replace('|', '\\|')
    return (lines[0] if lines else 'n/a').replace('|', '\\|')


def render_failures(lines, rows):
    failures = [
        row for row in rows if row['status'] == 'legacy-incorrect'
    ]
    lines.extend((
        '## Legacy correctness failures',
        '',
        'These rows are not performance regressions. Planned matched',
        'NumPy, while legacy returned a different value.',
        '',
        '| Family | Operation | Scenario | Error | Mismatch summary |',
        '| --- | --- | --- | --- | --- |',
    ))
    for row in failures:
        lines.append(
            f"| {row['family']} | {row['operation']} | "
            f"{row['layout']} | {row['legacy_error_type']} | "
            f"{mismatch_summary(row['legacy_error'])} |")
    lines.append('')


def render_notebook(payload):
    metadata = payload['metadata']
    rows = payload['results']
    families = list(dict.fromkeys(row['family'] for row in rows))
    lines = [
        f'# {platform_name(metadata)} detailed execution benchmark',
        '',
    ]
    render_environment(lines, metadata)
    render_reproduction(lines, metadata)
    render_reading_guide(lines)
    render_inventory(lines, rows, families)
    lines.extend(('## Complete results', ''))
    for family in families:
        render_family(
            lines,
            family,
            [row for row in rows if row['family'] == family],
        )
    render_failures(lines, rows)
    lines.extend((
        '## Interpretation boundary',
        '',
        'This notebook records every row, including inconclusive results.',
        'It is evidence for choosing reusable routes, not a claim that the',
        'prototype is the final implementation. Results from another OS,',
        'architecture, or BLAS backend require a separate run.',
        '',
        '<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 '
        'sts=2 tw=79: -->',
        '',
    ))
    return '\n'.join(lines)


def main():
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_notebook(payload), encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
