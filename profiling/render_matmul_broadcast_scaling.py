# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import pathlib
import shlex


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render the matmul broadcast scaling notebook.')
    parser.add_argument(
        'benchmark',
        type=pathlib.Path,
        help='Timing JSON from profile_matmul_broadcast_scaling.py.',
    )
    parser.add_argument(
        'output',
        type=pathlib.Path,
        help='Markdown output path.',
    )
    parser.add_argument(
        '--trace',
        type=pathlib.Path,
        help='Optional profile-enabled trace JSON.',
    )
    return parser.parse_args()


def format_shape(shape):
    if len(shape) == 1:
        return f'({shape[0]},)'
    return '(' + ', '.join(str(value) for value in shape) + ')'


def format_seconds(value):
    return f'{value * 1000:.4f}'


def format_ratio(ratio):
    return (
        f"{ratio['median']:.3f}x "
        f"({ratio['q10']:.3f}..{ratio['q90']:.3f})"
    )


def format_samples(values):
    return ', '.join(f'{value * 1000:.4f}' for value in values)


def operand_by_name(row, name):
    return next(
        operand for operand in row['operands']
        if operand['name'] == name
    )


def platform_name(metadata):
    system = metadata['platform'].lower()
    if system.startswith('macos'):
        return 'macOS'
    if system.startswith('linux'):
        return 'Linux'
    return metadata['machine']


def render_environment(lines, metadata):
    affinity = metadata.get('cpu_affinity')
    affinity_text = (
        'n/a' if affinity is None
        else ', '.join(str(cpu) for cpu in affinity)
    )
    lines.extend((
        '## Recorded environment',
        '',
        f"- Code revision: `{metadata['git_revision']}`.",
        f"- Dirty tree: `{str(metadata['git_dirty']).lower()}`.",
        f"- Platform: `{metadata['platform']}`.",
        f"- Machine: `{metadata['machine']}`.",
        f"- Python: `{metadata['python']}`.",
        f"- NumPy: `{metadata['numpy']}`.",
        f"- Matrix size: `{metadata['side']} by {metadata['side']}`.",
        '- Batch sizes: `'
        + ', '.join(str(value) for value in metadata['batches'])
        + '`.',
        f"- Samples per route: `{metadata['repeat']}`.",
        f"- Warmups per route: `{metadata['warmup']}`.",
        f"- Threads: `{metadata['thread_count']}`.",
        f"- CPU affinity: `{affinity_text}`.",
        '',
        'The JSON also retains every timing sample, NumPy build',
        'configuration, extension linkage, thread-control variables,',
        'source layouts, and planned batch mappings.',
        '',
    ))


def command_options(metadata):
    batches = ','.join(str(value) for value in metadata['batches'])
    cpu = metadata.get('requested_cpu')
    cpu_option = '' if cpu is None else f' --cpu {cpu}'
    return (
        f'    --batches {shlex.quote(batches)} \\\n'
        f"    --side {metadata['side']} \\\n"
        f"    --repeat {metadata['repeat']} \\\n"
        f"    --warmup {metadata['warmup']}{cpu_option}"
    )


def render_reproduction(lines, metadata, has_trace):
    revision = metadata['git_revision'][:8]
    timing_json = f'/tmp/matmul-broadcast-{revision}.json'
    trace_json = f'/tmp/matmul-broadcast-trace-{revision}.json'
    markdown = f'/tmp/matmul-broadcast-{revision}.md'
    lines.extend((
        '## Reproduction',
        '',
        'The timing build compiles every probe out. The trace build is',
        'separate so probe overhead never enters the timing samples.',
        '',
        '```console',
        '$ source /path/to/devenv/scripts/init',
        '$ devenv use prime',
        '$ export CMAKE_PREFIX_PATH="$DEVENVPREFIX"',
        '$ export CMAKE_ARGS="'
        '-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)"',
        '$ make BUILD_PATH_EXT=_benchmark BUILD_QT=OFF \\',
        '    SOLVCON_PROFILE=OFF',
        '$ python3 profiling/profile_matmul_broadcast_scaling.py \\',
        command_options(metadata) + ' \\',
        f'    --output {timing_json}',
        '$ make BUILD_PATH_EXT=_profile BUILD_QT=OFF \\',
        '    SOLVCON_PROFILE=ON',
        '$ python3 profiling/profile_matmul_broadcast_scaling.py \\',
        command_options(metadata) + ' \\',
        '    --trace-only \\',
        f'    --output {trace_json}',
        '$ python3 profiling/render_matmul_broadcast_scaling.py \\',
        f'    {timing_json} \\',
        f'    {markdown} \\',
        f'    --trace {trace_json}',
        '```',
        '',
        'On macOS, omit `--cpu` because process affinity is unavailable.',
        'The benchmark checks every planned result against a contiguous',
        'NumPy reference before either timing or tracing.',
        '',
    ))
    if not has_trace:
        lines.extend((
            'This notebook was rendered without the optional trace JSON.',
            'Run the profile-enabled commands above to record actual pack',
            'and GEMM counts.',
            '',
        ))


def render_question(lines, metadata):
    side = metadata['side']
    matrix_stride = side * side
    lines.extend((
        '## Question under test',
        '',
        f'The source lhs has shape `(1, {side}, {side})`. Its stored batch',
        f'stride is {matrix_stride:,} because consecutive physical',
        'matrices would be that many elements apart. For `B > 1`,',
        '`MatmulPlan` does not advance by that stride. Broadcast alignment',
        'replaces it with zero:',
        '',
        '```text',
        f'source lhs shape:          (1, {side}, {side})',
        'source lhs strides:        '
        f'({matrix_stride}, {side}, 1)',
        'result batch shape:        (B)',
        'planned lhs batch stride:  (0)',
        f'planned rhs batch stride:  ({matrix_stride})',
        '',
        'for b in [0, B):',
        '    gemm(lhs + 0, '
        f'rhs + b * {matrix_stride}, out + b * {matrix_stride})',
        '```',
        '',
        'The large source stride is therefore metadata for an extent-one',
        'axis, not a per-element memory jump. The controls below test',
        'whether this explanation matches both timing and actual dispatch.',
        '',
    ))


def render_routes(lines, rows):
    routes = {}
    for row in rows:
        routes[row['route']] = row
    lines.extend((
        '## Route design',
        '',
        '| Route | Purpose | Lhs shape | Lhs source strides | '
        'Expected dispatch |',
        '| --- | --- | --- | --- | --- |',
    ))
    for route, row in routes.items():
        lhs = operand_by_name(row, 'lhs')
        plan = row['plan']
        dispatch = (
            'pack lhs once, then batched GEMM'
            if plan['expected_pack_lhs']
            else 'direct batched GEMM'
        )
        lines.append(
            f"| {route} | {row['purpose']} | "
            f"`{format_shape(lhs['shape'])}` | "
            f"`{format_shape(lhs['strides'])}` | {dispatch} |"
        )
    lines.extend((
        '',
        '`prepacked-negative` and `prepacked-step2` preserve the same',
        'logical values as their views, but move the one-time copy outside',
        'the timed call. `materialized-dense` physically stores `B` lhs',
        'matrices and isolates the cost of zero-stride reuse.',
        '',
    ))


def trace_lookup(trace_payload):
    if trace_payload is None:
        return {}
    return {
        (row['batch'], row['route']): row
        for row in trace_payload['traces']
    }


def count_text(trace, name):
    if trace is None:
        return 'n/a'
    return str(trace['counts'].get(name, 0))


def render_results(lines, rows, traces):
    lines.extend((
        '## Complete timing results',
        '',
        'A NumPy/planned ratio greater than one means planned is faster.',
        'Intervals are q10 to q90 from paired samples.',
        '',
        '| B | Route | Lhs shape | Source batch stride | '
        'Planned batch stride | NumPy ms | Planned ms | '
        'NumPy/planned | Pack lhs | GEMM calls |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | '
        '---: | ---: |',
    ))
    for row in rows:
        lhs = operand_by_name(row, 'lhs')
        trace = traces.get((row['batch'], row['route']))
        lines.append(
            f"| {row['batch']} | {row['route']} | "
            f"`{format_shape(lhs['shape'])}` | "
            f"`{format_shape(lhs['strides'][:-2])}` | "
            f"`{format_shape(row['plan']['lhs_batch_strides'])}` | "
            f"{format_seconds(row['numpy_seconds'])} | "
            f"{format_seconds(row['planned_seconds'])} | "
            f"{format_ratio(row['numpy_over_planned'])} | "
            f"{count_text(trace, 'execution.matmul.pack_lhs')} | "
            f"{count_text(trace, 'execution.matmul.gemm')} |"
        )
    lines.append('')


def render_comparisons(lines, comparisons):
    lines.extend((
        '## Paired controls',
        '',
        'The ratio is tested route divided by matching control. One means',
        'parity. Greater than one means the view or broadcast route costs',
        'more than its control.',
        '',
        '| B | Comparison | NumPy tested/control | '
        'Planned tested/control |',
        '| ---: | --- | ---: | ---: |',
    ))
    for row in comparisons:
        lines.append(
            f"| {row['batch']} | {row['comparison']} | "
            f"{format_ratio(row['numpy_tested_over_control'])} | "
            f"{format_ratio(row['planned_tested_over_control'])} |"
        )
    lines.append('')


def render_trace_validation(lines, rows, traces):
    if not traces:
        return
    lines.extend((
        '## Dispatch validation',
        '',
        'These counts come from a separate profile-enabled build. The',
        'timing build contains no probes.',
        '',
        '| B | Route | Expected pack lhs | Actual pack lhs | '
        'Expected GEMM | Actual GEMM | Generic calls | Result |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |',
    ))
    for row in rows:
        trace = traces[(row['batch'], row['route'])]
        counts = trace['counts']
        expected_pack = int(row['plan']['expected_pack_lhs'])
        expected_gemm = row['plan']['expected_gemm_calls']
        actual_pack = counts['execution.matmul.pack_lhs']
        actual_gemm = counts['execution.matmul.gemm']
        generic = counts['execution.matmul.generic']
        passed = (
            expected_pack == actual_pack and
            expected_gemm == actual_gemm and
            generic == 0
        )
        lines.append(
            f"| {row['batch']} | {row['route']} | {expected_pack} | "
            f"{actual_pack} | {expected_gemm} | {actual_gemm} | "
            f"{generic} | {'pass' if passed else 'FAIL'} |"
        )
    lines.append('')


def render_raw_samples(lines, rows):
    lines.extend((
        '## Raw timing samples',
        '',
        'All values are milliseconds per complete matmul call.',
        '',
        '| B | Route | NumPy samples | Planned samples |',
        '| ---: | --- | --- | --- |',
    ))
    for row in rows:
        lines.append(
            f"| {row['batch']} | {row['route']} | "
            f"{format_samples(row['numpy_samples_seconds'])} | "
            f"{format_samples(row['planned_samples_seconds'])} |"
        )
    lines.append('')


def render_interpretation(lines, rows, comparisons, traces):
    largest_batch = max(row['batch'] for row in rows)
    largest = [
        row for row in comparisons
        if row['batch'] == largest_batch
    ]
    lines.extend((
        '## Interpretation boundary',
        '',
        f'At the largest measured batch, `B={largest_batch}`:',
        '',
    ))
    for row in largest:
        lines.append(
            f"- {row['comparison']}: NumPy "
            f"{row['numpy_tested_over_control']['median']:.3f}x, "
            f"planned "
            f"{row['planned_tested_over_control']['median']:.3f}x."
        )
    lines.extend((
        '',
        'Broadcast and materialized routes have equal arithmetic: each',
        'performs `B` GEMMs. Broadcast saves lhs storage and may improve',
        'cache reuse, but it does not reduce the contraction count.',
        '',
        'The negative and step-two comparisons isolate in-call packing.',
        'A planned ratio near one is consistent with one physical lhs copy',
        'amortized across `B` GEMMs. A large NumPy ratio says only that the',
        'specific NumPy backend handles that view differently. It is not',
        'evidence that broadcasting itself makes matrix multiplication',
        'hundreds of times faster.',
        '',
    ))
    if traces:
        lines.extend((
            'The trace table is the stronger implementation check: the',
            'strided routes must report one lhs pack and exactly `B` GEMM',
            'calls, while dense and prepacked routes report zero packs.',
            '',
        ))
    lines.extend((
        'Results remain backend-specific. Compare Apple Silicon numbers',
        'only after NumPy and `_solvcon` linkage confirm the intended',
        'Accelerate configuration.',
        '',
    ))


def render_notebook(benchmark, trace_payload):
    metadata = benchmark['metadata']
    rows = benchmark['results']
    traces = trace_lookup(trace_payload)
    lines = [
        f'# {platform_name(metadata)} matmul broadcast scaling',
        '',
    ]
    render_environment(lines, metadata)
    render_reproduction(lines, metadata, bool(traces))
    render_question(lines, metadata)
    render_routes(lines, rows)
    render_results(lines, rows, traces)
    render_comparisons(lines, benchmark['comparisons'])
    render_trace_validation(lines, rows, traces)
    render_raw_samples(lines, rows)
    render_interpretation(
        lines, rows, benchmark['comparisons'], traces)
    lines.extend((
        '<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 '
        'sts=2 tw=79: -->',
        '',
    ))
    return '\n'.join(lines)


def main():
    args = parse_args()
    benchmark = json.loads(
        args.benchmark.read_text(encoding='utf-8'))
    trace_payload = None
    if args.trace is not None:
        trace_payload = json.loads(
            args.trace.read_text(encoding='utf-8'))
        benchmark_revision = benchmark['metadata']['git_revision']
        trace_revision = trace_payload['metadata']['git_revision']
        if benchmark_revision != trace_revision:
            raise RuntimeError(
                'benchmark and trace revisions do not match')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_notebook(benchmark, trace_payload),
        encoding='utf-8',
    )
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
