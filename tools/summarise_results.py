"""Summarise saved measurements without retraining or changing any source data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/team_04'
URL='https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev'


def main():
    summary=json.loads((RESULTS/'summary.json').read_text())
    selected=summary['selected']
    info=json.loads((RESULTS/'model_info.json').read_text())
    labels={
        'windows_local_keepalive':'Windows localhost, keepalive',
        'windows_local_no_keepalive':'Windows localhost, no keepalive',
        'codespace_local_keepalive':'Codespace localhost, keepalive',
        'codespace_local_no_keepalive':'Codespace localhost, no keepalive',
        'public_keepalive':'Laptop to Codespace, keepalive',
        'public_no_keepalive':'Laptop to Codespace, no keepalive',
    }
    frames={}
    measurements={}
    for folder,label in labels.items():
        paths=list((RESULTS/'benchmarks'/folder).glob('bench_*.csv'))
        if not paths:
            raise RuntimeError(f'Missing benchmark: {folder}')
        frame=pd.read_csv(sorted(paths)[-1])
        if len(frame)!=200:
            raise RuntimeError(f'{folder} has {len(frame)} successful readings, expected 200')
        frames[folder]=frame
        measurements[folder]={
            'label':label,'requests':len(frame),'first_request_ms':float(frame.round_trip.iloc[0]),
            'percentiles_ms':{column:{str(p):float(np.percentile(frame[column].iloc[1:],p))
                                     for p in (50,95,99)} for column in frame.columns if column!='i'},
        }
    (RESULTS/'benchmarks/comparison.json').write_text(json.dumps(measurements,indent=2))
    lines=['# Team 4 - EdgeX: assignment results','',
           f'Public endpoint: {URL}', '', 'Repository: https://github.com/YashP001/EdgeAI', '',
           '## Data and baselines','',
           'All 2,000 original rows were retained: 1,600 training and 400 validation. The CSV is byte-for-byte unchanged. The audit flags 24 sensor sentinel readings, but removes zero rows at the team\'s request. This intentionally differs from the README instruction to remove glitches. API validation still rejects broken incoming readings.', '',
           '| Model | Validation MAE (days) |','|---|---:|',
           f'| Mean baseline | {summary["baselines"]["mean_validation_mae"]:.4f} |',
           f'| Linear least squares | {summary["baselines"]["linear_validation_mae"]:.4f} |']
    starter=json.loads((RESULTS/'original_starter/config.json').read_text())
    lines += [f'| Original instructor starter, unchanged | {starter["val_mae"]:.2f} |','',
              'The original starter was run separately using the instructor ZIP and this team\'s CSV. It did not overwrite the submission model. The controlled sweep below uses a learning-rate schedule and restores each run\'s best validation checkpoint, so its unnormalised run is a separate experiment.', '',
              'The relationships plot shows strong voltage curvature and declining battery life as ringing frequency increases. Linear regression misses high-life readings most severely: CSV row 1505 has 225.8 actual days versus 118.48 predicted days, a 107.32-day error. The ten largest errors are recorded in linear_worst_errors.csv.', '',
              '## Architecture and normalisation comparisons','',
              '| Hidden layers | Normalised | Parameters | Training MAE | Validation MAE | Best epoch |',
              '|---|---|---:|---:|---:|---:|']
    for run in summary['experiments']:
        lines.append(f'| {" -> ".join(map(str,run["hidden"]))} | {run["normalise"]} | {run["params"]} | {run["train_mae"]:.4f} | {run["val_mae"]:.4f} | {run["best_epoch"]} |')
    lines += ['',f'The saved model is 5 -> 64 -> 64 -> 1, restored from epoch {selected["best_epoch"]}. Its training MAE is {selected["train_mae"]:.4f} days and validation MAE is {selected["val_mae"]:.4f} days. Later epochs reduce training error while validation worsens; the earlier checkpoint limits overfitting. Normalisation statistics use training rows only. Validation selected both architecture and checkpoint; the hidden-test error is unknown. A provisional forecast is about 4 days MAE.', '',
              '## Per-layer memory and work','',
              '| Layer | Parameters | Float32 bytes | MACs |','|---|---:|---:|---:|']
    for layer in info['layers']:
        lines.append(f'| {layer["in"]} -> {layer["out"]} | {layer["params"]} | {layer["bytes_f32"]} | {layer["macs"]} |')
    lines += [f'| Total | {info["total_params"]} | {info["total_bytes_f32"]} | {info["total_macs"]} |','',
              'Each dense layer has input_width * output_width weights plus output_width biases. Each connection costs one MAC by the assignment convention. The starter 5 -> 32 -> 32 -> 1 counts are 192 + 1056 + 33 = 1281 parameters, 5124 float32 bytes and 1216 MACs.', '',
              '## Request timing','',
              'All modes used the supplied, unchanged client/bench.py with 200 requests. The first request is separate; p50/p95/p99 exclude it. These first requests follow a health request and are not fresh-process cold-start measurements.', '',
              '| Mode | First request ms | Round trip p50 | p95 | p99 | Inference p50 | Server total p50 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for mode,result in measurements.items():
        p=result['percentiles_ms']
        lines.append(f'| {result["label"]} | {result["first_request_ms"]:.3f} | {p["round_trip"]["50"]:.3f} | {p["round_trip"]["95"]:.3f} | {p["round_trip"]["99"]:.3f} | {p["inference"]["50"]:.3f} | {p["server_total"]["50"]:.3f} |')
    lines += ['', 'Every stage\'s p50/p95/p99 and all raw requests are preserved in benchmarks/comparison.json and the per-mode CSVs. Network plus framework is round trip minus server_total; it includes response serialization and routing overhead as well as transport.', '',
              'The LLM forecast was that transport/HTTP would dominate remote inference, with 0.1-1 ms inference and 50-300 ms remote round trip. Compare those guesses with the measured percentiles above. Connection reuse avoids reconnecting and TLS setup for every remote request. Localhost omits the internet/TLS tunnel but still includes HTTP and framework work. Different machines and run conditions also affect results; these comparisons do not isolate hardware speed alone.', '',
              '| Location | 400 single calls ms | One 400-row batch ms | Speedup |','|---|---:|---:|---:|']
    for name,file in [('Windows localhost','local_endpoint_check.json'),('Codespace localhost','codespace_endpoint_check.json'),('Laptop to Codespace','public_endpoint_check.json')]:
        result=json.loads((RESULTS/file).read_text())
        lines.append(f'| {name} | {result["single_400_ms"]:.3f} | {result["batch_400_ms"]:.3f} | {result["batch_speedup"]:.1f}x |')
    lines += ['', 'Batching amortises HTTP/tunnel overhead and uses one array and network call for many readings. The generated readings in these checks test API behaviour, not predictive accuracy.', '',
              '## Doorbell memory budget','',
              'Memory budgets use 300 * 1024 SRAM bytes and 8 * 1024 * 1024 PSRAM bytes. The flags concern parameter storage. Runtime allocations, stack and scratch space also need memory. Int8 is a theoretical one-byte-per-parameter estimate; no quantised model has been exported or accuracy-tested.', '',
              '| Width w in 5 -> w -> w -> 1 | Parameters | Float32 bytes | Int8 bytes | SRAM f32 | SRAM int8 | PSRAM f32 | PSRAM int8 | MACs |',
              '|---|---:|---:|---:|---|---|---|---|---:|']
    hardware=pd.read_csv(ROOT/'results/hardware/widths.csv')
    for _,row in hardware.iterrows():
        lines.append(f'| {int(row.width)} | {int(row.total_params)} | {int(row.total_bytes_f32)} | {int(row.total_bytes_int8)} | {row.fits_300kb_sram_f32} | {row.fits_300kb_sram_int8} | {row.fits_8mb_psram_f32} | {row.fits_8mb_psram_int8} | {int(row.total_macs)} |')
    lines += ['', 'P(w) = w*w + 8*w + 1. Float32: w=273 uses 306856 bytes and fits, while w=274 uses 309076 bytes and fails. Int8: w=550 uses 306901 bytes and fits, while w=551 uses 308010 bytes and fails. Thus the first failing widths are 274 and 551.', '',
              'The selected model uses 18436 float32 parameter bytes (4609 theoretical int8 bytes), and at most 128 simultaneous dense-layer input/output activation values: 512 float32 bytes or 128 int8 bytes. It fits the stated SRAM budget by a wide margin before runtime overhead.', '',
              'At an assumed conservative 10 million MAC/s, 4480 MACs take 0.448 ms of compute. This is an estimate, not an ESP32 measurement. Espressif\'s [ESP-NN benchmark](https://github.com/espressif/esp-nn/blob/master/docs/performance.md) lists a 271-by-3 int8 fully connected kernel at 8439 ANSI C cycles or 1543 optimised cycles at 240 MHz, about 23 or 126 million MAC/s for that shape. The lower assumption leaves margin; it is not a measured or guaranteed rate for this model, float32 code, preprocessing or invocation overhead.', '',
              'On-device inference is a reasonable production direction: it avoids remote latency and per-request cloud costs, keeps readings local, and works when Wi-Fi is down. It also avoids a radio transmission for each prediction, although net energy savings require board measurements. Cloud inference simplifies updates and central monitoring. The public cloud endpoint is still required for this assignment.', '',
              '## Submission and team review','',
              f'Submit the repository and verified endpoint address {URL} using the channel the instructor announces. Keep the Codespace and server running with port 8000 Public during the announced check window; verify visibility after restarts. Deadline: Monday, 23 November 2026.', '',
              'The team still needs to review and understand the code, rotate the assignment roles, add collaborators using their GitHub usernames, and submit via the announced channel. Those human actions are not represented as completed by these automated measurements. Pre-experiment forecasts in forecasts.txt are labelled LLM estimates, not invented student observations.']
    (ROOT/'ASSIGNMENT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Additive timing diagram uses means, because separately computed medians do not add.
    fig,ax=plt.subplots(figsize=(11,4.2))
    chosen_modes=['codespace_local_keepalive','public_keepalive']
    names=['Codespace localhost','Laptop to Codespace']
    left=np.zeros(2)
    for stage in ['parse','normalise','inference','build','other_server','network_and_framework']:
        values=[]
        for mode in chosen_modes:
            frame=frames[mode].iloc[1:]
            value=(frame.server_total-frame[['parse','normalise','inference','build']].sum(axis=1)).mean() if stage=='other_server' else frame[stage].mean()
            values.append(max(0,float(value)))
        ax.barh(names,values,left=left,label=stage)
        left+=values
    ax.set(xlabel='Mean milliseconds per warm request',title='Where the request time goes: actual Codespaces measurements')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(RESULTS/'benchmarks/remote_timing_breakdown.png',dpi=150,bbox_inches='tight');plt.close(fig)
    print('Wrote ASSIGNMENT_RESULTS.md, comparison.json and remote_timing_breakdown.png')


if __name__=='__main__':
    main()
