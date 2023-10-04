# Benchmark Embeddings

model: `BAAI/bge-base-en-v1.5`
sequence tokens: `512` | `256`
dtype: `bf16`
Accelerator: 1x inferentia2 chip (2x neuron cores)
Other References: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/benchmarks/inf2/inf2-performance.html#inf2-performance
Documentation: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuronperf/neuronperf_api.html

**Preperation:**
```bash
pip install neuronperf --extra-index-url=https://pip.repos.neuron.amazonaws.com
```

**Definitions:**

* `n_models` – The number of models to run in parallel. Default behavior runs 1 model and the max number of models possible, determined by a best effort from device_type, instance size, or other environment state.
* `workers_per_model` - controls how many worker threads will be feeding inputs to each model. The default is automatically determined.


# Execution

## Sequence Length 256


### BS1

```bash
python benchmark_embeddings.py --model_name_or_path BAAI/bge-base-en-v1.5 --batch_size 1 --sequence_length 256
```

Results: 
* batch size 1 => p99 2.0ms ; 966 rps ; 2 replica ; 0.218$ per 1M requests

| batch_size | sequence length | latency_ms_p50 | latency_ms_p99 | throughput_avg | n_models | workers_per_model | cost_per_1m_inf |
| ---------- | --------------- | -------------- | -------------- | -------------- | -------- | ----------------- | --------------- |
| 1          | 256             | 1.953          | 2.036          | 966.409        | 2        | 1                 | 0.218           |
| 1          | 256             | 1.834          | 1.863          | 542.479        | 1        | 1                 | 0.389           |
| 1          | 256             | 3.818          | 3.887          | 931.985        | 2        | 2                 | 0.227           |
| 1          | 256             | 3.465          | 3.528          | 574.395        | 1        | 2                 | 0.368           |

### BS32

```bash
python benchmark_embeddings.py --model_name_or_path BAAI/bge-base-en-v1.5 --batch_size 32 --sequence_length 256
```

Results:
* batch size 32 => p99 61.8 ms ; 1001 rps ; 2 replica ; 0.211$ per 1M requests

| batch_size | sequence length | throughput_avg | latency_ms_p50 | latency_ms_p99 | n_models | workers_per_model | cost_per_1m_inf |
| ---------- | --------------- | -------------- | -------------- | -------------- | -------- | ----------------- | --------------- |
| 32         | 256             | 1001.071       | 60.227         | 61.806         | 2        | 1                 | 0.211           |
| 32         | 256             | 595.899        | 53.687         | 54.642         | 1        | 1                 | 0.354           |
| 32         | 256             | 1053.228       | 114.352        | 116.63         | 2        | 2                 | 0.2             |
| 32         | 256             | 596.437        | 107.173        | 109.173        | 1        | 2                 | 0.354           |

## Sequence Length 512

### BS1


```bash
python benchmark_embeddings.py --model_name_or_path BAAI/bge-base-en-v1.5 --batch_size 1 --sequence_length 512
```

Results: 
* batch size 1 => p99 5.1ms ; 364 rps ; 2 replica ; 0.578$ per 1M requests


| batch_size | sequence length | latency_ms_p50 | latency_ms_p99 | throughput_avg | n_models | workers_per_model | cost_per_1m_inf |
| ---------- | --------------- | -------------- | -------------- | -------------- | -------- | ----------------- | --------------- |
| 1          | 512             | 5.112          | 5.286          | 364.984        | 2        | 1                 | 0.578           |
| 1          | 512             | 10.136         | 10.342         | 371.016        | 2        | 2                 | 0.569           |
| 1          | 512             | 4.816          | 4.835          | 207.286        | 1        | 1                 | 1.018           |
| 1          | 512             | 9.461          | 9.473          | 211.336        | 1        | 2                 | 0.999           |


### BS 16

```bash
python benchmark_embeddings.py --model_name_or_path BAAI/bge-base-en-v1.5 --batch_size 16 --sequence_length 512
```

Results:
* batch size 16 => p99 91.5 ms ; 341.8 rps ; 2 replica ; 0.618$ per 1M requests

| batch_size | sequence length | latency_ms_p50 | latency_ms_p99 | throughput_avg | n_models | workers_per_model | cost_per_1m_inf |
| ---------- | --------------- | -------------- | -------------- | -------------- | -------- | ----------------- | --------------- |
| 16         | 512             | 87.581         | 91.47          | 341.795        | 2        | 1                 | 0.618           |
| 16         | 512             | 84.876         | 85.872         | 188.504        | 1        | 1                 | 1.12            |
| 16         | 512             | 176.4          | 181.256        | 340.787        | 2        | 2                 | 0.619           |
| 16         | 512             | 169.534        | 171.308        | 188.773        | 1        | 2                 | 1.118           |

### BS 32

```bash
python benchmark_embeddings.py --model_name_or_path BAAI/bge-base-en-v1.5 --batch_size 32 --sequence_length 512
```

Results: 
* batch size 32 => p99 170.9 ms ; 352.0 rps ; 2 replica ; 0.6$ per 1M requests

| batch_size | sequence length | latency_ms_p50 | latency_ms_p99 | throughput_avg | n_models | workers_per_model | cost_per_1m_inf |
| ---------- | --------------- | -------------- | -------------- | -------------- | -------- | ----------------- | --------------- |
| 32         | 512             | 170.861        | 175.699        | 352.0          | 2        | 1                 | 0.6             |
| 32         | 512             | 164.693        | 167.176        | 194.42         | 1        | 1                 | 1.086           |
| 32         | 512             | 342.75         | 350.512        | 349.75         | 2        | 2                 | 0.604           |
| 32         | 512             | 329.216        | 334.95         | 194.42         | 1        | 2                 | 1.086           |