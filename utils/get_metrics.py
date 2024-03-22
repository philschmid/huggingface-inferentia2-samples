import time
import re

import pandas as pd


def convert_string_to_float_ms(input_string):
    if input_string[-2:] == "µs":
        return float(input_string[:-2]) / 1000
    elif input_string[-2:] == "ms":
        return float(input_string[:-2])
    elif input_string[-1:] == "s":
        return float(input_string[:-1]) * 1000


def extract_metrics(input_string):
    kpi_pattern = r'#\d+\[3m(\w+)#\d+\[0m#\d+\[2m=#\d+\[0m"([^"]+)"'
    kpis = re.findall(kpi_pattern, input_string)
    kpi_dict = dict(kpis)

    try:
        parsed_kpis = {
            "total_time_ms": convert_string_to_float_ms(kpi_dict["total_time"]),
            "inference_time_ms": convert_string_to_float_ms(kpi_dict["inference_time"]),
            "time_per_token_ms": convert_string_to_float_ms(kpi_dict["time_per_token"]),
            "queue_time_ms": convert_string_to_float_ms(kpi_dict["queue_time"]),
        }
    except:
        print(input_string)
        raise
    return parsed_kpis


def get_metrics_from_cloudwatch(
    endpoint_name=None,
    st=None,
    et=None,
    cu=None,
    boto3_session=None,
    total_requests=None,
    generated_tokens=250,
):
    print("Waiting for logs to be available ...")
    time.sleep(120)
    log_end_time = time.time()
    client = boto3_session.client("logs")

    loggroup = f"/aws/sagemaker/Endpoints/{endpoint_name}"

    start_query_response = client.start_query(
        logGroupName=loggroup,
        startTime=st,
        endTime=int(log_end_time),
        queryString="fields @message | sort @timestamp desc",
        limit=10000,
    )
    query_id = start_query_response["queryId"]

    response = None

    while response == None or response["status"] == "Running":
        print("Waiting for query to complete ...")
        time.sleep(1)
        response = client.get_query_results(queryId=query_id)
    metrics = []
    for record in response["results"]:
        if "3mtotal_time" in record[0]["value"]:
            metrics.append(extract_metrics(record[0]["value"]))

    if len(metrics) == 0:
        raise Exception("No metrics found")

    df = pd.DataFrame.from_records(metrics)

    throughput_gen_per_s = total_requests / (et - st) * generated_tokens

    # calculate the average inference time
    inference_time = {
        "Number of requests": len(df),
        "Concurrent Requests": int(cu),
        "Thorughput (tokens/second)": throughput_gen_per_s,
        "Latency (ms/token) min": df["time_per_token_ms"].min(),
        "Latency (ms/token) p(50)": df["time_per_token_ms"].median(),
        "Latency (ms/token) p(90)": df["time_per_token_ms"].quantile(0.9),
        "Latency Request ms min": df["total_time_ms"].min(),
        "Latency Request ms p(50)": df["total_time_ms"].median(),
        "Latency Request ms p(90)": df["total_time_ms"].quantile(0.9),
        "Latency Infernece ms min": df["inference_time_ms"].min(),
        "Latency Infernece ms p(50)": df["inference_time_ms"].median(),
        "Latency Infernece ms p(90)": df["inference_time_ms"].quantile(0.9),
        "Queue time ms min": df["queue_time_ms"].min(),
        "Queue time ms p(50)": df["queue_time_ms"].median(),
        "Queue time ms p(90)": df["queue_time_ms"].quantile(0.9),
    }
    return inference_time
