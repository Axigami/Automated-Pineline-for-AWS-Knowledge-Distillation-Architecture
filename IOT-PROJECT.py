import json, boto3, numpy as np, os, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "tf-endpoint")
OUTPUT_BUCKET      = os.environ.get("OUTPUT_BUCKET",      "anomalytraffic")
REGION             = os.environ.get("REGION",             "ap-southeast-2")

SCALER_BUCKET = "anomalytraffic"
SCALER_KEY    = "data/raw/log/scaler_stats.json"

ROUTE_MAP = {
    "data/anomalies/anomaly/": {
        "s3_prefix":      "predictions/anomalyprediction/",
        "dynamo_table":   "AnomalyPredictions",
        "expected_type":  "attack",  
        "conflict_rule":  "anomaly_source_must_be_attack"
    },
    "data/raw/log/": {
        "s3_prefix":      "predictions/logprediction/",
        "dynamo_table":   "LogPredictions",
        "expected_type":  "benign", 
        "conflict_rule":  "log_source_must_be_benign"
    },
}

CONFLICTS_TABLE = os.environ.get("CONFLICTS_TABLE", "anomaly-conflicts")

def get_route(key: str) -> dict:
    for prefix, config in ROUTE_MAP.items():
        if key.startswith(prefix):
            logger.info(f"Route matched: {prefix} -> {config}")
            return config
    logger.warning(f"No route matched for key: {key}, using default")
    return {
        "s3_prefix": "predictions/unknown/", 
        "dynamo_table": "AnomalyPredictions",
        "expected_type": None,
        "conflict_rule": None
    }

LABEL_MAP = {0: "Benign", 1: "Botnet", 2: "DDoS", 3: "DoS", 4: "PortScan"}

APP_CAT_MAP = {
    "Advertisement":  0,  "Chat":           1,  "Cloud":          2,
    "Collaborative":  3,  "ConnCheck":      4,  "Cybersecurity":  5,
    "DataTransfer":   6,  "Database":       7,  "Download":       8,
    "Email":          9,  "Game":          10,  "IoT-Scada":     11,
    "Media":         12,  "Mining":        13,  "Music":         14,
    "Network":       15,  "RPC":           16,  "RemoteAccess":  17,
    "Shopping":      18,  "SocialNetwork": 19,  "SoftwareUpdate":20,
    "Streaming":     21,  "System":        22,  "Unspecified":   23,
    "VPN":           24,  "Video":         25,  "VoIP":          26,
    "Web":           27,
}

_scaler   = None
_sm       = None
_dynamodb = None

def get_scaler() -> dict:
    global _scaler
    if _scaler is None:
        obj     = boto3.client("s3", region_name=REGION).get_object(
                      Bucket=SCALER_BUCKET, Key=SCALER_KEY)
        _scaler = json.loads(obj["Body"].read())
        logger.info(f"Scaler loaded: {_scaler['n_features']} features")
    return _scaler

def _g(flow: dict, key: str, default=0.0) -> float:
    try:
        v = float(flow.get(key, default))
        return 0.0 if (v != v) else v
    except (TypeError, ValueError):
        return float(default)

def _port_bucket(p) -> int:
    try:
        p = int(p)
    except (TypeError, ValueError):
        return 2
    if p <= 1023:  return 0
    if p <= 49151: return 1
    return 2

def engineer_features(flow: dict) -> dict:
    bi_pkts   = _g(flow, "bidirectional_packets")
    bi_bytes  = _g(flow, "bidirectional_bytes")
    s2d_pkts  = _g(flow, "src2dst_packets")
    d2s_pkts  = _g(flow, "dst2src_packets")
    s2d_bytes = _g(flow, "src2dst_bytes")
    d2s_bytes = _g(flow, "dst2src_bytes")

    feat = {}
    for k, v in flow.items():
        if isinstance(v, (int, float)):
            feat[k] = float(v)

    feat["pkt_per_byte_ratio"]  = bi_pkts  / (bi_bytes  + 1e-9)
    feat["flow_symmetry"]       = s2d_pkts / (s2d_pkts + d2s_pkts + 1e-9)
    feat["byte_symmetry"]       = s2d_bytes/ (s2d_bytes + d2s_bytes + 1e-9)

    feat["bidirectional_syn_ratio"] = _g(flow, "bidirectional_syn_packets") / (bi_pkts + 1e-9)
    feat["src2dst_syn_ratio"]       = _g(flow, "src2dst_syn_packets")       / (s2d_pkts + 1e-9)
    feat["dst2src_syn_ratio"]       = _g(flow, "dst2src_syn_packets")       / (d2s_pkts + 1e-9)

    for flag in ["ack", "psh", "rst", "fin", "cwr", "ece", "urg"]:
        feat[f"bidirectional_{flag}_ratio"] = \
            _g(flow, f"bidirectional_{flag}_packets") / (bi_pkts + 1e-9)

    dp = int(_g(flow, "dst_port"))
    feat["dst_port_bucket"]        = float(_port_bucket(dp))
    feat["dst_port_is_well_known"] = float(feat["dst_port_bucket"] == 0)

    cat_raw = flow.get("application_category_name", "Unspecified")
    feat["application_category_name"] = float(APP_CAT_MAP.get(str(cat_raw), -1))
    feat["application_confidence"]    = _g(flow, "application_confidence")

    return feat

def build_vector(flow: dict) -> np.ndarray:
    feature_names = get_scaler()["feature_names"]
    engineered    = engineer_features(flow)
    vec = np.array(
        [engineered.get(col, 0.0) for col in feature_names],
        dtype=np.float32
    )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

def standardize(vec: np.ndarray) -> np.ndarray:
    s     = get_scaler()
    mean  = np.array(s["mean"],  dtype=np.float32)
    scale = np.array(s["scale"], dtype=np.float32)
    return (vec - mean) / (scale + 1e-9)

def build_payload(vec_scaled: np.ndarray) -> str:
    sample = [[float(v)] for v in vec_scaled]
    return json.dumps({"inputs": [sample]})

def get_sm():
    global _sm
    if _sm is None:
        _sm = boto3.client("sagemaker-runtime", region_name=REGION)
    return _sm

def predict(vec_scaled: np.ndarray) -> dict:
    payload = build_payload(vec_scaled)
    resp    = get_sm().invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(resp["Body"].read())
    preds  = result.get("predictions") or result.get("outputs") or []

    if not preds:
        logger.error(f"Empty predictions. Raw: {result}")
        return {"error": "empty predictions", "raw": result}

    probs     = np.array(preds[0], dtype=np.float32)
    class_idx = int(np.argmax(probs))

    return {
        "label":       LABEL_MAP.get(class_idx, f"class_{class_idx}"),
        "confidence":  round(float(probs[class_idx]), 4),
        "probabilities": {
            LABEL_MAP.get(i, f"c{i}"): round(float(p), 4)
            for i, p in enumerate(probs)
        },
    }

def detect_conflict(prediction: dict, route: dict, flow: dict, source_key: str) -> bool:
    expected_type = route.get("expected_type")
    if not expected_type:
        return False
    
    predicted_label = prediction.get("label", "")
    is_benign = (predicted_label == "Benign")
    
    if expected_type == "attack" and is_benign:
        logger.warning(
            f"⚠️ CONFLICT! Source expects ATTACK but got Benign\n"
            f"   Source: {source_key}\n"
            f"   Prediction: {predicted_label}"
        )
        save_conflict(
            flow=flow,
            expected_label="attack",
            actual_prediction=prediction,
            conflict_reason=f"Anomaly source predicted as {predicted_label} (expected attack)",
            conflict_rule=route.get("conflict_rule"),
            source_key=source_key
        )
        return True
    
    if expected_type == "benign" and not is_benign:
        logger.warning(
            f"⚠️ CONFLICT! Source expects BENIGN but got {predicted_label}\n"
            f"   Source: {source_key}\n"
            f"   Prediction: {predicted_label}"
        )
        save_conflict(
            flow=flow,
            expected_label="Benign",
            actual_prediction=prediction,
            conflict_reason=f"Log source predicted as {predicted_label} (expected Benign)",
            conflict_rule=route.get("conflict_rule"),
            source_key=source_key
        )
        return True
    
    return False

def save_conflict(flow: dict, expected_label: str, actual_prediction: dict, 
                  conflict_reason: str, conflict_rule: str, source_key: str):
    try:
        table = get_dynamo().Table(CONFLICTS_TABLE)
        
        conflict_id = str(uuid.uuid4())
        now = int(datetime.now(timezone.utc).timestamp())
        
        flow_data_json = json.dumps(flow, ensure_ascii=False)
        
        conflict_item = {
            'conflict_id': conflict_id,
            'created_at': now,
            'status': 'pending',
            'flow_data': flow_data_json,
            'expected_label': expected_label,
            'actual_prediction': json.dumps(actual_prediction),
            'conflict_reason': conflict_reason,
            'conflict_rule': conflict_rule,
            'source_key': source_key,
            'device_id': flow.get('device_id', 'unknown'),
            'flow_id': flow.get('id', 0),
            'src_ip': flow.get('src_ip', ''),
            'dst_ip': flow.get('dst_ip', ''),
            'protocol': flow.get('protocol', 0),
            'timestamp': flow.get('timestamp', now)
        }
        
        table.put_item(Item=conflict_item)
        
        logger.info(
            f"💾 Conflict saved: {conflict_id}\n"
            f"   Expected: {expected_label}\n"
            f"   Actual: {actual_prediction.get('label')}\n"
            f"   Confidence: {actual_prediction.get('confidence')}"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to save conflict: {e}", exc_info=True)

def write_s3(source_key: str, payload: dict, s3_prefix: str) -> str:
    fname    = source_key.split("/")[-1].replace(".json", "_pred.json")
    dest_key = f"{s3_prefix}{fname}"
    boto3.client("s3", region_name=REGION).put_object(
        Bucket=OUTPUT_BUCKET,
        Key=dest_key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )
    logger.info(f"S3 written -> s3://{OUTPUT_BUCKET}/{dest_key}")
    return dest_key

def get_dynamo():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return _dynamodb

def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(i) for i in obj]
    return obj

def write_dynamodb(flow_result: dict, dynamo_table: str):
    table     = get_dynamo().Table(dynamo_table)
    device_id = str(flow_result.get("device_id") or "unknown")
    timestamp = flow_result.get("timestamp") or 0
    flow_id   = flow_result.get("flow_id")   or 0
    pred      = flow_result.get("prediction", {})

    item = {
        "device_id":     device_id,
        "sk":            f"{timestamp}#{flow_id}#{uuid.uuid4().hex[:8]}",
        "timestamp":     int(timestamp),
        "flow_id":       int(flow_id),
        "src_ip":        str(flow_result.get("src_ip")   or ""),
        "dst_ip":        str(flow_result.get("dst_ip")   or ""),
        "src_port":      int(flow_result.get("src_port") or 0),
        "dst_port":      int(flow_result.get("dst_port") or 0),
        "protocol":      int(flow_result.get("protocol") or 0),
        "label":         str(pred.get("label", "ERROR")),
        "confidence":    float_to_decimal(pred.get("confidence", 0.0)),
        "probabilities": float_to_decimal(pred.get("probabilities", {})),
        "processed_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "error":         str(pred.get("error", "")),
    }

    table.put_item(Item=item)
    logger.info(f"DynamoDB [{dynamo_table}] written: {device_id} / {timestamp}#{flow_id} -> {item['label']}")

# ─────────────────────────────────────────────────────────────────────────────
# Lambda Handler
# ─────────────────────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    results = []
    total_conflicts = 0

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key    = record["s3"]["object"]["key"]
        logger.info(f"--- s3://{bucket}/{key} ---")

        route        = get_route(key)
        s3_prefix    = route["s3_prefix"]
        dynamo_table = route["dynamo_table"]
        logger.info(f"Route: S3={s3_prefix} | DynamoDB={dynamo_table} | Expected={route.get('expected_type')}")

        try:
            obj  = boto3.client("s3", region_name=REGION).get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read().decode("utf-8").strip()
        except Exception as e:
            logger.error(f"S3 read error: {e}")
            results.append({"key": key, "error": str(e)})
            continue

        docs = []
        for line in body.splitlines():
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skip invalid line: {e}")

        if not docs:
            results.append({"key": key, "error": "no valid JSON"})
            continue

        flow_results = []
        conflicts_in_batch = 0
        
        for doc in docs:
            flow = doc.get("flow", doc)

            try:
                vec        = build_vector(flow)
                vec_scaled = standardize(vec)
                pred       = predict(vec_scaled)
            except Exception as e:
                logger.error(f"Flow {flow.get('id')} error: {e}", exc_info=True)
                pred = {"error": str(e)}

            if "error" not in pred:
                is_conflict = detect_conflict(pred, route, flow, key)
                if is_conflict:
                    conflicts_in_batch += 1
                    total_conflicts += 1

            flow_out = {
                "device_id": doc.get("device_id"),
                "timestamp": doc.get("timestamp"),
                "flow_id":   flow.get("id"),
                "src_ip":    flow.get("src_ip"),
                "dst_ip":    flow.get("dst_ip"),
                "src_port":  flow.get("src_port"),
                "dst_port":  flow.get("dst_port"),
                "protocol":  flow.get("protocol"),
                "prediction": pred,
            }
            flow_results.append(flow_out)

            try:
                write_dynamodb(flow_out, dynamo_table)
            except Exception as e:
                logger.error(f"DynamoDB write error flow {flow.get('id')}: {e}")

            logger.info(
                f"  flow={flow.get('id')} "
                f"-> {pred.get('label','ERR')} "
                f"({pred.get('confidence', 0):.1%})"
            )

        output_doc = {
            "source":      f"s3://{bucket}/{key}",
            "total_flows": len(flow_results),
            "flows":       flow_results,
        }
        try:
            dest = write_s3(key, output_doc, s3_prefix)
            results.append({
                "key": key, 
                "dest": dest, 
                "count": len(flow_results),
                "conflicts": conflicts_in_batch
            })
        except Exception as e:
            logger.error(f"S3 write error: {e}")
            results.append({"key": key, "error": str(e)})

    logger.info(f"=" * 60)
    logger.info(f"✅ Processed {len(results)} batches")
    logger.info(f"⚠️ Total conflicts detected: {total_conflicts}")
    logger.info(f"=" * 60)

    return {
        "statusCode": 200, 
        "body": json.dumps(results),
        "total_conflicts": total_conflicts
    }