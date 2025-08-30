# -*- coding: utf-8 -*-
import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse
from functools import wraps

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# from aliyunsdkcore.client import AcsClient
# from aliyunsdkram.request.v20150501 import CreateUserRequest, CreateLoginProfileRequest, GetUserRequest

# Configure logging with better format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("infer")
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Error handling decorator
def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error in {f.__name__}: {e}")
            return jsonify({"error": f"Invalid input: {str(e)}"}), 400
        except FileNotFoundError as e:
            logger.error(f"File not found in {f.__name__}: {e}")
            return jsonify({"error": f"Resource not found: {str(e)}"}), 404
        except Exception as e:
            logger.exception(f"Unexpected error in {f.__name__}: {e}")
            return jsonify({"error": "Internal server error"}), 500
    return decorated_function

# -------- ENV --------
REGION = os.getenv("OTS_REGION", "cn-hangzhou").strip()
OTS_INSTANCE = os.getenv("OTS_INSTANCE", "").strip()
OTS_TABLE = os.getenv("OTS_TABLE", "BirdFiles").strip()
OTS_ENDPOINT = os.getenv("OTS_ENDPOINT", f"https://{OTS_INSTANCE}.{REGION}.vpc.tablestore.aliyuncs.com").strip()

BUCKET = os.getenv("BUCKET", "").strip()
OSS_REGION = os.getenv("OSS_REGION", REGION).strip()
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", f"https://oss-{OSS_REGION}.aliyuncs.com").strip()

UPLOADS_PREFIX = os.getenv("UPLOADS_PREFIX", "uploads/").strip()
THUMBS_PREFIX = os.getenv("THUMBS_PREFIX", "thumbs/").strip()
PREVIEW_PREFIX = os.getenv("PREVIEW_PREFIX", "preview/").strip()

MODEL_PATH = os.getenv("MODEL_PATH", "/code/bird_detection/model.pt").strip()
try:
    CONFIDENCE = float(os.getenv("CONFIDENCE", "0.5"))
    if not 0.0 <= CONFIDENCE <= 1.0:
        raise ValueError("CONFIDENCE must be between 0.0 and 1.0")
except ValueError as e:
    logger.error(f"Invalid CONFIDENCE value: {e}")
    CONFIDENCE = 0.5

# Validate required environment variables
if not BUCKET:
    logger.warning("BUCKET environment variable is not set")
if not OTS_INSTANCE:
    logger.warning("OTS_INSTANCE environment variable is not set")

# Performance tuning
MAX_SCAN_LIMIT = int(os.getenv("MAX_SCAN_LIMIT", "2000"))
DEFAULT_SCAN_LIMIT = int(os.getenv("DEFAULT_SCAN_LIMIT", "100"))
THUMB_MAX_SIZE = int(os.getenv("THUMB_MAX_SIZE", "256"))
PREVIEW_MAX_SIZE = int(os.getenv("PREVIEW_MAX_SIZE", "1024"))

# 超时和性能配置
MAX_PROCESSING_TIME = int(os.getenv("MAX_PROCESSING_TIME", "280"))  # 最大处理时间（秒）
INFER_MAX_SIZE = int(os.getenv("INFER_MAX_SIZE", "1280"))  # 推理时最大图像尺寸
ENABLE_MODEL_WARMUP = os.getenv("ENABLE_MODEL_WARMUP", "true").lower() == "true"
LOG_PERFORMANCE = os.getenv("LOG_PERFORMANCE", "true").lower() == "true"

# 设置Ultralytics配置目录
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/tmp")

# 禁用一些可能导致问题的功能
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("ULTRALYTICS_ANALYTICS", "False")

# Suggested envs in FC for stability:
# YOLO_CONFIG_DIR=/tmp
# OPENBLAS_NUM_THREADS=1
# OMP_NUM_THREADS=1
# MKL_NUM_THREADS=1

# -------- LAZY SINGLETONS --------
_OSS_BUCKET = None
_OTS_CLIENT = None
_MODEL = None
_MODEL_NAMES: Dict[int, str] = {}

def _fc_creds():
    return (
        os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
        os.getenv("ALIBABA_CLOUD_SECURITY_TOKEN"),
    )

# Lazy imports
def _cv():
    import cv2 as _cv
    return _cv

def _np():
    import numpy as _np
    return _np

def _oss2():
    import oss2 as _oss2
    return _oss2

def _ts():
    import tablestore as _ts
    return _ts

def _yolo():
    from ultralytics import YOLO as _YOLO
    return _YOLO

# -------- CLIENTS --------
def get_oss_bucket():
    global _OSS_BUCKET
    if _OSS_BUCKET is None:
        ak, sk, token = _fc_creds()
        if not all([ak, sk]):
            raise RuntimeError("Missing OSS credentials (AK/SK)")
        if not BUCKET:
            raise RuntimeError("Missing BUCKET environment variable")
        
        auth = _oss2().StsAuth(ak, sk, token) if token else _oss2().Auth(ak, sk)
        _OSS_BUCKET = _oss2().Bucket(auth, OSS_ENDPOINT, BUCKET)
        logger.info(f"OSS client initialized: endpoint={OSS_ENDPOINT}, bucket={BUCKET}")
    return _OSS_BUCKET

def get_ots():
    global _OTS_CLIENT
    if _OTS_CLIENT is None:
        ak, sk, token = _fc_creds()
        if not all([ak, sk]):
            raise RuntimeError("Missing OTS credentials (AK/SK)")
        if not OTS_INSTANCE:
            raise RuntimeError("Missing OTS_INSTANCE environment variable")
        
        ts = _ts()
        # Optimize timeout settings for better performance
        _OTS_CLIENT = ts.OTSClient(
            OTS_ENDPOINT, ak, sk,
            instance_name=OTS_INSTANCE,
            sts_token=token,
            socket_timeout=60,  # Increased for large operations
            retry_times=2,      # Reduced for faster failure detection
            connection_timeout=15,  # Increased for better connection stability
            max_connection=10,  # Add connection pooling
        )
        logger.info(f"OTS client initialized: endpoint={OTS_ENDPOINT}, instance={OTS_INSTANCE}, region={REGION}")
    return _OTS_CLIENT

# -------- IMAGE --------
def _read_image_from_oss(key: str):
    obj = get_oss_bucket().get_object(key)
    data = obj.read()
    arr = _np().frombuffer(data, _np().uint8)
    img = _cv().imdecode(arr, _cv().IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv.imdecode failed")
    return img

def _jpeg_bytes(img_bgr, quality=88) -> bytes:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid image array")
    
    ok, buf = _cv().imencode(".jpg", img_bgr, [int(_cv().IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv.imencode failed")
    return buf.tobytes()

def _put_to_oss(key: str, data: bytes, content_type="image/jpeg"):
    if not data:
        raise ValueError("Empty data provided")
    get_oss_bucket().put_object(key, data, headers={"Content-Type": content_type})
    logger.info(f"put oss ok: {key}")

def _resize_long(img, max_side=1024):
    if img is None or img.size == 0:
        raise ValueError("Invalid image array")
    
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    
    # Calculate new dimensions maintaining aspect ratio
    scale = max_side / max(h, w)
    nw = int(w * scale)
    nh = int(h * scale)
    
    # Use appropriate interpolation based on scaling
    interpolation = _cv().INTER_AREA if scale < 1 else _cv().INTER_CUBIC
    resized = _cv().resize(img, (nw, nh), interpolation=interpolation)
    
    if resized is None:
        raise RuntimeError("Failed to resize image")
    return resized

# -------- YOLO (lazy) --------
def _ensure_model():
    global _MODEL, _MODEL_NAMES
    if _MODEL is None:
        YOLO = _yolo()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
        
        # Set environment variables to avoid configuration directory issues
        os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp')
        os.environ.setdefault('ULTRALYTICS_CONFIG_DIR', '/tmp')
        
        try:
            load_start = time.time()
            logger.info(f"Loading YOLO model from: {MODEL_PATH}")
            
            # 创建模型实例
            _MODEL = YOLO(MODEL_PATH)
            
            # Best-effort disable warmup on predictor if available
            try:
                if hasattr(_MODEL, "predictor"):
                    setattr(_MODEL.predictor, "warmup", lambda *a, **k: None)
            except Exception:
                pass
            
            # 获取类别名称
            names = _MODEL.names
            _MODEL_NAMES = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
            
            # 验证模型
            if not _MODEL_NAMES:
                raise ValueError("Invalid model: missing class names")
            
            load_time = time.time() - load_start
            if LOG_PERFORMANCE:
                logger.info(f"Model loaded in {load_time:.2f}s")
            
            # 模型预热（如果启用）
            if ENABLE_MODEL_WARMUP:
                try:
                    warmup_start = time.time()
                    np = _np()
                    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    _ = _MODEL.predict(dummy_img, verbose=False, conf=0.1, stream=False)
                    warmup_time = time.time() - warmup_start
                    if LOG_PERFORMANCE:
                        logger.info(f"Model warmup completed in {warmup_time:.2f}s")
                except Exception as warmup_error:
                    logger.warning(f"Model warmup failed: {warmup_error}")
            
            logger.info(f"YOLO model ready: {MODEL_PATH} | classes={len(_MODEL_NAMES)}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

def _infer(img_bgr, conf_thresh: float):
    """
    使用YOLO模型进行推理
    
    Args:
        img_bgr: BGR格式的图像数组
        conf_thresh: 置信度阈值
        
    Returns:
        tuple: (species_counts, annotated_image)
    """
    infer_start = time.time()
    
    try:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Invalid image for inference")
        if not 0.0 <= conf_thresh <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        _ensure_model()
        
        # Validate image dimensions
        if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Image must be a 3-channel BGR image")
        
        # Optimize image size for faster inference using new config variable
        h, w = img_bgr.shape[:2]
        original_size = (w, h)
        
        if max(h, w) > INFER_MAX_SIZE:
            scale = INFER_MAX_SIZE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = _cv().resize(img_bgr, (new_w, new_h), interpolation=_cv().INTER_AREA)
            if LOG_PERFORMANCE:
                logger.info(f"Resized image for inference: {w}x{h} -> {new_w}x{new_h}")
        
        # Inference stage with performance monitoring
        predict_start = time.time()
        
        try:
            # Optimized prediction parameters for speed
            preds = _MODEL.predict(
                img_bgr, 
                conf=conf_thresh, 
                stream=False, 
                verbose=False,
                imgsz=INFER_MAX_SIZE,  # Use config variable
                half=False,      # Disable half precision for stability
                device='cpu'     # Explicit CPU usage
            )
        except Exception as e:
            logger.error(f"YOLO prediction failed: {e}")
            # Try fallback with minimal parameters
            try:
                preds = _MODEL(img_bgr, conf=conf_thresh)
            except Exception as e2:
                logger.error(f"Fallback YOLO prediction also failed: {e2}")
                raise RuntimeError(f"Model prediction failed: {e2}") from e2
        
        predict_time = time.time() - predict_start
        if LOG_PERFORMANCE:
            logger.info(f"Prediction completed in {predict_time:.3f}s")
        
        if not preds:
            logger.warning("No predictions returned from model")
            return {}, img_bgr
        
        result = preds[0]
        
        # Species counting with performance monitoring
        count_start = time.time()
        counts: Dict[str, int] = {}
        detections = 0
        
        if result.boxes is not None and result.boxes.cls is not None:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else _np().ones_like(cls_ids, float)
            for c, s in zip(cls_ids, confs):
                if s >= conf_thresh:
                    label = _MODEL_NAMES.get(int(c), str(int(c)))
                    counts[label] = counts.get(label, 0) + 1
                    detections += 1
        
        count_time = time.time() - count_start
        
        # Annotation plotting with performance monitoring
        plot_start = time.time()
        try:
            annotated = result.plot()
        except Exception as e:
            logger.warning(f"Failed to plot annotations: {e}")
            annotated = img_bgr.copy()
        
        plot_time = time.time() - plot_start
        total_time = time.time() - infer_start
        
        if LOG_PERFORMANCE:
            logger.info(f"Inference summary: {detections} detections, {len(counts)} species, "
                       f"total: {total_time:.3f}s (predict: {predict_time:.3f}s, count: {count_time:.3f}s, plot: {plot_time:.3f}s)")
        
        logger.debug(f"Inference completed: detected {len(counts)} species")
        return counts, annotated
        
    except Exception as e:
        total_time = time.time() - infer_start
        logger.error(f"Inference failed after {total_time:.3f}s: {e}")
        # 返回空结果而不是抛出异常，让上层处理
        return {}, img_bgr if img_bgr is not None else _np().zeros((480, 640, 3), dtype=_np().uint8)

# -------- OTS (no Single/Range Criteria types) --------
def _attrs_to_dict(attrs) -> Dict:
    d = {}
    for item in attrs:
        # Handle different OTS SDK attribute formats
        if len(item) == 2:
            k, v = item
        elif len(item) == 3:
            k, v, _ = item  # Third element might be timestamp or metadata
        else:
            logger.warning(f"Unexpected attribute format: {item}")
            continue
            
        if k == "speciesCounts" and isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode("utf-8")
            except Exception:
                pass
        d[k] = v
    if "speciesCounts" in d and isinstance(d["speciesCounts"], str):
        try:
            d["speciesCounts"] = json.loads(d["speciesCounts"])
        except Exception:
            d["speciesCounts"] = {}
    return d

def _row_to_dict(row) -> Dict:
    out = {}
    for k, v in row.primary_key:
        out[k] = v
    out.update(_attrs_to_dict(row.attribute_columns))
    return out

def get_row_from_ots(file_key: str) -> Optional[Dict]:
    """Get a row from OTS with validation and error handling."""
    if not file_key or not isinstance(file_key, str):
        raise ValueError("file_key must be a non-empty string")
    
    client = get_ots()
    
    try:
        res = client.get_row(OTS_TABLE, [("fileKey", file_key)], None)
        # Handle different OTS SDK return formats
        if isinstance(res, tuple):
            if len(res) >= 2:
                consumed, row = res[0], res[1]
            else:
                row = res[0] if res else None
        else:
            row = getattr(res, "row", None)
        
        if row is None:
            return None
        return _row_to_dict(row)
    except Exception as e:
        logger.error(f"OTS get_row failed for {file_key}: {e}")
        raise RuntimeError(f"Failed to get row from OTS: {e}") from e

def write_row_to_ots(file_key: str, thumb_key: str, preview_key: str, species_counts: Dict[str, int]):
    """Write a row to OTS with validation and error handling."""
    if not file_key or not isinstance(file_key, str):
        raise ValueError("file_key must be a non-empty string")
    if not isinstance(species_counts, dict):
        raise ValueError("species_counts must be a dictionary")
    if not thumb_key or not isinstance(thumb_key, str):
        raise ValueError("thumb_key must be a non-empty string")
    if not preview_key or not isinstance(preview_key, str):
        raise ValueError("preview_key must be a non-empty string")
    
    client = get_ots()
    ts = _ts()
    now = int(time.time())
    ext = (file_key.rsplit(".", 1)[-1] if "." in file_key else "unknown").lower()
    
    # Validate and serialize species counts
    try:
        species_json = json.dumps(species_counts, ensure_ascii=False)
        if len(species_json) > 64000:  # OTS attribute size limit
            logger.warning(f"Species counts JSON too large for {file_key}, truncating")
            # Keep only top 100 species by count
            top_species = dict(sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:100])
            species_json = json.dumps(top_species, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize species_counts: {e}")
    
    attrs = [
        ("fileUrl", f"oss://{BUCKET}/{file_key}"),
        ("thumbUrl", f"oss://{BUCKET}/{thumb_key}"),
        ("previewUrl", f"oss://{BUCKET}/{preview_key}"),
        ("type", ext or "unknown"),
        ("createdAt", now),
        ("speciesCounts", species_json),
    ]
    
    row = ts.Row(primary_key=[("fileKey", file_key)], attribute_columns=attrs)
    cond = ts.Condition(ts.RowExistenceExpectation.IGNORE)
    
    try:
        client.put_row(OTS_TABLE, row, cond)
        logger.info(f"OTS put_row successful: table={OTS_TABLE}, key={file_key}")
    except Exception as e:
        logger.error(f"OTS put_row failed for {file_key}: {e}")
        raise RuntimeError(f"Failed to write to OTS: {e}") from e

def update_row_species(file_key: str, species_counts: Dict[str, int]):
    """Update species counts for an existing row with validation."""
    if not file_key or not isinstance(file_key, str):
        raise ValueError("file_key must be a non-empty string")
    if not isinstance(species_counts, dict):
        raise ValueError("species_counts must be a dictionary")
    
    client = get_ots()
    ts = _ts()
    
    # Validate and serialize species counts
    try:
        species_json = json.dumps(species_counts, ensure_ascii=False)
        if len(species_json) > 64000:  # OTS attribute size limit
            logger.warning(f"Species counts JSON too large for {file_key}, truncating")
            # Keep only top 100 species by count
            top_species = dict(sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:100])
            species_json = json.dumps(top_species, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize species_counts: {e}")
    
    updates = {"speciesCounts": ("PUT", species_json)}
    cond = ts.Condition(ts.RowExistenceExpectation.EXPECT_EXIST)
    
    try:
        client.update_row(OTS_TABLE, {"fileKey": file_key}, updates, cond)
        logger.info(f"OTS update_row successful: {file_key}")
    except Exception as e:
        logger.error(f"OTS update_row failed for {file_key}: {e}")
        raise RuntimeError(f"Failed to update OTS row: {e}") from e

def delete_row_ots(file_key: str):
    """Delete a row from OTS with validation and error handling."""
    if not file_key or not isinstance(file_key, str):
        raise ValueError("file_key must be a non-empty string")
    
    client = get_ots()
    ts = _ts()
    
    row = ts.Row(primary_key=[("fileKey", file_key)], attribute_columns=[])
    cond = ts.Condition(ts.RowExistenceExpectation.IGNORE)
    
    try:
        client.delete_row(OTS_TABLE, row, cond)
        logger.info(f"OTS delete_row successful: {file_key}")
    except Exception as e:
        logger.error(f"OTS delete_row failed for {file_key}: {e}")
        raise RuntimeError(f"Failed to delete row from OTS: {e}") from e

def _get_range_page(client, start_pk, end_pk, limit=None, max_versions=1):
    ts = _ts()
    try:
        result = client.get_range(
            OTS_TABLE, ts.Direction.FORWARD,
            start_pk, end_pk, None, limit, None, max_versions
        )
        # Handle different return value formats from different OTS SDK versions
        if isinstance(result, tuple):
            if len(result) == 4:
                # Some OTS SDK versions return 4 values: consumed, rows, next_pk, extra
                consumed, rows, next_pk, _ = result
            elif len(result) == 3:
                consumed, rows, next_pk = result
            elif len(result) == 2:
                rows, next_pk = result
            else:
                logger.error(f"Unexpected OTS get_range result format: {len(result)} values")
                raise RuntimeError(f"Unexpected OTS get_range result format: {len(result)} values")
        else:
            # Handle case where result is not a tuple
            logger.error(f"Unexpected OTS get_range result type: {type(result)}")
            raise RuntimeError(f"Unexpected OTS get_range result type: {type(result)}")
    except (TypeError, ValueError) as e:
        logger.error(f"OTS get_range error: {e}")
        raise RuntimeError(f"Failed to get range from OTS: {e}") from e
    return rows, next_pk

def scan_rows(limit: int = 100, prefix: Optional[str] = None) -> List[Dict]:
    """Scan rows from OTS with validation and error handling."""
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SCAN_LIMIT}")
    if prefix is not None and not isinstance(prefix, str):
        raise ValueError("prefix must be a string or None")
    
    client = get_ots()
    ts = _ts()
    start_pk = [("fileKey", ts.INF_MIN)]
    end_pk = [("fileKey", ts.INF_MAX)]
    items: List[Dict] = []
    
    try:
        while True:
            rows, next_start_pk = _get_range_page(client, start_pk, end_pk, limit=100, max_versions=1)
            if rows is None:
                logger.warning("OTS returned None for rows")
                break
            for r in rows:
                try:
                    d = _row_to_dict(r)
                    fk = d.get("fileKey")
                    if prefix and isinstance(fk, str) and not fk.startswith(prefix):
                        continue
                    items.append(d)
                    if len(items) >= limit:
                        return items
                except Exception as e:
                    logger.warning(f"Failed to process row: {e}")
                    continue
            if not next_start_pk:
                break
            start_pk = next_start_pk
        return items
    except Exception as e:
        logger.error(f"OTS scan_rows failed: {e}")
        raise RuntimeError(f"Failed to scan rows from OTS: {e}") from e

# -------- EVENT --------
def parse_oss_event(body: dict):
    events = body.get("events") or body.get("Records") or []
    if events:
        e = events[0]
        oss_info = e.get("oss") or e.get("ossInfo") or {}
        bucket = (oss_info.get("bucket") or {}).get("name") or body.get("bucketName") or BUCKET
        obj = (oss_info.get("object") or {})
        key = obj.get("key") or body.get("objectKey") or body.get("key")
    else:
        bucket = body.get("bucketName") or BUCKET
        key = body.get("objectKey") or body.get("key")
    if key:
        key = unquote(key)
    return bucket, key



@app.get("/files")
@login_required
@handle_errors
def list_files():
    try:
        limit = int(request.args.get("limit", str(DEFAULT_SCAN_LIMIT)))
    except ValueError:
        raise ValueError("limit must be a valid integer")
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SCAN_LIMIT}")
    
    prefix = request.args.get("prefix")
    if prefix is not None and not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    
    items = scan_rows(limit=limit, prefix=prefix)
    return jsonify({
        "count": len(items), 
        "items": items,
        "limit": limit,
        "prefix": prefix
    })

@app.post("/search")
@login_required
@handle_errors
def search_min_counts():
    payload = request.get_json(force=True, silent=True) or {}
    
    must = payload.get("min") or {}
    if not isinstance(must, dict):
        raise ValueError("min must be a dictionary")
    
    # Validate min_counts values
    for species, min_count in must.items():
        if not isinstance(species, str) or not species.strip():
            raise ValueError("Species names must be non-empty strings")
        try:
            min_count_int = int(min_count)
            if min_count_int < 0:
                raise ValueError("Minimum counts must be non-negative integers")
        except (ValueError, TypeError):
            raise ValueError("Minimum counts must be valid non-negative integers")
    
    try:
        limit = int(payload.get("limit", DEFAULT_SCAN_LIMIT))
    except (ValueError, TypeError):
        raise ValueError("limit must be a valid integer")
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SCAN_LIMIT}")
    
    rows = scan_rows(limit=1000)
    
    def ok(row):
        sc = row.get("speciesCounts") or {}
        try:
            return all(int(sc.get(k, 0)) >= int(v) for k, v in must.items())
        except (ValueError, TypeError):
            logger.warning(f"Failed to validate species counts for {row.get('fileKey', 'unknown')}")
            return False
    
    hits = [r for r in rows if ok(r)]
    return jsonify({
        "count": len(hits), 
        "items": hits[:limit],
        "searched": len(rows),
        "criteria": must
    })

@app.get("/by-species")
@login_required
@handle_errors
def by_species():
    name = request.args.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        raise ValueError("name parameter is required and must be a non-empty string")
    
    try:
        mincnt = int(request.args.get("min", "1"))
    except ValueError:
        raise ValueError("min parameter must be a valid integer")
    
    if mincnt < 0:
        raise ValueError("min parameter must be non-negative")
    
    try:
        limit = int(request.args.get("limit", "1000"))
    except ValueError:
        raise ValueError("limit parameter must be a valid integer")
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SCAN_LIMIT}")
    
    rows = scan_rows(limit=limit)
    hits = []
    
    for r in rows:
        try:
            species_counts = r.get("speciesCounts") or {}
            count = int(species_counts.get(name, 0))
            if count >= mincnt:
                hits.append(r)
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse species count for {r.get('fileKey', 'unknown')}")
            continue
    
    return jsonify({
        "count": len(hits), 
        "items": hits,
        "species": name,
        "minCount": mincnt,
        "searched": len(rows)
    })

@app.post("/reverse-thumb")
@login_required
@handle_errors
def reverse_thumb():
    payload = request.get_json(force=True, silent=True) or {}
    
    key = payload.get("thumbKey")
    url = payload.get("thumbUrl")
    
    if not key and not url:
        raise ValueError("Either thumbKey or thumbUrl is required")
    
    thumb_key = None
    
    if key:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("thumbKey must be a non-empty string")
        thumb_key = key.strip()
    elif url:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("thumbUrl must be a non-empty string")
        
        try:
            if url.startswith("oss://"):
                p = urlparse(url)
                thumb_key = p.path.lstrip("/")
            else:
                u = urlparse(url)
                parts = u.path.split("/", 2)
                if len(parts) >= 3:
                    thumb_key = parts[2]
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")
    
    if not thumb_key:
        raise ValueError("Could not extract valid thumb key from provided input")
    
    try:
        limit = int(payload.get("limit", "1000"))
    except (ValueError, TypeError):
        limit = 1000
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        limit = min(1000, MAX_SCAN_LIMIT)
    
    rows = scan_rows(limit=limit)
    full = f"oss://{BUCKET}/{thumb_key}"
    
    for r in rows:
        tu = r.get("thumbUrl")
        if tu == full or tu == thumb_key:
            return jsonify({
                "item": r,
                "matchedBy": "thumbUrl",
                "searchKey": thumb_key
            })
    
    return jsonify({
        "item": None,
        "searchKey": thumb_key,
        "searched": len(rows)
    })

@app.post("/intersect")
@login_required
@handle_errors
def intersect():
    payload = request.get_json(force=True, silent=True) or {}
    
    key = payload.get("objectKey")
    if not key or not isinstance(key, str) or not key.strip():
        raise ValueError("objectKey is required and must be a non-empty string")
    
    key = key.strip()
    
    try:
        limit = int(payload.get("limit", "1000"))
    except (ValueError, TypeError):
        limit = 1000
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        limit = min(1000, MAX_SCAN_LIMIT)
    
    base = get_row_from_ots(key)
    if not base:
        return jsonify({
            "queryLabels": {}, 
            "count": 0, 
            "items": [],
            "error": "Object not found"
        }), 404

    base_labels = set((base.get("speciesCounts") or {}).keys())
    if not base_labels:
        return jsonify({
            "queryLabels": list(base_labels), 
            "count": 0, 
            "items": []
        })

    rows = scan_rows(limit=limit)
    
    hits = []
    for r in rows:
        if r.get("fileKey") == key:
            continue
        
        other_labels = set((r.get("speciesCounts") or {}).keys())
        if base_labels.intersection(other_labels):
            hits.append(r)
            
    return jsonify({
        "queryLabels": list(base_labels),
        "count": len(hits),
        "items": hits,
        "searched": len(rows),
            "objectKey": key,
            "found": False
        })
    
    sc0 = base.get("speciesCounts") or {}
    
    try:
        labels = {k for k, v in sc0.items() if int(v) > 0}
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse species counts for base object {key}: {e}")
        labels = set()
    
    if not labels:
        return jsonify({
            "queryLabels": sc0, 
            "count": 0, 
            "items": [],
            "objectKey": key,
            "found": True,
            "message": "No species found in base object"
        })
    
    rows = scan_rows(limit=limit)
    hits = []
    
    for r in rows:
        if r.get("fileKey") == key:
            continue
        
        try:
            sc = r.get("speciesCounts") or {}
            row_labels = {k for k, v in sc.items() if int(v) > 0}
            if labels & row_labels:
                # Add intersection info
                intersection = labels & row_labels
                r_copy = r.copy()
                r_copy["commonSpecies"] = list(intersection)
                hits.append(r_copy)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse species counts for {r.get('fileKey', 'unknown')}: {e}")
            continue
    
    return jsonify({
        "queryLabels": sc0, 
        "count": len(hits), 
        "items": hits,
        "objectKey": key,
        "found": True,
        "searched": len(rows),
        "baseSpecies": list(labels)
    })

@app.post("/tags:update")
@login_required
@handle_errors
def tags_update():
    payload = request.get_json(force=True, silent=True) or {}
    
    keys = payload.get("fileKeys") or []
    adds = payload.get("add") or {}
    removes = set(payload.get("remove") or [])
    
    if not keys:
        raise ValueError("fileKeys is required and must be a non-empty list")
    
    if not isinstance(keys, list):
        raise ValueError("fileKeys must be a list")
    
    if not isinstance(adds, dict):
        raise ValueError("add must be a dictionary")
    
    if not isinstance(removes, (list, set)):
        raise ValueError("remove must be a list or set")
    
    # Validate add values
    for species, count in adds.items():
        if not isinstance(species, str) or not species.strip():
            raise ValueError("Species names in add must be non-empty strings")
        try:
            count_int = int(count)
            if count_int < 0:
                raise ValueError("Species counts in add must be non-negative integers")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid count for species '{species}': must be a non-negative integer")
    
    # Validate remove values
    for species in removes:
        if not isinstance(species, str) or not species.strip():
            raise ValueError("Species names in remove must be non-empty strings")
    
    updated = []
    not_found = []
    
    for fk in keys:
        if not isinstance(fk, str) or not fk.strip():
            logger.warning(f"Skipping invalid fileKey: {fk}")
            continue
            
        fk = fk.strip()
        
        try:
            row = get_row_from_ots(fk)
            if not row:
                not_found.append(fk)
                continue
            
            sc = dict(row.get("speciesCounts") or {})
            
            # Apply additions
            for k, v in adds.items():
                sc[k] = int(v)
            
            # Apply removals
            for k in removes:
                sc.pop(k, None)
            
            update_row_species(fk, sc)
            updated.append({"fileKey": fk, "speciesCounts": sc})
            
        except Exception as e:
            logger.error(f"Failed to update {fk}: {e}")
            continue
    
    response = {
        "updated": updated,
        "updatedCount": len(updated),
        "requestedCount": len(keys),
        "updatedAt": int(time.time())
    }
    
    if not_found:
        response["notFound"] = not_found
        response["notFoundCount"] = len(not_found)
    
    return jsonify(response)

@app.delete("/files")
@login_required
@handle_errors
def delete_file():
    payload = request.get_json(force=True, silent=True) or {}
    
    fk = payload.get("fileKey")
    if not fk or not isinstance(fk, str) or not fk.strip():
        raise ValueError("fileKey is required and must be a non-empty string")
    
    fk = fk.strip()
    del_objs = bool(payload.get("deleteObjects", True))
    
    # Check if row exists first
    existing = get_row_from_ots(fk)
    if not existing:
        raise FileNotFoundError(f"File not found: {fk}")
    
    deleted_objects = []
    failed_deletions = []
    
    if del_objs:
        b = get_oss_bucket()
        
        # Calculate related object keys
        rel = fk[len(UPLOADS_PREFIX):] if fk.startswith(UPLOADS_PREFIX) else fk
        preview_key = f"{PREVIEW_PREFIX}{rel}"
        thumb_key = f"{THUMBS_PREFIX}{rel}"
        
        # Delete all related objects
        for k in [fk, preview_key, thumb_key]:
            try:
                b.delete_object(k)
                deleted_objects.append(k)
                logger.info(f"delete oss ok: {k}")
            except Exception as e:
                logger.warning(f"delete oss fail {k}: {e}")
                failed_deletions.append({"object": k, "error": str(e)})
    
    # Delete from OTS
    try:
        delete_row_ots(fk)
        logger.info(f"Deleted OTS record: {fk}")
    except Exception as e:
        logger.error(f"Failed to delete OTS record {fk}: {e}")
        failed_deletions.append({"object": f"OTS:{fk}", "error": str(e)})
    
    response = {
        "status": "deleted" if not failed_deletions else "partially_deleted",
        "fileKey": fk,
        "deletedObjects": del_objs,
        "deletedAt": int(time.time())
    }
    
    if del_objs:
        response["ossObjects"] = {
            "deleted": deleted_objects,
            "deletedCount": len(deleted_objects)
        }
    
    if failed_deletions:
        response["failedDeletions"] = failed_deletions
        response["failedCount"] = len(failed_deletions)
        response["warning"] = "Some objects could not be deleted"
    
    return jsonify(response)

# Global error handler for unhandled exceptions
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500

# Add request logging middleware
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")

@app.after_request
def log_response_info(response):
    logger.info(f"Response: {response.status_code}")
    return response

if __name__ == "__main__":
    # Validate critical environment variables on startup
    missing_vars = []
    if not BUCKET:
        missing_vars.append("BUCKET")
    if not OTS_INSTANCE:
        missing_vars.append("OTS_INSTANCE")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    logger.info("Starting Flask application...")
    logger.info(f"Configuration: BUCKET={BUCKET}, OTS_INSTANCE={OTS_INSTANCE}, REGION={REGION}")
    logger.info(f"Model path: {MODEL_PATH}, Confidence: {CONFIDENCE}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("FC_SERVER_PORT", "9000")), debug=False, threaded=True)

# ...existing code...

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory user store for simplicity. In a real application, use a database.
# In-memory user store is replaced by Tablestore

class User(UserMixin):
    def __init__(self, id, email, password_hash):
        self.id = id
        self.email = email
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        try:
            client = get_ots()
            _, row, _ = client.get_row(
                'users',
                [('username', user_id)],
                max_versions=1
            )
            if row:
                user_data = {col.name: col.value for col in row.attribute_columns}
                return User(
                    id=user_id,
                    email=user_data.get('email'),
                    password_hash=user_data.get('password_hash')
                )
        except Exception as e:
            logger.error(f"Error getting user from OTS: {e}")
        return None

    @staticmethod
    def create(username, email, password_hash):
        try:
            client = get_ots()
            ts = _ts()
            row = ts.Row(
                primary_key=[('username', username)],
                attribute_columns=[('email', email), ('password_hash', password_hash)]
            )
            client.put_row('users', row)
            return True
        except Exception as e:
            logger.error(f"Error creating user in OTS: {e}")
            return False

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/')
@login_required
def index():
    files = scan_rows(limit=100)
    return render_template('index.html', files=files)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.get(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.get(username):
            flash('Username already exists', 'warning')
        else:
            password_hash = generate_password_hash(password)
            if User.create(username, email, password_hash):
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Registration failed.', 'danger')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    if file:
        # In a real scenario, you would save the file and process it.
        # For this example, we just flash a success message.
        flash(f'File {file.filename} uploaded successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/search_ui')
@login_required
def search_by_species_ui():
    name = request.args.get("name")
    if not name:
        flash('Species name is required', 'danger')
        return redirect(url_for('index'))
    
    # This is a simplified search. In a real app, you'd render a search results page.
    rows = scan_rows(limit=100)
    hits = []
    for r in rows:
        species_counts = r.get("speciesCounts") or {}
        if name in species_counts and int(species_counts.get(name, 0)) > 0:
            hits.append(r)
            
    return render_template('index.html', files=hits)

@app.route('/delete_file_ui', methods=['POST'])
@login_required
def delete_file_ui():
    file_key = request.form.get('fileKey')
    if file_key:
        try:
            delete_row_ots(file_key)
            b = get_oss_bucket()
            rel = file_key[len(UPLOADS_PREFIX):] if file_key.startswith(UPLOADS_PREFIX) else file_key
            preview_key = f"{PREVIEW_PREFIX}{rel}"
            thumb_key = f"{THUMBS_PREFIX}{rel}"
            for k in [file_key, preview_key, thumb_key]:
                try:
                    b.delete_object(k)
                except Exception:
                    pass
            flash(f'File {file_key} deleted.', 'success')
        except Exception as e:
            flash(f'Error deleting file: {e}', 'danger')
    else:
        flash('File key is missing.', 'danger')
    return redirect(url_for('index'))

@app.route('/health')
@handle_errors
def health():
    return jsonify({"ok": True, "ts": int(time.time())})

@app.post("/initialize")
@handle_errors
def initialize():
    _ensure_model()
    get_oss_bucket()
    get_ots()
    return jsonify({"status": "inited"}), 200

@app.post("/invoke")
@handle_errors
def invoke():
    start_time = time.time()
    request_id = request.headers.get('X-Request-ID', 'unknown')
    
    logger.info(f"FC Invoke Start RequestId: {request_id} (max processing time: {MAX_PROCESSING_TIME}s)")
    
    def check_timeout(stage_name):
        elapsed = time.time() - start_time
        if elapsed > MAX_PROCESSING_TIME:
            raise TimeoutError(f"Processing timeout after {elapsed:.1f}s in stage: {stage_name}")
        return elapsed
    
    try:
        body = request.get_json(force=True, silent=True) or {}
        bucket, key = parse_oss_event(body)
        logger.info(f"event key={key}, bucket={bucket}")

        if not key or not key.startswith(UPLOADS_PREFIX):
            return jsonify({"error": f"objectKey must start with {UPLOADS_PREFIX}"}), 400
        
        # 检查文件扩展名
        file_ext = os.path.splitext(key)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            logger.warning(f"Unsupported file type: {file_ext}")
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400

        # 从OSS下载文件
        check_timeout("validation")
        download_start = time.time()
        frame = _read_image_from_oss(key)
        download_time = time.time() - download_start
        if LOG_PERFORMANCE:
            logger.info(f"File download completed in {download_time:.3f}s")
        
        # 进行推理
        check_timeout("download")
        infer_start = time.time()
        species_counts, annotated = _infer(frame, CONFIDENCE)
        infer_time = time.time() - infer_start
        logger.info(f"Inference completed in {infer_time:.3f}s")

        # 生成缩略图和预览图
        check_timeout("inference")
        process_start = time.time()
        rel = key[len(UPLOADS_PREFIX):]
        preview_key = f"{PREVIEW_PREFIX}{rel}"
        thumb_key = f"{THUMBS_PREFIX}{rel}"

        _put_to_oss(preview_key, _jpeg_bytes(annotated, 90))
        thumb = _resize_long(annotated, THUMB_MAX_SIZE)
        _put_to_oss(thumb_key, _jpeg_bytes(thumb, 85))
        process_time = time.time() - process_start
        if LOG_PERFORMANCE:
            logger.info(f"Image processing and upload completed in {process_time:.3f}s")

        # 写入OTS
        check_timeout("processing")
        ots_start = time.time()
        write_row_to_ots(key, thumb_key, preview_key, species_counts)
        ots_time = time.time() - ots_start
        if LOG_PERFORMANCE:
            logger.info(f"OTS write completed in {ots_time:.3f}s")
        
        processing_time = time.time() - start_time
        logger.info(f"FC Invoke End RequestId: {request_id} - Total time: {processing_time:.3f}s")

        response = {
            "status": "ok", 
            "fileKey": key, 
            "speciesCounts": species_counts,
            "processingTime": round(processing_time, 3),
            "requestId": request_id
        }
        
        if LOG_PERFORMANCE:
            response["performance"] = {
                "downloadTime": round(download_time, 3),
                "inferenceTime": round(infer_time, 3),
                "processingTime": round(process_time, 3),
                "databaseTime": round(ots_time, 3)
            }

        return jsonify(response), 200
        
    except TimeoutError as e:
        processing_time = time.time() - start_time
        logger.error(f"Processing timeout after {processing_time:.3f}s: {e}")
        return jsonify({
            'error': 'Processing timeout',
            'errorType': 'timeout',
            'processingTime': round(processing_time, 3),
            'requestId': request_id
        }), 408
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Invoke processing failed after {processing_time:.3f}s: {e}")
        return jsonify({
            'error': 'Internal processing error',
            'errorType': 'processing_error',
            'processingTime': round(processing_time, 3),
            'requestId': request_id
        }), 500

@app.get("/files")
@handle_errors
def list_files():
    try:
        limit = int(request.args.get("limit", str(DEFAULT_SCAN_LIMIT)))
    except ValueError:
        raise ValueError("limit must be a valid integer")
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SCAN_LIMIT}")
    
    prefix = request.args.get("prefix")
    if prefix is not None and not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    
    items = scan_rows(limit=limit, prefix=prefix)
    return jsonify({
        "count": len(items), 
        "items": items,
        "limit": limit,
        "prefix": prefix
    })

@app.post("/search")
@handle_errors
def search_min_counts():
    payload = request.get_json(force=True, silent=True) or {}
    
    must = payload.get("min") or {}
    if not isinstance(must, dict):
        raise ValueError("min must be a dictionary")
    
    # Validate min_counts values
    for species, min_count in must.items():
        if not isinstance(species, str) or not species.strip():
            raise ValueError("Species names must be non-empty strings")
        try:
            min_count_int = int(min_count)
            if min_count_int < 0:
                raise ValueError("Minimum counts must be non-negative integers")
        except (ValueError, TypeError):
            raise ValueError("Minimum counts must be valid non-negative integers")
    
    try:
        limit = int(payload.get("limit", DEFAULT_SCAN_LIMIT))
    except (ValueError, TypeError):
        raise ValueError("limit must be a valid integer")
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SCAN_LIMIT}")
    
    rows = scan_rows(limit=1000)
    
    def ok(row):
        sc = row.get("speciesCounts") or {}
        try:
            return all(int(sc.get(k, 0)) >= int(v) for k, v in must.items())
        except (ValueError, TypeError):
            logger.warning(f"Failed to validate species counts for {row.get('fileKey', 'unknown')}")
            return False
    
    hits = [r for r in rows if ok(r)]
    return jsonify({
        "count": len(hits), 
        "items": hits[:limit],
        "searched": len(rows),
        "criteria": must
    })

@app.get("/by-species")
@handle_errors
def by_species():
    name = request.args.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        raise ValueError("name parameter is required and must be a non-empty string")
    
    try:
        mincnt = int(request.args.get("min", "1"))
    except ValueError:
        raise ValueError("min parameter must be a valid integer")
    
    if mincnt < 0:
        raise ValueError("min parameter must be non-negative")
    
    try:
        limit = int(request.args.get("limit", "1000"))
    except ValueError:
        raise ValueError("limit parameter must be a valid integer")
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SCAN_LIMIT}")
    
    rows = scan_rows(limit=limit)
    hits = []
    
    for r in rows:
        try:
            species_counts = r.get("speciesCounts") or {}
            count = int(species_counts.get(name, 0))
            if count >= mincnt:
                hits.append(r)
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse species count for {r.get('fileKey', 'unknown')}")
            continue
    
    return jsonify({
        "count": len(hits), 
        "items": hits,
        "species": name,
        "minCount": mincnt,
        "searched": len(rows)
    })

@app.post("/reverse-thumb")
@handle_errors
def reverse_thumb():
    payload = request.get_json(force=True, silent=True) or {}
    
    key = payload.get("thumbKey")
    url = payload.get("thumbUrl")
    
    if not key and not url:
        raise ValueError("Either thumbKey or thumbUrl is required")
    
    thumb_key = None
    
    if key:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("thumbKey must be a non-empty string")
        thumb_key = key.strip()
    elif url:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("thumbUrl must be a non-empty string")
        
        try:
            if url.startswith("oss://"):
                p = urlparse(url)
                thumb_key = p.path.lstrip("/")
            else:
                u = urlparse(url)
                parts = u.path.split("/", 2)
                if len(parts) >= 3:
                    thumb_key = parts[2]
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")
    
    if not thumb_key:
        raise ValueError("Could not extract valid thumb key from provided input")
    
    try:
        limit = int(payload.get("limit", "1000"))
    except (ValueError, TypeError):
        limit = 1000
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        limit = min(1000, MAX_SCAN_LIMIT)
    
    rows = scan_rows(limit=limit)
    full = f"oss://{BUCKET}/{thumb_key}"
    
    for r in rows:
        tu = r.get("thumbUrl")
        if tu == full or tu == thumb_key:
            return jsonify({
                "item": r,
                "matchedBy": "thumbUrl",
                "searchKey": thumb_key
            })
    
    return jsonify({
        "item": None,
        "searchKey": thumb_key,
        "searched": len(rows)
    })

@app.post("/intersect")
@handle_errors
def intersect():
    payload = request.get_json(force=True, silent=True) or {}
    
    key = payload.get("objectKey")
    if not key or not isinstance(key, str) or not key.strip():
        raise ValueError("objectKey is required and must be a non-empty string")
    
    key = key.strip()
    
    try:
        limit = int(payload.get("limit", "1000"))
    except (ValueError, TypeError):
        limit = 1000
    
    if limit <= 0 or limit > MAX_SCAN_LIMIT:
        limit = min(1000, MAX_SCAN_LIMIT)
    
    base = get_row_from_ots(key)
    if not base:
        return jsonify({
            "queryLabels": {}, 
            "count": 0, 
            "items": [],})