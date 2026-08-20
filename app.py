from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from PIL import Image, ImageOps, ImageEnhance
import imagehash, requests, io, os, time, threading, json, re, hashlib
# Deep visual embedding (DINOv2)

app = Flask(__name__)
CORS(app)

SHOP_BASE_URL = os.environ.get("SHOP_BASE_URL", "https://freeorder1.cafe24.com").rstrip("/")
DATA_DIR = os.environ.get("DATA_DIR", "/var/data")
INDEX_FILE = os.path.join(DATA_DIR, "product_index_ai.json")

MAX_CATEGORY_PAGES = int(os.environ.get("MAX_CATEGORY_PAGES", "200"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "15"))
RESULT_LIMIT = int(os.environ.get("RESULT_LIMIT", "12"))
AUTO_REFRESH_SECONDS = int(os.environ.get("AUTO_REFRESH_SECONDS", "300"))
MAX_PRODUCT_IMAGES = int(os.environ.get("MAX_PRODUCT_IMAGES", "8"))

# DINOv2-small is much more robust than pHash for the same clothing item
# photographed with different poses/backgrounds.
EMBEDDING_VERSION = "lightweight-design-hash-v3"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0 Safari/537.36"
)

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"
})

product_index = []
product_index_by_url = {}
index_lock = threading.Lock()
model_lock = threading.Lock()
indexing = False
reindex_progress = {
    "running": False,
    "force": False,
    "started_at": None,
    "finished_at": None,
    "current": 0,
    "total": 0,
    "added": 0,
    "reused": 0,
    "failed": 0,
    "message": "대기 중",
    "error": None,
}
last_indexed_at = None
last_refresh_check_at = 0

# =========================================================
# Lightweight design fingerprint
# =========================================================

def prepare_design_image(image):
    """
    색상 차이를 없애고 의류의 형태/명암/봉제선 위주로 비교한다.
    PyTorch 없이 Pillow의 해시만 사용하므로 Render 저사양에서도 가볍게 동작한다.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")

    # 너무 큰 이미지는 먼저 축소
    max_side = 1200
    if max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS
        )

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    return gray


def center_crop(image, ratio=0.82):
    """배경 영향을 줄이기 위해 중앙 의류 영역을 넓게 잘라낸다."""
    w, h = image.size
    cw, ch = int(w * ratio), int(h * ratio)
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return image.crop((left, top, left + cw, top + ch))


def edge_image(gray):
    """
    Pillow FIND_EDGES를 사용해 색상이 아닌 넥라인/소매/밑단/스티치 등의
    구조적 경계선을 강조한다.
    """
    from PIL import ImageFilter, ImageEnhance
    e = gray.filter(ImageFilter.FIND_EDGES)
    e = ImageOps.autocontrast(e)
    e = ImageEnhance.Contrast(e).enhance(1.25)
    return e


def calculate_fingerprint(image):
    """
    여러 종류의 색상 비의존 해시를 함께 저장한다.
    전체 사진 + 중앙 크롭 + 에지 이미지를 조합해서
    누끼/착용컷/배경 차이에 대한 내성을 높인다.
    """
    gray = prepare_design_image(image)
    crop = center_crop(gray)
    edge_full = edge_image(gray)
    edge_crop = edge_image(crop)

    return {
        "phash_full": str(imagehash.phash(gray, hash_size=16)),
        "phash_crop": str(imagehash.phash(crop, hash_size=16)),
        "dhash_crop": str(imagehash.dhash(crop, hash_size=16)),
        "whash_crop": str(imagehash.whash(crop, hash_size=16)),
        "edge_full": str(imagehash.phash(edge_full, hash_size=16)),
        "edge_crop": str(imagehash.phash(edge_crop, hash_size=16)),
    }


def calculate_hash(image):
    # 기존 호환용
    return calculate_fingerprint(image)["phash_full"]


def hash_distance(hash_a, hash_b):
    return int(imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b))


def fingerprint_score(query_fp, product_fp):
    """
    0~1 점수. 색상은 사용하지 않고 구조 해시들만 비교한다.
    중앙 크롭/에지 비교 비중을 크게 둔다.
    """
    keys_weights = [
        ("phash_full", 0.12),
        ("phash_crop", 0.24),
        ("dhash_crop", 0.14),
        ("whash_crop", 0.12),
        ("edge_full", 0.14),
        ("edge_crop", 0.24),
    ]

    total = 0.0
    weight_sum = 0.0
    for key, weight in keys_weights:
        a = query_fp.get(key)
        b = product_fp.get(key)
        if not a or not b:
            continue

        # hash_size=16 => 256 bits
        dist = hash_distance(a, b)
        sim = max(0.0, 1.0 - (dist / 256.0))
        total += sim * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0
    return total / weight_sum


# =========================================================
# Indexing
# =========================================================
def build_index(force=False):
    global product_index, indexing, last_indexed_at, reindex_progress

    with index_lock:
        if indexing:
            return {"success": False, "message": "이미 인덱싱 중입니다."}
        indexing = True

    started = time.time()
    reindex_progress.update({
        "running": True,
        "force": bool(force),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "current": 0,
        "total": 0,
        "added": 0,
        "reused": 0,
        "failed": 0,
        "message": "상품 URL 수집 중",
        "error": None,
    })

    try:
        existing_map = {}
        if not force:
            existing_map = {
                item.get("product_url"): item
                for item in product_index
                if item.get("product_url")
            }

        product_urls = discover_product_urls()
        reindex_progress["total"] = len(product_urls)
        reindex_progress["message"] = f"상품 분석 시작: {len(product_urls)}개"
        print(f"[INDEX] discovered={len(product_urls)}", flush=True)

        new_index = []
        added = reused = failed = 0

        for idx, product_url in enumerate(product_urls, start=1):
            reindex_progress["current"] = idx
            reindex_progress["message"] = f"상품 분석 중 {idx}/{len(product_urls)}"
            existing = existing_map.get(product_url)
            if not force and usable_existing_item(existing):
                new_index.append(existing)
                reused += 1
                reindex_progress["reused"] = reused
                continue

            try:
                info = extract_product_info(product_url)
                final_url = info["product_url"]

                existing = existing_map.get(final_url)
                if not force and usable_existing_item(existing):
                    new_index.append(existing)
                    reused += 1
                    reindex_progress["reused"] = reused
                    continue

                if not info["image_urls"]:
                    failed += 1
                    reindex_progress["failed"] = failed
                    continue

                image_entries = []
                for image_url in info["image_urls"]:
                    try:
                        image = download_image(image_url)

                        # Skip tiny theme/UI images that slipped through HTML filtering.
                        if image.width < 180 or image.height < 180:
                            continue

                        fingerprint = calculate_fingerprint(image)
                        image_entries.append({
                            "image_url": str(image_url),
                            "phash": str(calculate_hash(image)),
                            "fingerprint": fingerprint,
                        })
                    except Exception as image_error:
                        print("[INDEX] image failed:", image_url, repr(image_error), flush=True)

                if not image_entries:
                    failed += 1
                    reindex_progress["failed"] = failed
                    continue

                new_index.append({
                    "title": str(info["title"]),
                    "product_url": str(final_url),
                    "image_url": str(image_entries[0]["image_url"]),
                    "price": str(info.get("price", "")),
                    "embedding_version": EMBEDDING_VERSION,
                    "images": image_entries,
                })

                added += 1
                reindex_progress["added"] = added

                if idx % 10 == 0 or idx == len(product_urls):
                    print(
                        f"[INDEX] {idx}/{len(product_urls)} "
                        f"added={added} reused={reused} failed={failed}",
                        flush=True
                    )

            except Exception as e:
                failed += 1
                reindex_progress["failed"] = failed
                print("[INDEX] product failed:", product_url, repr(e), flush=True)

        deduped = {}
        for item in new_index:
            url = item.get("product_url")
            if url:
                deduped[url] = item

        product_index = list(deduped.values())
        last_indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")

        rebuild_url_map()
        save_index()

        elapsed = round(time.time() - started, 2)
        return {
            "success": True,
            "indexed": int(len(product_index)),
            "added": int(added),
            "reused": int(reused),
            "failed": int(failed),
            "elapsed_seconds": float(elapsed),
            "last_indexed_at": last_indexed_at,
            "model": MODEL_NAME,
        }

    finally:
        indexing = False


def refresh_index_if_needed():
    global last_refresh_check_at

    now = time.time()
    if AUTO_REFRESH_SECONDS <= 0:
        return

    if now - last_refresh_check_at < AUTO_REFRESH_SECONDS:
        return

    last_refresh_check_at = now

    if indexing:
        return

    try:
        result = build_index(force=False)
        if result.get("success"):
            print(
                f"[AUTO REFRESH] indexed={result.get('indexed')} "
                f"added={result.get('added')} reused={result.get('reused')} "
                f"failed={result.get('failed')}",
                flush=True
            )
    except Exception as e:
        print("[AUTO REFRESH ERROR]", repr(e), flush=True)


def ensure_index():
    if product_index:
        refresh_index_if_needed()
        return

    if load_index():
        refresh_index_if_needed()
        return

    result = build_index(force=False)

    if not result.get("success") and not product_index:
        raise RuntimeError(result.get("message", "상품 인덱스를 만들 수 없습니다."))


# =========================================================
# Routes
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Cafe24 AI Clothing Image Search API is running.",
        "shop": SHOP_BASE_URL,
        "indexed_products": int(len(product_index)),
        "indexing": bool(indexing),
        "last_indexed_at": last_indexed_at,
        "model": MODEL_NAME,
        "embedding_version": EMBEDDING_VERSION,
        "index_file": INDEX_FILE,
        "index_file_exists": os.path.exists(INDEX_FILE),
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "success": True,
        "shop": SHOP_BASE_URL,
        "indexed_products": int(len(product_index)),
        "indexing": bool(indexing),
        "last_indexed_at": last_indexed_at,
        "result_limit": int(RESULT_LIMIT),
        "auto_refresh_seconds": int(AUTO_REFRESH_SECONDS),
        "max_product_images": int(MAX_PRODUCT_IMAGES),
        "search_mode": "lightweight_color_invariant_design",
        "model": MODEL_NAME,
        "index_file": INDEX_FILE,
        "index_file_exists": os.path.exists(INDEX_FILE),
    })




def run_reindex_background(force=False):
    global reindex_progress
    try:
        result = build_index(force=force)
        if not result.get("success"):
            reindex_progress.update({
                "running": False,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": result.get("message", "재색인 실패"),
                "error": result.get("message", "재색인 실패"),
            })
    except Exception as e:
        reindex_progress.update({
            "running": False,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": "오류 발생",
            "error": str(e),
        })
        print("[BACKGROUND REINDEX ERROR]", repr(e), flush=True)


@app.route("/reindex/start", methods=["POST"])
def reindex_start():
    """재색인을 백그라운드에서 시작하고 즉시 응답한다."""
    force = request.args.get("force", "0") == "1"

    if indexing or reindex_progress.get("running"):
        return jsonify({
            "success": False,
            "message": "이미 재색인 작업이 진행 중입니다.",
            "progress": reindex_progress,
        }), 409

    worker = threading.Thread(
        target=run_reindex_background,
        kwargs={"force": force},
        daemon=True,
    )
    worker.start()

    return jsonify({
        "success": True,
        "message": "재색인 작업을 시작했습니다.",
        "force": force,
    })


@app.route("/reindex/progress", methods=["GET"])
def reindex_progress_api():
    return jsonify({
        "success": True,
        "progress": reindex_progress,
        "indexed_products": len(product_index),
        "last_indexed_at": last_indexed_at,
    })


@app.route("/admin/reindex", methods=["GET"])
def admin_reindex_page():
    """브라우저에서 상품 인덱스를 쉽게 갱신하는 관리 페이지."""
    return """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>상품 이미지 검색 DB 관리</title>
<style>
body{font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:40px 20px;color:#111}
.box{max-width:680px;margin:auto;background:#fff;border-radius:18px;padding:32px;box-shadow:0 8px 30px rgba(0,0,0,.08)}
h1{font-size:24px;margin:0 0 12px}.desc{color:#666;line-height:1.6;margin-bottom:24px}
button{width:100%;border:0;border-radius:12px;padding:16px;font-size:16px;font-weight:700;cursor:pointer;margin-top:10px}
.refresh{background:#111;color:#fff}.force{background:#eee;color:#111}
button:disabled{opacity:.45;cursor:not-allowed}
#result{white-space:pre-wrap;background:#f7f7f7;padding:16px;border-radius:12px;margin-top:20px;min-height:80px;line-height:1.55}
.barwrap{height:12px;background:#ececec;border-radius:999px;overflow:hidden;margin-top:18px}
.bar{height:100%;width:0%;background:#111;transition:width .3s}
.status{font-size:14px;font-weight:700;margin-top:12px}
.small{font-size:13px;color:#888;margin-top:14px;line-height:1.5}
</style>
</head>
<body>
<div class="box">
<h1>상품 이미지 검색 DB 관리</h1>
<div class="desc">
새 상품을 등록한 뒤에는 <b>신규 상품 반영</b>을 누르세요.<br>
검색 방식이 바뀌었거나 전체 DB를 다시 만들 때만 <b>전체 강제 재생성</b>을 사용하세요.
</div>
<button id="refreshBtn" class="refresh" onclick="startJob(false)">신규 상품 반영</button>
<button id="forceBtn" class="force" onclick="startJob(true)">전체 강제 재생성</button>
<div class="barwrap"><div id="bar" class="bar"></div></div>
<div id="status" class="status">상태 확인 중...</div>
<div id="result">대기 중</div>
<div class="small">
재색인은 서버 백그라운드에서 실행됩니다. 브라우저 요청이 오래 대기하지 않기 때문에 Render 요청 타임아웃 영향을 덜 받습니다.
페이지를 닫았다가 다시 열어도 서버가 살아있는 동안 진행 상태를 다시 확인할 수 있습니다.
</div>
</div>
<script>
let timer=null;

function setButtons(disabled){
  document.getElementById('refreshBtn').disabled=disabled;
  document.getElementById('forceBtn').disabled=disabled;
}

async function startJob(force){
  if(force && !confirm('전체 상품 이미지 특징값을 다시 생성합니다. 계속할까요?')) return;

  setButtons(true);
  document.getElementById('status').textContent='작업 시작 요청 중...';

  try{
    const r=await fetch('/reindex/start' + (force ? '?force=1' : ''), {method:'POST'});
    const txt=await r.text();
    let data;
    try { data=JSON.parse(txt); }
    catch(e){ throw new Error('서버가 JSON 대신 다른 응답을 보냈습니다: ' + txt.slice(0,160)); }

    if(!r.ok && r.status !== 409){
      throw new Error(data.message || '재색인 시작 실패');
    }

    document.getElementById('result').textContent=JSON.stringify(data,null,2);
    poll();
  }catch(e){
    document.getElementById('status').textContent='오류';
    document.getElementById('result').textContent='오류: '+e.message;
    setButtons(false);
  }
}

async function poll(){
  try{
    const r=await fetch('/reindex/progress', {cache:'no-store'});
    const txt=await r.text();
    let data;
    try { data=JSON.parse(txt); }
    catch(e){ throw new Error('진행상태 응답 오류: ' + txt.slice(0,160)); }

    const p=data.progress || {};
    const total=Number(p.total || 0);
    const current=Number(p.current || 0);
    const pct=total ? Math.min(100, Math.round(current/total*100)) : 0;

    document.getElementById('bar').style.width=pct+'%';
    document.getElementById('status').textContent=
      (p.running ? '진행 중' : (p.message || '대기')) +
      (total ? ` · ${current}/${total} (${pct}%)` : '');

    document.getElementById('result').textContent=
      `상태: ${p.message || '-'}
` +
      `현재: ${current}/${total || '-'}
` +
      `추가: ${p.added || 0}
` +
      `재사용: ${p.reused || 0}
` +
      `실패: ${p.failed || 0}
` +
      `시작: ${p.started_at || '-'}
` +
      `완료: ${p.finished_at || '-'}
` +
      `오류: ${p.error || '없음'}
` +
      `현재 인덱스 상품 수: ${data.indexed_products || 0}
` +
      `마지막 인덱싱: ${data.last_indexed_at || '-'}`;

    setButtons(Boolean(p.running));

    if(p.running){
      clearTimeout(timer);
      timer=setTimeout(poll, 1500);
    }
  }catch(e){
    document.getElementById('status').textContent='진행상태 확인 오류';
    document.getElementById('result').textContent='오류: '+e.message;
    setButtons(false);
  }
}

poll();
</script>
</body>
</html>"""


@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        force = request.args.get("force", "0") == "1"
        result = build_index(force=force)
        return jsonify(result), 200 if result.get("success") else 409
    except Exception as e:
        return jsonify({"error": "재색인 실패", "detail": str(e)}), 500


@app.route("/refresh-index", methods=["POST"])
def refresh_index():
    try:
        force = request.args.get("force", "0") == "1"
        result = build_index(force=force)
        return jsonify(result), 200 if result.get("success") else 409
    except Exception as e:
        return jsonify({"error": "인덱스 갱신 실패", "detail": str(e)}), 500


@app.route("/reload-index", methods=["POST"])
def reload_index():
    global product_index
    product_index = []

    if load_index():
        return jsonify({
            "success": True,
            "indexed_products": len(product_index),
        })

    return jsonify({
        "success": False,
        "error": "저장된 AI 인덱스를 불러오지 못했습니다.",
    }), 500


@app.route("/search", methods=["POST"])
def image_search():
    try:
        if "image" not in request.files:
            return jsonify({"error": "이미지 파일이 전송되지 않았습니다."}), 400

        ensure_index()

        if request.args.get("refresh", "0") == "1":
            result = build_index(force=False)
            if not result.get("success") and result.get("message") != "이미 인덱싱 중입니다.":
                print("[SEARCH REFRESH WARNING]", result, flush=True)

        image_bytes = request.files["image"].read()
        if not image_bytes:
            return jsonify({"error": "업로드 이미지가 비어있습니다."}), 400

        query_image = Image.open(io.BytesIO(image_bytes))
        query_image = ImageOps.exif_transpose(query_image).convert("RGB")
        query_fp = calculate_fingerprint(query_image)
        query_hash = query_fp["phash_full"]

        matches = []

        for product in product_index:
            try:
                best_similarity = 0.0
                best_reference_image = product.get("image_url", "")

                features = product.get("images") or []
                if not features and product.get("fingerprint"):
                    features = [{
                        "image_url": product.get("image_url", ""),
                        "fingerprint": product.get("fingerprint"),
                    }]

                for feature in features:
                    fp = feature.get("fingerprint")
                    if not fp:
                        continue
                    sim = fingerprint_score(query_fp, fp)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_reference_image = feature.get("image_url") or best_reference_image

                score = best_similarity

                matches.append({
                    "score": round(float(score), 6),
                    "similarity": round(float(best_similarity), 6),
                    "similarity_percent": round(float(best_similarity) * 100.0, 2),
                    "title": str(product.get("title", "")),
                    "product_url": str(product.get("product_url", "")),
                    "image_url": str(product.get("image_url", "")),
                    "matched_reference_image": str(best_reference_image),
                    "price": str(product.get("price", "")),
                })

            except Exception as e:
                print("[SEARCH ITEM ERROR]", repr(e), flush=True)

        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:RESULT_LIMIT]

        if top_matches:
            best = top_matches[0]
            print(
                f"[SEARCH] BEST MATCH title={best['title']} "
                f"similarity={best['similarity']} score={best['score']} "
                f"url={best['product_url']}",
                flush=True
            )

        return jsonify({
            "success": True,
            "matches": top_matches,
            "indexed_products": int(len(product_index)),
            "last_indexed_at": last_indexed_at,
            "model": MODEL_NAME,
        })

    except Exception as e:
        print("[SEARCH ERROR]", repr(e), flush=True)
        return jsonify({
            "error": "AI 이미지 검색 서버 처리 중 오류가 발생했습니다.",
            "detail": str(e),
        }), 500


try:
    load_index()
except Exception as startup_error:
    print("[STARTUP INDEX ERROR]", repr(startup_error), flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
