from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from PIL import Image, ImageOps, ImageEnhance
import imagehash, requests, io, os, time, threading, json, re
import numpy as np

# Deep visual embedding (DINOv2)
import torch
from transformers import AutoImageProcessor, AutoModel

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
MODEL_NAME = os.environ.get("VISION_MODEL", "facebook/dinov2-small")
EMBEDDING_VERSION = f"dinov2:{MODEL_NAME}:design-gray-v2"

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
last_indexed_at = None
last_refresh_check_at = 0

_image_processor = None
_vision_model = None
_device = None


# =========================================================
# AI model
# =========================================================
def get_vision_model():
    global _image_processor, _vision_model, _device

    if _vision_model is not None and _image_processor is not None:
        return _image_processor, _vision_model, _device

    with model_lock:
        if _vision_model is not None and _image_processor is not None:
            return _image_processor, _vision_model, _device

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[AI MODEL] loading {MODEL_NAME} on {_device}", flush=True)

        _image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _vision_model = AutoModel.from_pretrained(MODEL_NAME)
        _vision_model.eval()
        _vision_model.to(_device)

        print("[AI MODEL] ready", flush=True)

    return _image_processor, _vision_model, _device


def prepare_design_image(image):
    """
    색상 차이를 최대한 제거하고 디자인/형태/봉제 디테일 중심으로 비교하기 위한 전처리.
    - RGB 색상을 그레이스케일로 제거
    - 명암 자동 보정으로 검정/아이보리/그레이 등 서로 다른 컬러의 형태를 비슷하게 만듦
    - 약한 대비/선명도 보정으로 넥라인, 밑단, 스티치, 소매선 같은 구조를 강조
    """
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Extremely large/detail-page images are downscaled before preprocessing.
    max_side = 1400
    if max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS
        )

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.10)
    gray = ImageEnhance.Sharpness(gray).enhance(1.15)

    # DINO expects RGB input, but all three channels now carry identical
    # luminance information, so clothing colour itself contributes very little.
    return gray.convert("RGB")


def calculate_embedding(image):
    processor, model, device = get_vision_model()

    design_image = prepare_design_image(image)
    inputs = processor(images=design_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding[0].detach().cpu().numpy().astype(np.float32)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


# =========================================================
# Persistent index
# =========================================================
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def rebuild_url_map():
    global product_index_by_url
    product_index_by_url = {
        item["product_url"]: item
        for item in product_index
        if item.get("product_url")
    }


def save_index():
    ensure_data_dir()
    payload = {
        "version": 2,
        "embedding_version": EMBEDDING_VERSION,
        "search_mode": "color_invariant_design",
        "shop": SHOP_BASE_URL,
        "last_indexed_at": last_indexed_at,
        "count": len(product_index),
        "products": product_index,
    }

    temp_path = INDEX_FILE + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    os.replace(temp_path, INDEX_FILE)
    print(f"[INDEX SAVE] {len(product_index)} products -> {INDEX_FILE}", flush=True)


def load_index():
    global product_index, last_indexed_at

    ensure_data_dir()

    if not os.path.exists(INDEX_FILE):
        print("[INDEX LOAD] no saved AI index", flush=True)
        return False

    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("embedding_version") != EMBEDDING_VERSION:
            print("[INDEX LOAD] embedding version changed; rebuild required", flush=True)
            return False

        products = payload.get("products", [])
        if not isinstance(products, list):
            raise ValueError("Invalid product index format")

        product_index = products
        last_indexed_at = payload.get("last_indexed_at")
        rebuild_url_map()

        print(f"[INDEX LOAD] loaded {len(product_index)} AI products", flush=True)
        return True

    except Exception as e:
        print("[INDEX LOAD ERROR]", repr(e), flush=True)
        return False


# =========================================================
# URL helpers
# =========================================================
def normalize_url(url, allow_external=False):
    if not url:
        return None

    absolute = urljoin(SHOP_BASE_URL + "/", url)
    parsed = urlparse(absolute)

    if not allow_external:
        base_host = urlparse(SHOP_BASE_URL).netloc
        if parsed.netloc and parsed.netloc != base_host:
            return None

    return urlunparse(parsed._replace(fragment=""))


def is_product_url(url):
    if not url:
        return False

    parsed = urlparse(url)
    path = parsed.path.lower()
    params = dict(parse_qsl(parsed.query))

    blocked = [
        "/board/", "/product/image_zoom", "/product/zoom",
        "/product/list.html", "/product/search.html",
        "/product/recent_view_product.html", "/product/compare.html",
        "/order/", "/myshop/", "/member/"
    ]

    if any(x in path for x in blocked):
        return False

    if path.endswith("/product/detail.html"):
        return bool(params.get("product_no"))

    if "/product/" in path:
        parts = [p for p in parsed.path.split("/") if p]
        try:
            i = parts.index("product")
        except ValueError:
            return False

        remaining = parts[i + 1:]
        return len(remaining) >= 2 and remaining[1].isdigit()

    return False


def is_category_url(url):
    if not url:
        return False

    parsed = urlparse(url)
    path = parsed.path.lower()
    params = dict(parse_qsl(parsed.query))

    if "/category/" in path:
        return True

    return path.endswith("/product/list.html") and "cate_no" in params


def with_page(url, page_no):
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["page"] = str(page_no)
    return urlunparse(parsed._replace(query=urlencode(query)))


# =========================================================
# Crawling
# =========================================================
def fetch_html(url):
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text


def discover_categories():
    categories = set()
    soup = BeautifulSoup(fetch_html(SHOP_BASE_URL + "/"), "html.parser")

    for a in soup.find_all("a", href=True):
        url = normalize_url(a.get("href"))
        if url and is_category_url(url):
            categories.add(url)

    return sorted(categories)


def discover_product_urls():
    product_urls = set()

    try:
        soup = BeautifulSoup(fetch_html(SHOP_BASE_URL + "/"), "html.parser")
        for a in soup.find_all("a", href=True):
            url = normalize_url(a.get("href"))
            if url and is_product_url(url):
                product_urls.add(url)
    except Exception as e:
        print("[INDEX] main page crawl failed:", repr(e), flush=True)

    categories = discover_categories()
    print(f"[INDEX] categories={len(categories)}", flush=True)

    for category_url in categories:
        previous_signature = None

        for page in range(1, MAX_CATEGORY_PAGES + 1):
            page_url = with_page(category_url, page)

            try:
                soup = BeautifulSoup(fetch_html(page_url), "html.parser")
            except Exception as e:
                print("[INDEX] category page failed:", page_url, repr(e), flush=True)
                break

            found = set()
            for a in soup.find_all("a", href=True):
                url = normalize_url(a.get("href"))
                if url and is_product_url(url):
                    found.add(url)

            if not found:
                break

            signature = tuple(sorted(found))
            if signature == previous_signature:
                break

            previous_signature = signature
            product_urls.update(found)

            print(
                f"[INDEX] category_page={page} total_urls={len(product_urls)}",
                flush=True
            )

    return sorted(product_urls)


# =========================================================
# Product parsing
# =========================================================
def find_canonical_product_url(soup, fallback_url):
    candidates = []

    canonical = soup.find("link", attrs={"rel": "canonical"})
    if canonical and canonical.get("href"):
        candidates.append(canonical.get("href"))

    og_url = soup.find("meta", attrs={"property": "og:url"})
    if og_url and og_url.get("content"):
        candidates.append(og_url.get("content"))

    candidates.append(fallback_url)

    for candidate in candidates:
        url = normalize_url(candidate)
        if url and is_product_url(url):
            return url

    return fallback_url


def clean_price(value):
    if value is None:
        return ""

    text = str(value).strip()
    numbers = re.sub(r"[^0-9]", "", text)

    if not numbers:
        return text

    try:
        return f"{int(numbers):,}원"
    except ValueError:
        return text


def extract_price(soup):
    meta_candidates = [
        ("property", "product:price:amount"),
        ("property", "og:price:amount"),
        ("name", "product:price:amount"),
        ("name", "price")
    ]

    for attr, value in meta_candidates:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            price = clean_price(tag.get("content"))
            if price:
                return price

    for selector in [
        ".xans-product-detail .price",
        ".xans-product-detaildesign td span",
        ".product_price", ".price", "[data-price]"
    ]:
        el = soup.select_one(selector)
        if not el:
            continue

        value = el.get("data-price") or el.get_text(" ", strip=True)
        price = clean_price(value)
        if price:
            return price

    return ""


def _candidate_img_url(img, base_url):
    raw = (
        img.get("ec-data-src") or img.get("data-src") or img.get("data-original")
        or img.get("src")
    )
    if not raw or raw.startswith("data:"):
        return None

    url = urljoin(base_url, raw)
    lower = url.lower()

    # Skip obvious UI assets/icons.
    blocked_words = [
        "icon", "btn_", "button", "loading", "spinner", "logo", "banner",
        "common/", "board/", "review", "coupon", "arrow", "close", "ico_"
    ]
    if any(x in lower for x in blocked_words):
        return None

    return url


def extract_product_info(product_url):
    soup = BeautifulSoup(fetch_html(product_url), "html.parser")
    canonical_url = find_canonical_product_url(soup, product_url)

    title = ""
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title = og_title.get("content").strip()

    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(" ", strip=True)

    image_urls = []
    seen = set()

    def add_url(u):
        if not u:
            return
        u = urljoin(canonical_url, u)
        if u.startswith("data:") or u in seen:
            return
        seen.add(u)
        image_urls.append(u)

    # Main product image first.
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        add_url(og_image.get("content"))

    # Cafe24 main/additional/detail product images.
    selectors = [
        ".keyImg img",
        ".xans-product-addimage img",
        ".xans-product-detail img",
        "#prdDetail img",
        ".edibot-product-detail img",
        ".detailArea img",
        ".thumbnail img",
        ".prdImg img",
        "img.BigImage",
    ]

    for selector in selectors:
        for img in soup.select(selector):
            u = _candidate_img_url(img, canonical_url)
            if u:
                add_url(u)
            if len(image_urls) >= MAX_PRODUCT_IMAGES:
                break
        if len(image_urls) >= MAX_PRODUCT_IMAGES:
            break

    # Fallback if the theme uses unusual classes.
    if len(image_urls) < 2:
        for img in soup.find_all("img"):
            u = _candidate_img_url(img, canonical_url)
            if u:
                add_url(u)
            if len(image_urls) >= MAX_PRODUCT_IMAGES:
                break

    return {
        "product_url": canonical_url,
        "title": title,
        "image_url": image_urls[0] if image_urls else None,
        "image_urls": image_urls[:MAX_PRODUCT_IMAGES],
        "price": extract_price(soup),
    }


# =========================================================
# Image functions
# =========================================================
def download_image(image_url):
    r = session.get(image_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    image = Image.open(io.BytesIO(r.content))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def calculate_hash(image):
    design_image = prepare_design_image(image)
    return str(imagehash.phash(design_image, hash_size=16))


def hash_distance(hash_a, hash_b):
    return int(imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b))


def usable_existing_item(item):
    if not item:
        return False
    if item.get("embedding_version") != EMBEDDING_VERSION:
        return False
    images = item.get("images") or []
    return any(x.get("embedding") for x in images if isinstance(x, dict))


# =========================================================
# Indexing
# =========================================================
def build_index(force=False):
    global product_index, indexing, last_indexed_at

    with index_lock:
        if indexing:
            return {"success": False, "message": "이미 인덱싱 중입니다."}
        indexing = True

    started = time.time()

    try:
        existing_map = {}
        if not force:
            existing_map = {
                item.get("product_url"): item
                for item in product_index
                if item.get("product_url")
            }

        product_urls = discover_product_urls()
        print(f"[INDEX] discovered={len(product_urls)}", flush=True)

        new_index = []
        added = reused = failed = 0

        for idx, product_url in enumerate(product_urls, start=1):
            existing = existing_map.get(product_url)
            if not force and usable_existing_item(existing):
                new_index.append(existing)
                reused += 1
                continue

            try:
                info = extract_product_info(product_url)
                final_url = info["product_url"]

                existing = existing_map.get(final_url)
                if not force and usable_existing_item(existing):
                    new_index.append(existing)
                    reused += 1
                    continue

                if not info["image_urls"]:
                    failed += 1
                    continue

                image_entries = []
                for image_url in info["image_urls"]:
                    try:
                        image = download_image(image_url)

                        # Skip tiny theme/UI images that slipped through HTML filtering.
                        if image.width < 180 or image.height < 180:
                            continue

                        embedding = calculate_embedding(image)
                        image_entries.append({
                            "image_url": str(image_url),
                            "phash": str(calculate_hash(image)),
                            "embedding": embedding.tolist(),
                        })
                    except Exception as image_error:
                        print("[INDEX] image failed:", image_url, repr(image_error), flush=True)

                if not image_entries:
                    failed += 1
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

                if idx % 10 == 0 or idx == len(product_urls):
                    print(
                        f"[INDEX] {idx}/{len(product_urls)} "
                        f"added={added} reused={reused} failed={failed}",
                        flush=True
                    )

            except Exception as e:
                failed += 1
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
        "search_mode": "color_invariant_design",
        "model": MODEL_NAME,
        "index_file": INDEX_FILE,
        "index_file_exists": os.path.exists(INDEX_FILE),
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
.box{max-width:620px;margin:auto;background:#fff;border-radius:18px;padding:32px;box-shadow:0 8px 30px rgba(0,0,0,.08)}
h1{font-size:24px;margin:0 0 12px}.desc{color:#666;line-height:1.6;margin-bottom:24px}
button{width:100%;border:0;border-radius:12px;padding:16px;font-size:16px;font-weight:700;cursor:pointer;margin-top:10px}
.refresh{background:#111;color:#fff}.force{background:#eee;color:#111}
#result{white-space:pre-wrap;background:#f7f7f7;padding:16px;border-radius:12px;margin-top:20px;min-height:60px;line-height:1.5}
.small{font-size:13px;color:#888;margin-top:14px}
</style>
</head>
<body>
<div class="box">
<h1>상품 이미지 검색 DB 관리</h1>
<div class="desc">
새 상품을 등록한 뒤에는 <b>신규 상품 반영</b>을 누르세요.<br>
검색 방식이 바뀌었거나 전체 DB를 다시 만들 때만 <b>전체 강제 재생성</b>을 사용하세요.
</div>
<button class="refresh" onclick="run(false)">신규 상품 반영</button>
<button class="force" onclick="run(true)">전체 강제 재생성</button>
<div id="result">대기 중</div>
<div class="small">전체 재생성은 상품 수에 따라 시간이 오래 걸릴 수 있습니다. 작업 중에는 창을 닫지 않는 것을 권장합니다.</div>
</div>
<script>
async function run(force){
  const result=document.getElementById('result');
  if(force && !confirm('전체 상품 이미지 특징값을 다시 생성합니다. 계속할까요?')) return;
  result.textContent = force ? '전체 재생성 중... 잠시 기다려주세요.' : '신규 상품 확인 중...';
  try{
    const r=await fetch('/reindex' + (force ? '?force=1' : ''), {method:'POST'});
    const data=await r.json();
    result.textContent=JSON.stringify(data,null,2);
  }catch(e){
    result.textContent='오류: '+e;
  }
}
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
        query_embedding = calculate_embedding(query_image)
        query_hash = calculate_hash(query_image)

        matches = []

        for product in product_index:
            try:
                best_similarity = -1.0
                best_hash_distance = 999
                best_reference_image = product.get("image_url", "")

                for ref in product.get("images", []):
                    embedding = ref.get("embedding")
                    if not embedding:
                        continue

                    sim = cosine_similarity(query_embedding, embedding)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_reference_image = ref.get("image_url", best_reference_image)

                    try:
                        distance = hash_distance(query_hash, ref.get("phash", ""))
                        best_hash_distance = min(best_hash_distance, distance)
                    except Exception:
                        pass

                if best_similarity < -0.5:
                    continue

                # Grayscale DINO design similarity is the main signal.
                # pHash is colour-independent too and is used only as a tiny
                # near-identical/crop tie breaker.
                hash_bonus = 0.0
                if best_hash_distance != 999:
                    hash_bonus = max(0.0, 1.0 - (best_hash_distance / 256.0))

                score = (best_similarity * 0.985) + (hash_bonus * 0.015)

                matches.append({
                    "score": round(float(score), 6),
                    "similarity": round(float(best_similarity), 6),
                    "similarity_percent": round(float(best_similarity) * 100.0, 2),
                    "distance": int(best_hash_distance) if best_hash_distance != 999 else None,
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
