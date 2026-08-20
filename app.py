from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import imagehash, requests, io, os, time, threading, json, re

app = Flask(__name__)
CORS(app)

SHOP_BASE_URL = os.environ.get("SHOP_BASE_URL", "https://freeorder1.cafe24.com").rstrip("/")
DATA_DIR = os.environ.get("DATA_DIR", "/var/data")
INDEX_FILE = os.path.join(DATA_DIR, "product_index.json")

MAX_CATEGORY_PAGES = int(os.environ.get("MAX_CATEGORY_PAGES", "200"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "15"))
RESULT_LIMIT = int(os.environ.get("RESULT_LIMIT", "12"))
MAX_PRODUCT_IMAGES = int(os.environ.get("MAX_PRODUCT_IMAGES", "8"))
AUTO_REFRESH_SECONDS = int(os.environ.get("AUTO_REFRESH_SECONDS", "300"))

INDEX_VERSION = 4
SEARCH_MODE = "lightweight_color_invariant_design_v4"

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
indexing = False
last_indexed_at = None
last_refresh_check_at = 0

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
        "version": INDEX_VERSION,
        "search_mode": SEARCH_MODE,
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
        print("[INDEX LOAD] no saved index", flush=True)
        return False

    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("version") != INDEX_VERSION or payload.get("search_mode") != SEARCH_MODE:
            print("[INDEX LOAD] old/incompatible index ignored", flush=True)
            return False

        products = payload.get("products", [])
        if not isinstance(products, list):
            raise ValueError("Invalid product index format")

        product_index = products
        last_indexed_at = payload.get("last_indexed_at")
        rebuild_url_map()

        print(f"[INDEX LOAD] loaded {len(product_index)} products", flush=True)
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

    try:
        soup = BeautifulSoup(fetch_html(SHOP_BASE_URL + "/"), "html.parser")
    except Exception as e:
        print("[INDEX] main page category crawl failed:", repr(e), flush=True)
        return []

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


def looks_like_product_image(url):
    if not url:
        return False

    low = url.lower()

    blocked = [
        "icon", "logo", "banner", "btn_", "button", "loading",
        "common", "layout", "board", "member", "review", "coupon"
    ]
    if any(x in low for x in blocked):
        return False

    return True


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

    def add_image(value):
        if not value:
            return
        absolute = urljoin(canonical_url, value)
        if not looks_like_product_image(absolute):
            return
        if absolute not in image_urls:
            image_urls.append(absolute)

    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        add_image(og_image.get("content"))

    selectors = [
        ".keyImg img",
        ".thumbnail img",
        ".prdImg img",
        "img.BigImage",
        ".xans-product-detail img",
        ".xans-product-addimage img",
        ".detailArea img",
        "#prdDetail img",
        ".cont img",
    ]

    for selector in selectors:
        for img in soup.select(selector):
            src = (
                img.get("ec-data-src")
                or img.get("data-src")
                or img.get("src")
            )
            add_image(src)

            if len(image_urls) >= MAX_PRODUCT_IMAGES:
                break

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
# Lightweight, color-independent design fingerprint
# =========================================================
def download_image(image_url):
    r = session.get(image_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    image = Image.open(io.BytesIO(r.content))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def prepare_design_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")

    max_side = 1200
    if max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS
        )

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.08)
    return gray


def center_crop(image, ratio=0.82):
    w, h = image.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return image.crop((left, top, left + cw, top + ch))


def edge_image(gray):
    e = gray.filter(ImageFilter.FIND_EDGES)
    e = ImageOps.autocontrast(e)
    e = ImageEnhance.Contrast(e).enhance(1.25)
    return e


def calculate_fingerprint(image):
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


def hash_distance(hash_a, hash_b):
    return int(imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b))


def fingerprint_score(query_fp, product_fp):
    keys_weights = [
        ("phash_full", 0.10),
        ("phash_crop", 0.26),
        ("dhash_crop", 0.14),
        ("whash_crop", 0.10),
        ("edge_full", 0.14),
        ("edge_crop", 0.26),
    ]

    total = 0.0
    weight_sum = 0.0

    for key, weight in keys_weights:
        a = query_fp.get(key)
        b = product_fp.get(key)
        if not a or not b:
            continue

        dist = hash_distance(a, b)
        sim = max(0.0, 1.0 - (dist / 256.0))
        total += sim * weight
        weight_sum += weight

    return total / weight_sum if weight_sum else 0.0


def usable_existing_item(item):
    if not item:
        return False
    images = item.get("images")
    if not isinstance(images, list) or not images:
        return False
    return any(isinstance(x, dict) and x.get("fingerprint") for x in images)


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

                image_entries = []

                for image_url in info["image_urls"]:
                    try:
                        image = download_image(image_url)

                        if image.width < 180 or image.height < 180:
                            continue

                        fingerprint = calculate_fingerprint(image)
                        image_entries.append({
                            "image_url": str(image_url),
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
                    "image_url": str(info["image_url"] or image_entries[0]["image_url"]),
                    "price": str(info.get("price", "")),
                    "images": image_entries,
                })

                added += 1
                reindex_progress["added"] = added

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

        reindex_progress.update({
            "running": False,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current": len(product_urls),
            "total": len(product_urls),
            "added": added,
            "reused": reused,
            "failed": failed,
            "message": "완료",
            "error": None,
        })

        return {
            "success": True,
            "indexed": len(product_index),
            "added": added,
            "reused": reused,
            "failed": failed,
            "elapsed_seconds": elapsed,
            "last_indexed_at": last_indexed_at,
        }

    except Exception as e:
        reindex_progress.update({
            "running": False,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": "오류 발생",
            "error": str(e),
        })
        raise

    finally:
        indexing = False


def run_reindex_background(force=False):
    try:
        build_index(force=force)
    except Exception as e:
        print("[BACKGROUND REINDEX ERROR]", repr(e), flush=True)


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

    worker = threading.Thread(
        target=run_reindex_background,
        kwargs={"force": False},
        daemon=True,
    )
    worker.start()


def ensure_index():
    if product_index:
        refresh_index_if_needed()
        return

    if load_index():
        refresh_index_if_needed()
        return

    # 검색 요청을 오래 붙잡지 않도록 최초 인덱싱도 백그라운드 시작
    if not indexing:
        worker = threading.Thread(
            target=run_reindex_background,
            kwargs={"force": False},
            daemon=True,
        )
        worker.start()


# =========================================================
# Routes
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Cafe24 Lightweight Clothing Image Search API is running.",
        "shop": SHOP_BASE_URL,
        "indexed_products": len(product_index),
        "indexing": indexing,
        "last_indexed_at": last_indexed_at,
        "search_mode": SEARCH_MODE,
        "index_file_exists": os.path.exists(INDEX_FILE),
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "success": True,
        "shop": SHOP_BASE_URL,
        "indexed_products": len(product_index),
        "indexing": indexing,
        "last_indexed_at": last_indexed_at,
        "result_limit": RESULT_LIMIT,
        "max_product_images": MAX_PRODUCT_IMAGES,
        "search_mode": SEARCH_MODE,
        "progress": reindex_progress,
    })


@app.route("/reindex/start", methods=["POST"])
def reindex_start():
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


@app.route("/reindex", methods=["POST"])
def reindex_legacy():
    # 기존 호출 호환. 실제 작업은 백그라운드로 시작한다.
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
        "message": "재색인 작업을 백그라운드에서 시작했습니다.",
        "force": force,
    })


@app.route("/admin/reindex", methods=["GET"])
def admin_reindex_page():
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
</style>
</head>
<body>
<div class="box">
<h1>상품 이미지 검색 DB 관리</h1>
<div class="desc">새 상품은 <b>신규 상품 반영</b>, 검색 방식 변경 후에는 <b>전체 강제 재생성</b>을 사용하세요.</div>
<button id="refreshBtn" class="refresh" onclick="startJob(false)">신규 상품 반영</button>
<button id="forceBtn" class="force" onclick="startJob(true)">전체 강제 재생성</button>
<div class="barwrap"><div id="bar" class="bar"></div></div>
<div id="status" class="status">상태 확인 중...</div>
<div id="result">대기 중</div>
</div>
<script>
let timer=null;
function setButtons(v){
  document.getElementById('refreshBtn').disabled=v;
  document.getElementById('forceBtn').disabled=v;
}
async function startJob(force){
  if(force && !confirm('전체 상품 검색 DB를 다시 만듭니다. 계속할까요?')) return;
  setButtons(true);
  document.getElementById('status').textContent='작업 시작 요청 중...';
  try{
    const r=await fetch('/reindex/start'+(force?'?force=1':''),{method:'POST'});
    const data=await r.json();
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
    const r=await fetch('/reindex/progress',{cache:'no-store'});
    const data=await r.json();
    const p=data.progress||{};
    const total=Number(p.total||0);
    const current=Number(p.current||0);
    const pct=total?Math.min(100,Math.round(current/total*100)):0;
    document.getElementById('bar').style.width=pct+'%';
    document.getElementById('status').textContent=
      (p.running?'진행 중':(p.message||'대기'))+(total?` · ${current}/${total} (${pct}%)`:'');
    document.getElementById('result').textContent=
      `상태: ${p.message||'-'}\n현재: ${current}/${total||'-'}\n추가: ${p.added||0}\n재사용: ${p.reused||0}\n실패: ${p.failed||0}\n시작: ${p.started_at||'-'}\n완료: ${p.finished_at||'-'}\n오류: ${p.error||'없음'}\n현재 인덱스 상품 수: ${data.indexed_products||0}\n마지막 인덱싱: ${data.last_indexed_at||'-'}`;
    setButtons(Boolean(p.running));
    if(p.running){
      clearTimeout(timer);
      timer=setTimeout(poll,1500);
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


@app.route("/search", methods=["POST"])
def image_search():
    try:
        if "image" not in request.files:
            return jsonify({"error": "이미지 파일이 전송되지 않았습니다."}), 400

        ensure_index()

        if not product_index:
            return jsonify({
                "success": False,
                "error": "상품 검색 DB가 아직 준비되지 않았습니다.",
                "indexing": indexing,
                "progress": reindex_progress,
            }), 503

        image_bytes = request.files["image"].read()
        if not image_bytes:
            return jsonify({"error": "업로드 이미지가 비어있습니다."}), 400

        query_image = Image.open(io.BytesIO(image_bytes))
        query_image = ImageOps.exif_transpose(query_image).convert("RGB")
        query_fp = calculate_fingerprint(query_image)

        matches = []

        for product in product_index:
            try:
                best_similarity = 0.0
                best_reference_image = product.get("image_url", "")

                for feature in product.get("images", []):
                    fp = feature.get("fingerprint")
                    if not fp:
                        continue

                    sim = fingerprint_score(query_fp, fp)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_reference_image = feature.get("image_url") or best_reference_image

                matches.append({
                    "score": round(best_similarity, 6),
                    "similarity": round(best_similarity, 6),
                    "similarity_percent": round(best_similarity * 100.0, 2),
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

        return jsonify({
            "success": True,
            "matches": top_matches,
            "indexed_products": len(product_index),
            "last_indexed_at": last_indexed_at,
            "search_mode": SEARCH_MODE,
        })

    except Exception as e:
        print("[SEARCH ERROR]", repr(e), flush=True)
        return jsonify({
            "error": "이미지 검색 서버 처리 중 오류가 발생했습니다.",
            "detail": str(e),
        }), 500


try:
    load_index()
except Exception as startup_error:
    print("[STARTUP INDEX ERROR]", repr(startup_error), flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
