from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from PIL import Image, ImageOps
import imagehash, requests, io, os, time, threading, json, re

app = Flask(__name__)
CORS(app)

SHOP_BASE_URL = os.environ.get("SHOP_BASE_URL", "https://freeorder1.cafe24.com").rstrip("/")
DATA_DIR = os.environ.get("DATA_DIR", "/var/data")
INDEX_FILE = os.path.join(DATA_DIR, "product_index.json")

MAX_CATEGORY_PAGES = int(os.environ.get("MAX_CATEGORY_PAGES", "200"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "15"))
RESULT_LIMIT = int(os.environ.get("RESULT_LIMIT", "12"))
AUTO_REFRESH_SECONDS = int(os.environ.get("AUTO_REFRESH_SECONDS", "300"))

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
        "version": 1,
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

    image_url = None
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        image_url = og_image.get("content")

    if not image_url:
        for selector in [".keyImg img", ".thumbnail img", ".prdImg img", "img.BigImage"]:
            img = soup.select_one(selector)
            if not img:
                continue

            image_url = img.get("src") or img.get("ec-data-src") or img.get("data-src")
            if image_url:
                break

    if image_url:
        image_url = urljoin(canonical_url, image_url)

    return {
        "product_url": canonical_url,
        "title": title,
        "image_url": image_url,
        "price": extract_price(soup),
    }


# =========================================================
# Image hash
# =========================================================
def download_image(image_url):
    r = session.get(image_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    image = Image.open(io.BytesIO(r.content))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def calculate_hash(image):
    return str(imagehash.phash(image, hash_size=16))


def hash_distance(hash_a, hash_b):
    return int(
        imagehash.hex_to_hash(hash_a)
        - imagehash.hex_to_hash(hash_b)
    )


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

            if not force and product_url in existing_map:
                new_index.append(existing_map[product_url])
                reused += 1
                continue

            try:
                info = extract_product_info(product_url)
                final_url = info["product_url"]

                if not force and final_url in existing_map:
                    new_index.append(existing_map[final_url])
                    reused += 1
                    continue

                if not info["image_url"]:
                    failed += 1
                    continue

                image = download_image(info["image_url"])
                image_hash = calculate_hash(image)

                new_index.append({
                    "title": str(info["title"]),
                    "product_url": str(final_url),
                    "image_url": str(info["image_url"]),
                    "price": str(info.get("price", "")),
                    "phash": str(image_hash),
                })

                added += 1

                if idx % 25 == 0 or idx == len(product_urls):
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
        }

    finally:
        indexing = False



def refresh_index_if_needed():
    """
    검색 시 일정 시간이 지났으면 신규/삭제 상품을 자동 반영한다.
    기본값: 300초(5분). AUTO_REFRESH_SECONDS 환경변수로 조정 가능.
    """
    global last_refresh_check_at

    now = time.time()
    if AUTO_REFRESH_SECONDS <= 0:
        return

    if now - last_refresh_check_at < AUTO_REFRESH_SECONDS:
        return

    last_refresh_check_at = now

    # 다른 요청에서 인덱싱 중이면 건너뜀
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
        "message": "Cafe24 Image Search API is running.",
        "shop": SHOP_BASE_URL,
        "indexed_products": int(len(product_index)),
        "indexing": bool(indexing),
        "last_indexed_at": last_indexed_at,
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
        "index_file": INDEX_FILE,
        "index_file_exists": os.path.exists(INDEX_FILE),
    })


@app.route("/reindex", methods=["POST"])
def reindex():
    """
    신규/삭제 상품 반영:
      POST /reindex

    전체 재생성:
      POST /reindex?force=1
    """
    try:
        force = request.args.get("force", "0") == "1"
        result = build_index(force=force)
        return jsonify(result), 200 if result.get("success") else 409
    except Exception as e:
        return jsonify({"error": "재색인 실패", "detail": str(e)}), 500



@app.route("/refresh-index", methods=["POST"])
def refresh_index():
    """
    신규/삭제 상품 빠른 반영:
      POST /refresh-index

    전체 강제 재생성:
      POST /refresh-index?force=1
    """
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
        "error": "저장된 인덱스를 불러오지 못했습니다.",
    }), 500


@app.route("/search", methods=["POST"])
def image_search():
    try:
        if "image" not in request.files:
            return jsonify({"error": "이미지 파일이 전송되지 않았습니다."}), 400

        ensure_index()

        # 클라이언트에서 refresh=1을 보내면 검색 직전 신규 상품을 즉시 반영
        if request.args.get("refresh", "0") == "1":
            result = build_index(force=False)
            if not result.get("success") and result.get("message") != "이미 인덱싱 중입니다.":
                print("[SEARCH REFRESH WARNING]", result, flush=True)

        image_bytes = request.files["image"].read()
        if not image_bytes:
            return jsonify({"error": "업로드 이미지가 비어있습니다."}), 400

        query_image = Image.open(io.BytesIO(image_bytes))
        query_image = ImageOps.exif_transpose(query_image).convert("RGB")
        query_hash = calculate_hash(query_image)

        matches = []

        for product in product_index:
            try:
                distance = hash_distance(query_hash, product["phash"])

                matches.append({
                    "distance": int(distance),
                    "title": str(product.get("title", "")),
                    "product_url": str(product.get("product_url", "")),
                    "image_url": str(product.get("image_url", "")),
                    "price": str(product.get("price", "")),
                })

            except Exception as e:
                print("[SEARCH ITEM ERROR]", repr(e), flush=True)

        matches.sort(key=lambda x: x["distance"])
        top_matches = matches[:RESULT_LIMIT]

        if top_matches:
            best = top_matches[0]
            print(
                f"[SEARCH] BEST MATCH title={best['title']} "
                f"distance={best['distance']} url={best['product_url']}",
                flush=True
            )

        return jsonify({
            "success": True,
            "matches": top_matches,
            "indexed_products": int(len(product_index)),
            "last_indexed_at": last_indexed_at,
        })

    except Exception as e:
        print("[SEARCH ERROR]", repr(e), flush=True)
        return jsonify({
            "error": "이미지 검색 서버 처리 중 오류가 발생했습니다.",
            "detail": str(e),
        }), 500


# 서버 시작 시 /var/data/product_index.json 자동 로드
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
