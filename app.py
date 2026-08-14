from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from PIL import Image, ImageOps
import imagehash
import requests
import io
import os
import time
import threading
import re

app = Flask(__name__)
CORS(app)

SHOP_BASE_URL = os.environ.get(
    "SHOP_BASE_URL",
    "https://freeorder1.cafe24.com"
).rstrip("/")

MAX_CATEGORY_PAGES = int(os.environ.get("MAX_CATEGORY_PAGES", "20"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "15"))
RESULT_LIMIT = int(os.environ.get("RESULT_LIMIT", "12"))
MATCH_THRESHOLD = int(os.environ.get("MATCH_THRESHOLD", "40"))

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0 Safari/537.36"
)

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"
})

product_index = []
index_lock = threading.Lock()
indexing = False
last_indexed_at = None


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
        "/board/",
        "/product/image_zoom",
        "/product/zoom",
        "/product/list.html",
        "/product/search.html",
        "/product/recent_view_product.html",
        "/product/compare.html",
        "/order/",
        "/myshop/",
        "/member/"
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

        if len(remaining) >= 2 and remaining[1].isdigit():
            return True

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
        print("[INDEX] 메인 수집 실패:", repr(e), flush=True)

    categories = discover_categories()
    print(f"[INDEX] 카테고리 {len(categories)}개", flush=True)

    for category_url in categories:
        previous_count = -1

        for page in range(1, MAX_CATEGORY_PAGES + 1):
            page_url = with_page(category_url, page)

            try:
                soup = BeautifulSoup(fetch_html(page_url), "html.parser")
            except Exception as e:
                print("[INDEX] 카테고리 실패:", page_url, repr(e), flush=True)
                break

            found = set()

            for a in soup.find_all("a", href=True):
                url = normalize_url(a.get("href"))
                if url and is_product_url(url):
                    found.add(url)

            if not found:
                break

            before = len(product_urls)
            product_urls.update(found)
            after = len(product_urls)

            if after == before and previous_count == after:
                break

            previous_count = after

    return sorted(product_urls)


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
    # Open Graph / product meta 우선
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

    # 카페24에서 자주 쓰이는 가격 영역
    selectors = [
        ".xans-product-detail .price",
        ".xans-product-detaildesign td span",
        ".product_price",
        ".price",
        "[data-price]"
    ]

    for selector in selectors:
        el = soup.select_one(selector)
        if not el:
            continue

        if el.get("data-price"):
            price = clean_price(el.get("data-price"))
        else:
            price = clean_price(el.get_text(" ", strip=True))

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

            candidate = img.get("src") or img.get("ec-data-src") or img.get("data-src")
            if candidate:
                image_url = candidate
                break

    if image_url:
        image_url = urljoin(canonical_url, image_url)

    return {
        "product_url": canonical_url,
        "title": title,
        "image_url": image_url,
        "price": extract_price(soup)
    }


def download_image(image_url):
    r = session.get(image_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    image = Image.open(io.BytesIO(r.content))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def calculate_hash(image):
    return str(imagehash.phash(image, hash_size=16))


def hash_distance(hash_a, hash_b):
    a = imagehash.hex_to_hash(hash_a)
    b = imagehash.hex_to_hash(hash_b)
    return int(a - b)


def build_index():
    global product_index, indexing, last_indexed_at

    with index_lock:
        if indexing:
            return {"success": False, "message": "이미 인덱싱 중입니다."}
        indexing = True

    started = time.time()
    new_index = []
    used_urls = set()

    try:
        product_urls = discover_product_urls()
        print(f"[INDEX] 실제 상품 URL {len(product_urls)}개", flush=True)

        for idx, product_url in enumerate(product_urls, start=1):
            try:
                info = extract_product_info(product_url)

                if info["product_url"] in used_urls:
                    continue

                used_urls.add(info["product_url"])

                if not info["image_url"]:
                    print("[INDEX] 대표 이미지 없음:", info["product_url"], flush=True)
                    continue

                image = download_image(info["image_url"])

                new_index.append({
                    "title": info["title"],
                    "product_url": info["product_url"],
                    "image_url": info["image_url"],
                    "price": info["price"],
                    "phash": calculate_hash(image)
                })

                print(
                    f"[INDEX] {idx}/{len(product_urls)} {info['title']} -> OK",
                    flush=True
                )

            except Exception as e:
                print("[INDEX] 상품 실패:", product_url, repr(e), flush=True)

        with index_lock:
            product_index = new_index
            last_indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")

        elapsed = round(time.time() - started, 2)

        print(f"[INDEX] 완료 {len(new_index)}개 / {elapsed}초", flush=True)

        return {
            "success": True,
            "indexed": int(len(new_index)),
            "elapsed_seconds": float(elapsed),
            "last_indexed_at": last_indexed_at
        }

    finally:
        indexing = False


def ensure_index():
    if product_index:
        return

    result = build_index()

    if not result.get("success") and not product_index:
        raise RuntimeError(result.get("message", "인덱스 생성 실패"))


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Cafe24 Image Search API is running.",
        "shop": SHOP_BASE_URL,
        "indexed_products": int(len(product_index)),
        "indexing": bool(indexing),
        "last_indexed_at": last_indexed_at
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
        "match_threshold": int(MATCH_THRESHOLD)
    })


@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        result = build_index()
        return jsonify(result), 200 if result.get("success") else 409
    except Exception as e:
        return jsonify({"error": "재색인 실패", "detail": str(e)}), 500


@app.route("/search", methods=["POST"])
def image_search():
    try:
        if "image" not in request.files:
            return jsonify({"error": "이미지 파일이 전송되지 않았습니다."}), 400

        ensure_index()

        image_bytes = request.files["image"].read()
        if not image_bytes:
            return jsonify({"error": "업로드 이미지가 비어있습니다."}), 400

        query_image = Image.open(io.BytesIO(image_bytes))
        query_image = ImageOps.exif_transpose(query_image).convert("RGB")
        query_hash = calculate_hash(query_image)

        matches = []

        for product in product_index:
            distance = int(hash_distance(query_hash, product["phash"]))

            matches.append({
                "distance": distance,
                "title": str(product["title"]),
                "product_url": str(product["product_url"]),
                "image_url": str(product["image_url"]),
                "price": str(product.get("price", ""))
            })

        matches.sort(key=lambda x: x["distance"])
        top_matches = matches[:RESULT_LIMIT]

        # 결과 페이지에서는 여러 상품을 보여주므로 product_url로 바로 이동하지 않습니다.
        return jsonify({
            "success": True,
            "matches": top_matches,
            "indexed_products": int(len(product_index))
        })

    except Exception as e:
        print("[SEARCH ERROR]", repr(e), flush=True)
        return jsonify({
            "error": "이미지 검색 서버 처리 중 오류가 발생했습니다.",
            "detail": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
