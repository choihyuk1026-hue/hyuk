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

app = Flask(__name__)
CORS(app)

# =========================================================
# 설정
# =========================================================
SHOP_BASE_URL = os.environ.get(
    "SHOP_BASE_URL",
    "https://freeorder1.cafe24.com"
).rstrip("/")

REINDEX_TOKEN = os.environ.get("REINDEX_TOKEN", "")
MAX_CATEGORY_PAGES = int(os.environ.get("MAX_CATEGORY_PAGES", "20"))
MATCH_THRESHOLD = int(os.environ.get("MATCH_THRESHOLD", "14"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "15"))

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


# =========================================================
# 공통 함수
# =========================================================
def normalize_url(url):
    if not url:
        return None

    url = urljoin(SHOP_BASE_URL + "/", url)
    parsed = urlparse(url)

    base_host = urlparse(SHOP_BASE_URL).netloc
    if parsed.netloc and parsed.netloc != base_host:
        return None

    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


def is_product_url(url):
    if not url:
        return False

    parsed = urlparse(url)
    path = parsed.path.lower()

    if "/product/" in path and path not in ["/product/list.html", "/product/search.html"]:
        return True

    if path.endswith("/product/detail.html"):
        params = dict(parse_qsl(parsed.query))
        return "product_no" in params

    return False


def is_category_url(url):
    if not url:
        return False

    parsed = urlparse(url)
    path = parsed.path.lower()
    params = dict(parse_qsl(parsed.query))

    if "/category/" in path:
        return True

    if path.endswith("/product/list.html") and "cate_no" in params:
        return True

    return False


def with_page(url, page_no):
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["page"] = str(page_no)
    return urlunparse(parsed._replace(query=urlencode(query)))


def fetch_html(url):
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def discover_categories():
    categories = set()

    html = fetch_html(SHOP_BASE_URL + "/")
    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        url = normalize_url(a.get("href"))
        if url and is_category_url(url):
            categories.add(url)

    return sorted(categories)


def discover_product_urls():
    product_urls = set()

    try:
        html = fetch_html(SHOP_BASE_URL + "/")
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            url = normalize_url(a.get("href"))
            if url and is_product_url(url):
                product_urls.add(url)

    except Exception as e:
        print("[INDEX] 메인 페이지 수집 실패:", repr(e), flush=True)

    categories = discover_categories()
    print(f"[INDEX] 발견한 카테고리: {len(categories)}개", flush=True)

    for category_url in categories:
        previous_count = -1

        for page in range(1, MAX_CATEGORY_PAGES + 1):
            page_url = with_page(category_url, page)

            try:
                html = fetch_html(page_url)

            except Exception as e:
                print(f"[INDEX] 카테고리 요청 실패: {page_url} / {e}", flush=True)
                break

            soup = BeautifulSoup(html, "html.parser")
            found_this_page = set()

            for a in soup.find_all("a", href=True):
                url = normalize_url(a.get("href"))
                if url and is_product_url(url):
                    found_this_page.add(url)

            if not found_this_page:
                break

            before = len(product_urls)
            product_urls.update(found_this_page)
            after = len(product_urls)

            print(
                f"[INDEX] {category_url} page={page} "
                f"상품 {len(found_this_page)}개 / 누적 {after}개",
                flush=True
            )

            if after == before and previous_count == after:
                break

            previous_count = after

    return sorted(product_urls)


def extract_product_info(product_url):
    html = fetch_html(product_url)
    soup = BeautifulSoup(html, "html.parser")

    title = None
    image_url = None

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title:
        title = og_title.get("content")

    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(" ", strip=True)

    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image:
        image_url = og_image.get("content")

    if not image_url:
        selectors = [
            ".keyImg img",
            ".thumbnail img",
            ".prdImg img",
            "img.BigImage",
            "img"
        ]

        for selector in selectors:
            img = soup.select_one(selector)
            if not img:
                continue

            candidate = (
                img.get("src")
                or img.get("ec-data-src")
                or img.get("data-src")
            )

            if candidate:
                image_url = candidate
                break

    normalized_image_url = normalize_url(image_url) if image_url else None

    if not normalized_image_url and image_url:
        normalized_image_url = urljoin(product_url, image_url)

    return {
        "product_url": product_url,
        "title": title or "",
        "image_url": normalized_image_url
    }


def download_image(image_url):
    response = session.get(image_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content))
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
            return {
                "success": False,
                "message": "이미 인덱싱 중입니다."
            }

        indexing = True

    started = time.time()
    new_index = []

    try:
        product_urls = discover_product_urls()

        print(f"[INDEX] 상품 상세 URL 총 {len(product_urls)}개 발견", flush=True)

        for idx, product_url in enumerate(product_urls, start=1):
            try:
                info = extract_product_info(product_url)

                if not info["image_url"]:
                    print(
                        f"[INDEX] 대표 이미지 없음: {product_url}",
                        flush=True
                    )
                    continue

                image = download_image(info["image_url"])
                image_hash = calculate_hash(image)

                new_index.append({
                    "title": info["title"],
                    "product_url": info["product_url"],
                    "image_url": info["image_url"],
                    "phash": image_hash
                })

                print(
                    f"[INDEX] {idx}/{len(product_urls)} "
                    f"{info['title']} -> OK",
                    flush=True
                )

            except Exception as e:
                print(
                    f"[INDEX] 상품 처리 실패: {product_url} / {repr(e)}",
                    flush=True
                )

        with index_lock:
            product_index = new_index
            last_indexed_at = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime()
            )

        elapsed = round(time.time() - started, 2)

        print(
            f"[INDEX] 완료: {len(new_index)}개 / {elapsed}초",
            flush=True
        )

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
        raise RuntimeError(
            result.get("message", "상품 인덱스를 만들 수 없습니다.")
        )


# =========================================================
# 라우트
# =========================================================
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
        "match_threshold": int(MATCH_THRESHOLD)
    })


@app.route("/reindex", methods=["POST"])
def reindex():
    if REINDEX_TOKEN:
        supplied = request.headers.get("X-Reindex-Token", "")
        if supplied != REINDEX_TOKEN:
            return jsonify({
                "error": "재색인 권한이 없습니다."
            }), 403

    try:
        result = build_index()
        status_code = 200 if result.get("success") else 409
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "error": "상품 인덱스 생성 실패",
            "detail": str(e)
        }), 500


@app.route("/search", methods=["POST"])
def image_search():
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "이미지 파일이 전송되지 않았습니다."
            }), 400

        ensure_index()

        image_file = request.files["image"]
        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({
                "error": "업로드된 이미지 파일이 비어있습니다."
            }), 400

        query_image = Image.open(io.BytesIO(image_bytes))
        query_image = ImageOps.exif_transpose(query_image).convert("RGB")
        query_hash = calculate_hash(query_image)

        matches = []

        for product in product_index:
            # 중요:
            # ImageHash가 numpy.int64를 반환할 수 있으므로
            # 반드시 Python 기본 int로 변환합니다.
            distance = int(
                hash_distance(
                    query_hash,
                    product["phash"]
                )
            )

            matches.append({
                "distance": int(distance),
                "title": str(product["title"]),
                "product_url": str(product["product_url"]),
                "image_url": str(product["image_url"])
            })

        matches.sort(key=lambda x: x["distance"])

        top_matches = matches[:5]
        best = top_matches[0] if top_matches else None

        if not best:
            return jsonify({
                "success": True,
                "product_url": None,
                "message": "등록된 상품 이미지가 없습니다.",
                "matches": []
            })

        print(
            f"[SEARCH] BEST MATCH: "
            f"title={best['title']} "
            f"distance={best['distance']} "
            f"url={best['product_url']}",
            flush=True
        )

        if int(best["distance"]) <= int(MATCH_THRESHOLD):
            return jsonify({
                "success": True,
                "product_url": best["product_url"],
                "best_match": best,
                "matches": top_matches,
                "indexed_products": int(len(product_index))
            })

        return jsonify({
            "success": True,
            "product_url": None,
            "message": "유사도가 충분히 높은 상품을 찾지 못했습니다.",
            "best_match": best,
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
