from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os

app = Flask(__name__)
CORS(app)

# Render Environment에서 Google Vision API 키를 불러옵니다.
# Render 환경변수:
# Key   = GOOGLE_VISION_API_KEY
# Value = Google Cloud에서 발급받은 실제 API 키
GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Image Search API is running."
    })


@app.route("/search", methods=["POST"])
def image_search():

    # API 키 설정 확인
    if not GOOGLE_VISION_API_KEY:
        return jsonify({
            "error": "Google Vision API 키가 설정되지 않았습니다.",
            "detail": "Render Environment에 GOOGLE_VISION_API_KEY를 등록해주세요."
        }), 500

    # 이미지 업로드 확인
    if "image" not in request.files:
        return jsonify({
            "error": "이미지를 업로드해주세요."
        }), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    if not image_bytes:
        return jsonify({
            "error": "이미지 파일이 비어있습니다."
        }), 400

    # 이미지를 Base64로 변환
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Google Cloud Vision API
    url = (
        "https://vision.googleapis.com/v1/images:annotate"
        f"?key={GOOGLE_VISION_API_KEY}"
    )

    payload = {
        "requests": [
            {
                "image": {
                    "content": encoded_image
                },
                "features": [
                    {
                        "type": "WEB_DETECTION",
                        "maxResults": 30
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

    except requests.RequestException as e:
        return jsonify({
            "error": "Google Vision API 연결 실패",
            "detail": str(e)
        }), 502

    # Google이 4xx/5xx를 반환한 경우 실제 오류 내용을 전달
if not response.ok:
    try:
        google_error = response.json()
    except ValueError:
        google_error = response.text

    # Render Logs에서 Google의 실제 오류 확인
    print("========== GOOGLE VISION ERROR ==========", flush=True)
    print("Google Status:", response.status_code, flush=True)
    print("Google Response:", google_error, flush=True)
    print("=========================================", flush=True)

    return jsonify({
        "error": "Google Vision API 요청 실패",
        "google_status": response.status_code,
        "detail": google_error
    }), 502

    try:
        data = response.json()
    except ValueError:
        return jsonify({
            "error": "Google Vision API 응답을 읽을 수 없습니다."
        }), 502

    if not data.get("responses"):
        return jsonify({
            "error": "검색 결과가 없습니다."
        }), 404

    response_data = data["responses"][0]

    # Google Vision 자체 오류
    if "error" in response_data:
        return jsonify({
            "error": "Google Vision API 오류",
            "detail": response_data["error"]
        }), 502

    web = response_data.get("webDetection", {})

    results = []

    # 1. 완전히 동일한 이미지
    for image in web.get("fullMatchingImages", []):
        image_url = image.get("url")

        if image_url:
            results.append({
                "type": "exact",
                "image_url": image_url
            })

    # 2. 부분적으로 일치하는 이미지
    for image in web.get("partialMatchingImages", []):
        image_url = image.get("url")

        if image_url:
            results.append({
                "type": "partial",
                "image_url": image_url
            })

    # 3. 시각적으로 유사한 이미지
    for image in web.get("visuallySimilarImages", []):
        image_url = image.get("url")

        if image_url:
            results.append({
                "type": "similar",
                "image_url": image_url
            })

    # 이미지 URL 중복 제거
    unique_results = []
    used_urls = set()

    for result in results:
        image_url = result["image_url"]

        if image_url not in used_urls:
            used_urls.add(image_url)
            unique_results.append(result)

    # 이미지가 발견된 웹페이지
    pages = []
    used_page_urls = set()

    for page in web.get("pagesWithMatchingImages", []):
        page_url = page.get("url")

        if page_url and page_url not in used_page_urls:
            used_page_urls.add(page_url)

            pages.append({
                "title": page.get("pageTitle", ""),
                "url": page_url
            })

    # 기존 카페24 헤더 코드와 호환:
    # product_url이 있으면 해당 페이지로 이동합니다.
    first_page_url = pages[0]["url"] if pages else None

    return jsonify({
        "success": True,
        "product_url": first_page_url,
        "count": len(unique_results),
        "images": unique_results,
        "pages": pages
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
