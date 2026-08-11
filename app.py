from flask import Flask, request, jsonify
from flask_cors import CORS  # 👈 CORS 불러오기
import requests
import base64

app = Flask(__name__)
CORS(app)  # 👈 모든 도메인에서의 접속 요청 허용 설정

# ==============================
# AIzaSyDuh1zA4VFo-a4PP4NCDGfJBJBxawbbVSQ
# ==============================
GOOGLE_VISION_API_KEY = "여기에_API_KEY_입력"


# 카페24 스크립트의 /search 경로와 통일
@app.route("/search", methods=["POST"])
def image_search():
    if "image" not in request.files:
        return jsonify({"error": "이미지를 업로드해주세요."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    if not image_bytes:
        return jsonify({"error": "이미지 파일이 비어있습니다."}), 400

    # 이미지를 Base64로 변환
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Google Vision API 요청
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

        response.raise_for_status()

    except requests.RequestException as e:
        return jsonify({
            "error": "Google Vision API 요청 실패",
            "detail": str(e)
        }), 500

    data = response.json()

    if not data.get("responses"):
        return jsonify({
            "error": "검색 결과가 없습니다."
        }), 404

    response_data = data["responses"][0]

    # Google Vision 자체 오류
    if "error" in response_data:
        return jsonify({
            "error": response_data["error"]
        }), 500

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

    # 중복 URL 제거
    unique_results = []
    used_urls = set()

    for result in results:
        if result["image_url"] not in used_urls:
            used_urls.add(result["image_url"])
            unique_results.append(result)

    # 이미지가 발견된 웹페이지
    pages = []

    for page in web.get("pagesWithMatchingImages", []):
        page_url = page.get("url")

        if page_url:
            pages.append({
                "title": page.get("pageTitle", ""),
                "url": page_url
            })

    # 카페24 연동용 결과 반환
    first_page_url = pages[0]["url"] if pages else None

    return jsonify({
        "success": True,
        "product_url": first_page_url,
        "count": len(unique_results),
        "images": unique_results,
        "pages": pages
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
