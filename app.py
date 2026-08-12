from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import traceback

app = Flask(__name__)
CORS(app)

GOOGLE_VISION_API_KEY = "AIzaSyDuh1zA4VFo-a4PP4NCDGfJBJBxawbbVSQ"

@app.route("/search", methods=["POST"])
def image_search():
    try:
        if "image" not in request.files:
            return jsonify({"error": "이미지를 업로드해주세요."}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({"error": "이미지 파일이 비어있습니다."}), 400

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

        payload = {
            "requests": [
                {
                    "image": {"content": encoded_image},
                    "features": [{"type": "WEB_DETECTION", "maxResults": 30}]
                }
            ]
        }

        response = requests.post(url, json=payload, timeout=30)
        
        # Render 로그창에 구글 응답 상태 및 에러 메시지 출력
        print("=== GOOGLE RESPONSE STATUS ===")
        print(response.status_code)
        print("=== GOOGLE RESPONSE BODY ===")
        print(response.text)

        data = response.json()

        if response.status_code != 200:
            return jsonify({"error": "Google Vision API 오류", "details": data}), 500

        if not data.get("responses"):
            return jsonify({"error": "검색 결과가 없습니다."}), 404

        response_data = data["responses"][0]

        if "error" in response_data:
            return jsonify({"error": response_data["error"]}), 500

        web = response_data.get("webDetection", {})

        results = []
        for image in web.get("fullMatchingImages", []):
            if image.get("url"):
                results.append({"type": "exact", "image_url": image.get("url")})

        for image in web.get("partialMatchingImages", []):
            if image.get("url"):
                results.append({"type": "partial", "image_url": image.get("url")})

        for image in web.get("visuallySimilarImages", []):
            if image.get("url"):
                results.append({"type": "similar", "image_url": image.get("url")})

        unique_results = []
        used_urls = set()
        for result in results:
            if result["image_url"] not in used_urls:
                used_urls.add(result["image_url"])
                unique_results.append(result)

        pages = []
        for page in web.get("pagesWithMatchingImages", []):
            if page.get("url"):
                pages.append({"title": page.get("pageTitle", ""), "url": page.get("url")})

        first_page_url = pages[0]["url"] if pages else None

        return jsonify({
            "success": True,
            "product_url": first_page_url,
            "count": len(unique_results),
            "images": unique_results,
            "pages": pages
        })

    except Exception as e:
        print("=== CRITICAL PYTHON SERVER ERROR ===")
        print(traceback.format_exc())
        return jsonify({"error": "파이썬 서버 내부 오류 발생", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
