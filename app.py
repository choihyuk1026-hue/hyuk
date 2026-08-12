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
                    "features": [{"type": "WEB_DETECTION", "maxResults": 50}]
                }
            ]
        }

        response = requests.post(url, json=payload, timeout=30)
        data = response.json()

        if response.status_code != 200:
            return jsonify({"error": "Google Vision API 오류", "details": data}), 500

        if not data.get("responses"):
            return jsonify({"error": "검색 결과가 없습니다."}), 404

        response_data = data["responses"][0]
        web = response_data.get("webDetection", {})

        pages = web.get("pagesWithMatchingImages", [])
        
        # 1. 매칭된 웹페이지 URL 추출
        target_url = None
        if pages:
            target_url = pages[0].get("url")

        # 2. 만약 페이지 URL이 없으면 이미지 직접 경로 활용
        full_images = web.get("fullMatchingImages", [])
        visually_similar = web.get("visuallySimilarImages", [])

        return jsonify({
            "success": True,
            "product_url": target_url,
            "has_pages": len(pages) > 0,
            "has_images": len(full_images) > 0 or len(visually_similar) > 0,
            "pages": pages,
            "full_images": full_images
        })

    except Exception as e:
        return jsonify({"error": "서버 내부 오류", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
