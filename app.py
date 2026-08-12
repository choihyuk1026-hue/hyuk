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
            return jsonify({"error": "이미지 파일이 전송되지 않았습니다."}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({"error": "업로드된 이미지 파일 내용이 비어있습니다."}), 400

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
        
        try:
            data = response.json()
        except Exception:
            return jsonify({
                "error": "Google Vision API 응답 JSON 파싱 실패",
                "status_code": response.status_code,
                "raw_text": response.text[:200]
            }), 500

        if response.status_code != 200:
            return jsonify({
                "error": "Google Vision API 거절",
                "status_code": response.status_code,
                "details": data
            }), 500

        responses_list = data.get("responses", [])
        if not responses_list:
            return jsonify({"error": "Google API 응답이 비어있습니다."}), 500

        first_response = responses_list[0]
        if "error" in first_response:
            return jsonify({
                "error": "Google Vision API 내부 에러",
                "details": first_response["error"]
            }), 500

        web = first_response.get("webDetection", {})
        pages = web.get("pagesWithMatchingImages", [])
        
        target_url = None
        if pages:
            target_url = pages[0].get("url")

        return jsonify({
            "success": True,
            "product_url": target_url,
            "pages": pages
        })

    except Exception as e:
        return jsonify({
            "error": "파이썬 서버 처리 중 오류 발생",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
