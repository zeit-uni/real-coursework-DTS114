import os
os.environ["APIFREE_API_KEY"] = "sk-pYf1a9sXfrbK3y053ZZ69FG1271n6"
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import json, uuid, re
from utils import get_completion, setup_llm_client

app = Flask(__name__)
CORS(app)
feedback_db = []  # list of dicts: {gif_id, rating, comment, timestamp}


def generate_default_gif():
    os.makedirs('static', exist_ok=True)
    w, h = 256, 256
    frames = []
    cx, cy = w // 2, h // 2

    # Simple pulsing circle using Pillow
    for i in range(20):
        t = i / 19.0
        # Grow then shrink
        if t <= 0.5:
            r = int(20 + (80 - 20) * (t / 0.5))
        else:
            r = int(80 - (80 - 20) * ((t - 0.5) / 0.5))

        img = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        # Draw circle via alpha mask technique (no ImageDraw import requested, so use basic pixel ops)
        # Efficient enough for small frames.
        pix = img.load()
        r2 = r * r
        for y in range(max(0, cy - r - 2), min(h, cy + r + 3)):
            dy = y - cy
            dy2 = dy * dy
            for x in range(max(0, cx - r - 2), min(w, cx + r + 3)):
                dx = x - cx
                if dx * dx + dy2 <= r2:
                    # Blue circle with mild gradient based on distance
                    dist = (dx * dx + dy2) ** 0.5
                    shade = int(160 + 80 * (1 - min(1.0, dist / max(1, r))))
                    pix[x, y] = (40, 120, shade, 255)
        frames.append(img)

    out_path = os.path.join('static', 'default.gif')
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=80, loop=0, disposal=2)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/default_gif')
def default_gif():
    return send_from_directory('static', 'default.gif')


@app.route('/generate_gif', methods=['POST'])
def generate_gif():
    # Get inputs
    image_file = request.files['image']
    action = request.form.get('action', '').strip()
    if not action:
        return jsonify({"error": "Missing action"}), 400

    # Save uploaded image temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        image_file.save(tmp.name)
        tmp_path = tmp.name

    # ---- CORRECT WAY TO CALL LLM ----
    llm_prompt = f"You are an animation engine. Given action: '{action}', generate 10 frames as JSON. Each frame: {{'dx':int, 'dy':int, 'rotate':float, 'scale':float}}. Return only JSON array."
    client, model, provider = setup_llm_client('openai/gpt-5.2')
    frames_json_str = get_completion(llm_prompt, client, model, provider, temperature=0.3)
    # ----

    # Extract JSON array from response (in case of extra text)
    json_match = re.search(r'\[.*\]', frames_json_str, re.DOTALL)
    if not json_match:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify({"error": "LLM did not return valid JSON"}), 500
    frames = json.loads(json_match.group(0))

    # Generate GIF frames using Pillow (apply dx, dy, scale; rotate optional)
    original = Image.open(tmp_path).convert('RGBA')
    w, h = original.size
    frame_images = []
    for f in frames:
        canvas = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        dx = f.get('dx', 0)
        dy = f.get('dy', 0)
        scale = f.get('scale', 1.0)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = original.resize(new_size, Image.Resampling.LANCZOS)
        paste_x = (w - new_size[0]) // 2 + dx
        paste_y = (h - new_size[1]) // 2 + dy
        canvas.paste(resized, (paste_x, paste_y), resized)
        frame_images.append(canvas)

    gif_id = str(uuid.uuid4())
    os.makedirs('static', exist_ok=True)
    gif_path = f'static/gif_{gif_id}.gif'
    frame_images[0].save(gif_path, save_all=True, append_images=frame_images[1:], duration=100, loop=0, disposal=2)

    os.unlink(tmp_path)
    return jsonify({"gif_url": f"/{gif_path}", "id": gif_id})


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    feedback_db.append({
        "gif_id": data['gif_id'],
        "rating": data['rating'],
        "comment": data['comment'],
        "timestamp": __import__('time').time()
    })
    return jsonify({"status": "ok"})


@app.route('/feedback', methods=['GET'])
def get_feedback():
    gif_id = request.args.get('gif_id')
    result = [fb for fb in feedback_db if fb['gif_id'] == gif_id]
    return jsonify(result)


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    generate_default_gif()
    app.run(host='0.0.0.0', port=5005, debug=True)