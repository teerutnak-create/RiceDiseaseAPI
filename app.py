import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
CORS(app)  # อนุญาตให้แอปมือถือเชื่อมต่อเข้ามาได้

# โหลดโมเดล
print("กำลังโหลดโมเดล กรุณารอสักครู่...")
MODEL_PATH = 'rice_stage1.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# รายชื่อ 6 คลาส เรียงลำดับตรงกับตอนเทรนบน Kaggle
classes = ["Bacterialblight", "Blast", "Brownspot", "Tungro", "Healthy", "Not_Rice"]
print("โหลดโมเดลสำเร็จ! เซิร์ฟเวอร์พร้อมทำงาน (6 คลาส + ระบบวิเคราะห์สีใบเบื้องต้น)")


def analyze_leaf_color(img_pil):
    """ประเมินธาตุอาหารเบื้องต้นจากช่วงสีใบข้าว (HSV Color Range + DGCI)"""
    hsv = np.array(img_pil.resize((224, 224)).convert('HSV'), dtype=np.float32)
    h = hsv[:, :, 0] * (360.0 / 255.0)  # Hue: 0 - 360 องศา
    s = hsv[:, :, 1] / 255.0            # Saturation: 0 - 1
    v = hsv[:, :, 2] / 255.0            # Value (ความสว่าง): 0 - 1

    # แยกช่วงสีของพิกเซลใบข้าว (ตัดพื้นหลังขาว/ดำ/เทาออก)
    valid_sv = (s >= 0.20) & (v >= 0.15) & (v <= 0.92)

    orange_brown_mask = valid_sv & (h >= 15) & (h < 38)   # โทนส้ม/น้ำตาลไหม้ (ปลายใบแห้ง)
    yellow_mask = valid_sv & (h >= 38) & (h < 68)         # โทนเหลืองอมเขียว (ใบซีด)
    green_mask = valid_sv & (h >= 68) & (h <= 165)        # โทนเขียวปกติถึงเขียวเข้ม

    all_leaf_mask = orange_brown_mask | yellow_mask | green_mask
    total_leaf_px = int(np.sum(all_leaf_mask))

    if total_leaf_px < 200:
        return None  # พิกเซลใบข้าวน้อยเกินกว่าจะประเมินสีได้

    # คำนวณสัดส่วนสี (%)
    orange_ratio = float(np.sum(orange_brown_mask) / total_leaf_px) * 100.0
    yellow_ratio = float(np.sum(yellow_mask) / total_leaf_px) * 100.0

    # คำนวณค่าดัชนีความเขียว (DGCI) จากส่วนใบสีเหลือง-เขียว
    yg_mask = yellow_mask | green_mask
    if np.sum(yg_mask) > 0:
        h_mean = float(np.mean(h[yg_mask]))
        s_mean = float(np.mean(s[yg_mask]))
        v_mean = float(np.mean(v[yg_mask]))
        dgci = ((h_mean - 60.0) / 60.0 + (1.0 - s_mean) + (1.0 - v_mean)) / 3.0
        dgci = max(0.0, min(1.0, dgci))
    else:
        dgci = 0.30

    # ตัดสินผลการประเมินเบื้องต้นตามช่วงสี
    if orange_ratio >= 18.0:
        return {
            "color_group": "ใบมีสีเหลืองส้ม/ขอบใบเริ่มไหม้",
            "nutrient_status": "ประเมินเบื้องต้น: เสี่ยงขาดธาตุโพแทสเซียม (K)",
            "fertilizer": "แนะนำเสริมปุ๋ยโพแทสเซียม (เช่น 0-0-60 อัตรา 5-10 กก./ไร่) หรือน้ำหมักผลไม้สุก",
            "dgci": round(dgci, 2),
        }
    elif dgci < 0.38 or yellow_ratio >= 45.0:
        return {
            "color_group": "ใบสีเหลืองอมเขียว (ระดับสีใบ 2)",
            "nutrient_status": "ประเมินเบื้องต้น: เสี่ยงขาดธาตุไนโตรเจน (N) ค่อนข้างมาก",
            "fertilizer": "แนะนำเติมปุ๋ยยูเรีย (46-0-0) อัตรา 10-15 กก./ไร่ หรือปุ๋ยอินทรีย์/น้ำหมักปลาที่มีไนโตรเจนสูง",
            "dgci": round(dgci, 2),
        }
    elif dgci < 0.48 or yellow_ratio >= 25.0:
        return {
            "color_group": "ใบสีเขียวอ่อน (ระดับสีใบ 3)",
            "nutrient_status": "ประเมินเบื้องต้น: เริ่มขาดธาตุไนโตรเจน (N) เล็กน้อย",
            "fertilizer": "แนะนำเสริมปุ๋ย 46-0-0 อัตรา 5-10 กก./ไร่ หรือสูตร 16-20-0 หากอยู่ในระยะแตกกอ",
            "dgci": round(dgci, 2),
        }
    elif dgci <= 0.66:
        return {
            "color_group": "ใบสีเขียวสดสมบูรณ์ (ระดับสีใบ 4)",
            "nutrient_status": "ประเมินเบื้องต้น: ธาตุอาหารสมดุลดี",
            "fertilizer": "ไม่ต้องเติมปุ๋ยไนโตรเจนเพิ่ม ดูแลรักษาและบันทึกกิจกรรมประจำวันตามปกติ",
            "dgci": round(dgci, 2),
        }
    else:
        return {
            "color_group": "ใบสีเขียวเข้มจัด (ระดับสีใบ 5)",
            "nutrient_status": "ประเมินเบื้องต้น: ไนโตรเจน (N) สูงเกินความจำเป็น",
            "fertilizer": "ควรงดใส่ปุ๋ยไนโตรเจน (46-0-0) ชั่วคราว เพื่อลดเสี่ยงโรคไหม้และเพลี้ยกระโดด และเน้นเสริมโพแทสเซียมแทน",
            "dgci": round(dgci, 2),
        }


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์รูปภาพ'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400

    try:
        # อ่านรูปภาพและปรับขนาดให้ตรงกับตอนเทรน (224x224)
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        x = tf.image.resize(np.array(img), (224, 224))
        x = tf.expand_dims(x, 0)  # (1, 224, 224, 3)

        # ให้ AI ทำนายผล
        probs = model.predict(x, verbose=0)[0]

        debug_str = " | ".join(f"{classes[i]}: {probs[i]*100:.1f}%" for i in range(len(classes)))
        print(f"🔍 ผลทำนาย -> {debug_str}")

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx] * 100)
        predicted_class = classes[top_idx]

        # ดักจับสิ่งแปลกปลอมเฉพาะเมื่อมั่นใจว่าเป็น Not_Rice >= 70%
        if predicted_class == "Not_Rice":
            if confidence >= 70.0:
                return jsonify({
                    'error': 'ตรวจพบภาพสิ่งของหรือใบไม้อื่น กรุณาถ่ายเฉพาะใบต้นข้าวให้ชัดเจนอีกครั้งครับ'
                }), 400
            else:
                rice_probs = probs[:5]
                top_idx = int(np.argmax(rice_probs))
                predicted_class = classes[top_idx]
                confidence = float(rice_probs[top_idx] * 100)

        display_names = {
            "Bacterialblight": "โรคขอบใบแห้ง (Bacterial Blight)",
            "Blast": "โรคไหม้ (Blast)",
            "Brownspot": "โรคใบจุดสีน้ำตาล (Brown Spot)",
            "Tungro": "โรคใบสีส้ม / ทังโกร (Tungro)",
            "Healthy": "ใบข้าวปกติ (ไม่พบโรค)",
        }

        advices = {
            "Bacterialblight": "พบโรคขอบใบแห้ง: แนะนำให้ลดการใช้ปุ๋ยไนโตรเจนสูง ระบายน้ำออกจากแปลง และใช้สารป้องกันกำจัดแบคทีเรีย",
            "Blast": "พบโรคไหม้: แนะนำให้ฉีดพ่นสารป้องกันกำจัดเชื้อราไตรไซคลาโซล และหลีกเลี่ยงการใส่ปุ๋ยไนโตรเจนมากเกินไป",
            "Brownspot": "พบโรคใบจุดสีน้ำตาล: แนะนำให้ใส่ปุ๋ยบำรุงดินให้สมบูรณ์ (โพแทสเซียม/อินทรียวัตถุ) หรือพ่นสารป้องกันกำจัดเชื้อรา",
            "Tungro": "พบโรคทังโกร: แนะนำให้กำจัดเพลี้ยจักจั่นสีเขียวที่เป็นพาหะ และถอนทำลายต้นที่ติดโรค",
            "Healthy": "ใบข้าวปกติ สุขภาพแข็งแรงดี ไม่พบอาการของโรค",
        }

        advice_text = advices.get(predicted_class, "ปรึกษาเจ้าหน้าที่เกษตรเพิ่มเติม")
        color_info = None

        # 🟢 ถ้าเป็นใบข้าวปกติ ให้วิเคราะห์สีใบประเมินธาตุอาหารเบื้องต้นต่อทันที
        if predicted_class == "Healthy":
            color_info = analyze_leaf_color(img)
            if color_info:
                print(f"🎨 วิเคราะห์สีใบ -> {color_info['color_group']} (DGCI={color_info['dgci']})")
                advice_text = (
                    f"{color_info['color_group']} | "
                    f"{color_info['nutrient_status']} | "
                    f"{color_info['fertilizer']}"
                )

        return jsonify({
            'disease': display_names.get(predicted_class, predicted_class),
            'raw_class': predicted_class,
            'accuracy': f"{confidence:.1f}%",
            'advice': advice_text,
            'leaf_color': color_info,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)