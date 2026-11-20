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
print("โหลดโมเดลสำเร็จ! เซิร์ฟเวอร์พร้อมทำงาน (6 คลาส)")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์รูปภาพ'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400

    try:
        # อ่านรูปภาพและปรับขนาดด้วย tf.image.resize ให้ตรงกับตอนเทรน (224x224)
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        x = tf.image.resize(np.array(img), (224, 224))
        x = tf.expand_dims(x, 0)  # (1, 224, 224, 3)

        # ให้ AI ทำนายผล
        probs = model.predict(x, verbose=0)[0]

        # ปริ้นต์ดูค่า % ใน Terminal เผื่อเช็ก
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
                # ถ้าไม่ถึง 70% ให้เลือกคลาสใบข้าว (index 0-4) ที่คะแนนสูงสุดแทน
                rice_probs = probs[:5]
                top_idx = int(np.argmax(rice_probs))
                predicted_class = classes[top_idx]
                confidence = float(rice_probs[top_idx] * 100)

        # ชื่อแสดงผลและคำแนะนำครบทั้ง 5 คลาสใบข้าว
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
            "Healthy": "ใบข้าวปกติ สุขภาพแข็งแรงดี ไม่พบอาการของโรค แนะนำให้ดูแลรักษาและบันทึกกิจกรรมประจำวันตามปกติครับ",
        }

        return jsonify({
            'disease': display_names.get(predicted_class, predicted_class),
            'raw_class': predicted_class,
            'accuracy': f"{confidence:.1f}%",
            'advice': advices.get(predicted_class, "ปรึกษาเจ้าหน้าที่เกษตรเพิ่มเติม")
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # รันเซิร์ฟเวอร์ (0.0.0.0 คือให้มือถือในวง WiFi เดียวกันเข้าถึงได้)
    app.run(host='0.0.0.0', port=5000, debug=True)