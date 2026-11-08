import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # อนุญาตให้แอปมือถือเชื่อมต่อเข้ามาได้

# โหลดโมเดล
print("กำลังโหลดโมเดล กรุณารอสักครู่...")
MODEL_PATH = 'rice_stage1.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# 🟢 [แก้จุดที่ 1] อัปเดตรายชื่อคลาสให้มี 5 คลาสตามที่เทรนมาใหม่
classes = ["Bacterialblight", "Blast", "Brownspot", "Tungro", "Not_Rice"]
print("โหลดโมเดลสำเร็จ! เซิร์ฟเวอร์พร้อมทำงาน")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์รูปภาพ'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400

    try:
        # อ่านรูปภาพและปรับขนาดให้ตรงกับที่โมเดลเทรนมา (224x224)
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_resized = img.resize((224, 224))
        
        # แปลงรูปลง Array
        x = np.array(img_resized)
        x = tf.expand_dims(x, 0) # เพิ่มมิติให้เป็น (1, 224, 224, 3)

        # ให้ AI ทำนายผล
        probs = model.predict(x, verbose=0)[0]
        top_idx = np.argmax(probs)
        confidence = probs[top_idx] * 100
        
        predicted_class = classes[top_idx]

        # 🟢 [แก้จุดที่ 2] ดักจับสิ่งแปลกปลอม (ไม่ใช่ใบข้าว)
        if predicted_class == "Not_Rice":
            return jsonify({
                'error': 'ตรวจพบภาพสิ่งของหรือใบไม้อื่น กรุณาถ่ายเฉพาะใบต้นข้าวให้ชัดเจนอีกครั้งครับ'
            }), 400

        # คำแนะนำอิงตามโรค
        advices = {
            "Bacterialblight": "พบโรคขอบใบแห้ง: แนะนำให้ลดการใช้ปุ๋ยไนโตรเจนสูง และใช้สารป้องกันกำจัดแบคทีเรีย",
            "Blast": "พบโรคไหม้: แนะนำให้ฉีดพ่นสารป้องกันกำจัดเชื้อราไตรไซคลาโซล",
            "Brownspot": "พบโรคใบจุดสีน้ำตาล: แนะนำให้ใส่ปุ๋ยบำรุงดินให้สมบูรณ์ หรือพ่นสารป้องกันกำจัดเชื้อรา",
            "Tungro": "พบโรคทังโกร: แนะนำให้กำจัดแมลงกระโดดสีเขียวที่เป็นพาหะ และทำลายต้นที่ติดโรค"
        }

        return jsonify({
            'disease': predicted_class,
            'accuracy': f"{confidence:.1f}%",
            'advice': advices.get(predicted_class, "ปรึกษาเจ้าหน้าที่เกษตรเพิ่มเติม")
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # รันเซิร์ฟเวอร์ (0.0.0.0 คือให้มือถือในวง WiFi เดียวกันเข้าถึงได้)
    app.run(host='0.0.0.0', port=5000, debug=True)