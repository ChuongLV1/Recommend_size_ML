# import pandas as pd
# import numpy as np
# import joblib
# from catboost import CatBoostClassifier

# # Load model và encoder đã lưu
# model = CatBoostClassifier()
# model.load_model("E:/Capstone_AIFSHOP/Recommend_size/catboost_pipeline_model_final.cbm")
# le = joblib.load("E:/Capstone_AIFSHOP/Recommend_size/catboost_pipeline_label_encoder_final.pkl")

# # Hàm dự đoán size quần áo
# def predict_size(height, weight, age, gender):
#     bmi = weight / ((height / 100) ** 2)
#     input_df = pd.DataFrame([[height, weight, age, gender, bmi]], 
#                              columns=['height', 'weight', 'age', 'gender', 'BMI'])
#     pred_encoded = model.predict(input_df)[0]
#     predicted_size = le.inverse_transform([int(pred_encoded)])[0]
#     return predicted_size

# # Nhập input từ người dùng
# if __name__ == "__main__":
#     try:
#         height = float(input("Nhập chiều cao (cm): "))
#         weight = float(input("Nhập cân nặng (kg): "))
#         age = int(input("Nhập tuổi: "))
#         gender = input("Nhập giới tính (Nam/Nữ): ").strip().lower()

#         if gender not in ['nam', 'nữ']:
#             print("⚠️ Giới tính chỉ được nhập: Nam hoặc Nữ.")
#         else:
#             result = predict_size(height, weight, age, gender)
#             print(f"🎯 Recommended size: {result}")
#     except Exception as e:
#         print("Lỗi nhập dữ liệu:", e)



### predict_catboostmodel.py sử dụng demo trên web
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier

app = Flask(__name__)

# Load model và encoder
model = CatBoostClassifier()
model.load_model("catboost_pipeline_model_final.cbm")
le = joblib.load("catboost_pipeline_label_encoder_final.pkl")

def predict_size(height, weight, age, gender):
    bmi = weight / ((height / 100) ** 2)
    input_df = pd.DataFrame([[height, weight, age, gender, bmi]], 
                             columns=['height', 'weight', 'age', 'gender', 'BMI'])
    pred_encoded = model.predict(input_df)[0]
    predicted_size = le.inverse_transform([int(pred_encoded)])[0]
    return predicted_size

@app.route('/')
def home():
    return render_template('ui.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    height = float(data['height'])
    weight = float(data['weight'])
    age = int(data['age'])
    gender = data['gender'].lower()

    result = predict_size(height, weight, age, gender)
    return render_template('ui.html', prediction_text=f'Recommended Size: {result}')

if __name__ == "__main__":
    app.run(debug=True)
