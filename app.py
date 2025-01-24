from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# 加载模型
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    # 获取输入数据
    data = request.json.get("data", [])
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # 执行预测
    try:
        predictions = model.predict(data).tolist()
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
