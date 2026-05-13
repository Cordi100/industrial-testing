import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import joblib

class IndustrialAnomalyDetector:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        # 使用预训练的 ResNet50 作为特征提取器 (迁移学习核心)
        print("[PM] Loading Pre-trained ResNet50 (Transfer Learning)...")
        base_model = tf.keras.applications.ResNet50(
            include_top=False, 
            weights='imagenet', 
            pooling='avg',
            input_shape=self.input_shape
        )
        # 冻结模型，我们只提取特征，不更新权重
        base_model.trainable = False
        self.feature_extractor = base_model
        
        # 异常检测器 (只需好品数据即可训练)
        # contamination参数决定了我们对"好品"中可能存在异常的容忍度
        self.anomaly_detector = IsolationForest(contamination=0.01, random_state=42)

    def load_and_preprocess_image(self, img_path):
        """加载图片并处理成 ResNet 需要的格式"""
        img = tf.keras.utils.load_img(img_path, target_size=self.input_shape[:2])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return tf.keras.applications.resnet50.preprocess_input(img_array)

    def extract_features(self, image_paths):
        """批量提取图片的深度特征"""
        features = []
        for path in image_paths:
            img = self.load_and_preprocess_image(path)
            feat = self.feature_extractor.predict(img, verbose=0)
            features.append(feat[0])
        return np.array(features)

    def train(self, normal_data_dir):
        """
        只需传入“好品” (good) 的文件夹路径！
        工业场景绝招：单分类学习 (One-Class Learning)
        """
        print(f"[PM] Scanning for normal images in: {normal_data_dir}")
        image_paths = glob.glob(os.path.join(normal_data_dir, "*.png"))
        if not image_paths:
            raise ValueError(f"No PNG images found in {normal_data_dir}")
            
        print(f"[PM] Extracting features from {len(image_paths)} normal images...")
        X_train = self.extract_features(image_paths)
        
        print("[PM] Training Anomaly Detector (Isolation Forest)...")
        self.anomaly_detector.fit(X_train)
        print("[PM] Training Complete! Model has learned what 'Normal' looks like.")
        
    def save_model(self, save_path):
        """保存模型供后续 Web UI 使用"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.anomaly_detector, save_path)
        print(f"[PM] Anomaly detector saved to {save_path}")

    def predict(self, img_path):
        """
        预测单张图片是否为合格品
        返回: (is_pass: bool, anomaly_score: float)
        """
        feat = self.extract_features([img_path])
        # Isolation Forest: 1 是正常(Pass), -1 是异常(Fail)
        prediction = self.anomaly_detector.predict(feat)[0]
        # score_samples 越小(越负)表示越异常
        score = self.anomaly_detector.score_samples(feat)[0]
        
        is_pass = (prediction == 1)
        return is_pass, score

if __name__ == "__main__":
    # 测试代码
    # 假设你解压的 MVTec 数据在 D:\industrial_ML\data\mvtec_ad
    # 我们可以用 bottle 类别的 train/good 文件夹来训练
    train_dir = r"D:\industrial_ML\data\mvtec_ad\bottle\train\good"
    model_save_path = r"D:\industrial_ML\models\bottle_anomaly_model.pkl"
    
    if os.path.exists(train_dir):
        detector = IndustrialAnomalyDetector()
        detector.train(train_dir)
        detector.save_model(model_save_path)
        print("\n--- Model is ready for production ---")
    else:
        print(f"Path not found: {train_dir}. Please extract MVTec dataset here.")
