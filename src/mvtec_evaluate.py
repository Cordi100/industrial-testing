import os
import glob
from sklearn.metrics import confusion_matrix
import joblib
from .mvtec_anomaly import IndustrialAnomalyDetector

def evaluate_model(model_path, test_dir):
    """
    评估模型在测试集上的表现，输出 PM 关心的核心指标：FAR 和 FRR
    """
    print(f"[PM] Loading Anomaly Detector from {model_path}...")
    detector = IndustrialAnomalyDetector()
    detector.anomaly_detector = joblib.load(model_path)
    
    y_true = []
    y_pred = []
    
    print("[PM] Running inference on test set...")
    
    # 1. 测正常品 (Good)
    good_dir = os.path.join(test_dir, "good")
    if os.path.exists(good_dir):
        good_imgs = glob.glob(os.path.join(good_dir, "*.png"))
        for img in good_imgs:
            y_true.append("PASS")  # 真实是好品
            is_pass, _ = detector.predict(img)
            y_pred.append("PASS" if is_pass else "FAIL")
            
    # 2. 测不良品 (Defects - 会有多个不同的子文件夹)
    for folder in os.listdir(test_dir):
        if folder == "good":
            continue
        defect_dir = os.path.join(test_dir, folder)
        if os.path.isdir(defect_dir):
            defect_imgs = glob.glob(os.path.join(defect_dir, "*.png"))
            for img in defect_imgs:
                y_true.append("FAIL")  # 真实是不良品
                is_pass, _ = detector.predict(img)
                y_pred.append("PASS" if is_pass else "FAIL")

    # 计算混淆矩阵
    # labels=["PASS", "FAIL"] 意味着:
    # matrix[0][0]: True PASS (真阴性 - 正常品被识别为正常)
    # matrix[0][1]: False FAIL (假阳性 - 过杀 FAR! 正常品被当成废品)
    # matrix[1][0]: False PASS (假阴性 - 漏杀 FRR! 废品被当成正常品，严重客诉！)
    # matrix[1][1]: True FAIL (真阳性 - 废品被正确拦截)
    cm = confusion_matrix(y_true, y_pred, labels=["PASS", "FAIL"])
    
    true_pass = cm[0][0]
    false_fail = cm[0][1] # 过杀
    false_pass = cm[1][0] # 漏杀
    true_fail = cm[1][1]
    
    total_good = true_pass + false_fail
    total_bad = true_fail + false_pass
    
    far = (false_fail / total_good) * 100 if total_good > 0 else 0
    frr = (false_pass / total_bad) * 100 if total_bad > 0 else 0
    accuracy = ((true_pass + true_fail) / (total_good + total_bad)) * 100
    
    print("\n" + "="*40)
    print("AI PM Evaluation Report")
    print("="*40)
    print(f"Total Test Images: {total_good + total_bad}")
    print(f"Total Good: {total_good} | Total Defect: {total_bad}")
    print("-"*40)
    print(f"[+] Accuracy: {accuracy:.2f}%")
    print(f"[!] FAR (False Reject - 过杀率): {far:.2f}%  <- 影响复核成本")
    print(f"[X] FRR (False Accept - 漏杀率): {frr:.2f}%  <- 影响客诉(必须极低)")
    print("="*40)
    print("Confusion Matrix:")
    print(f"               Predicted PASS | Predicted FAIL")
    print(f"Actual PASS  | {true_pass:12d} | {false_fail:12d}")
    print(f"Actual FAIL  | {false_pass:12d} | {true_fail:12d}")
    print("="*40)

if __name__ == "__main__":
    test_dir = r"D:\industrial_ML\data\mvtec_ad\bottle\test"
    model_path = r"D:\industrial_ML\models\bottle_anomaly_model.pkl"
    evaluate_model(model_path, test_dir)
