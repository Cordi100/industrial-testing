"""
PatchCore - 工业级异常检测算法 (算法升级版)
原理: 提取图片每个空间位置的 patch 特征 → 建立"正常 patch 记忆库"
      推理时: 找每个 patch 最近邻 → 最大距离即为异常分数

参考: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022
"""

import os
import glob
import numpy as np
import tensorflow as tf
import joblib
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score


class PatchCore:
    def __init__(self, input_shape=(224, 224, 3), subsample_ratio=0.1):
        """
        subsample_ratio: 从记忆库中随机抽样的比例 (避免记忆库过大)
        """
        self.input_shape = input_shape
        self.subsample_ratio = subsample_ratio
        self.memory_bank = None        # 正常 patch 特征库
        self.threshold = None          # 判定阈值

        print("[PatchCore] Building feature extractor from ResNet50 intermediate layers...")

        # 使用预训练的 ResNet50
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )

        # 核心: 提取中间层的空间特征图 (28x28 and 14x14)
        # layer2 输出: (batch, 28, 28, 512)
        # layer3 输出: (batch, 14, 14, 1024)
        out_layer2 = base.get_layer("conv3_block4_out").output
        out_layer3 = base.get_layer("conv4_block6_out").output

        self.feature_extractor = tf.keras.Model(
            inputs=base.input,
            outputs=[out_layer2, out_layer3]
        )
        self.feature_extractor.trainable = False
        print("[PatchCore] Feature extractor ready (using layer2 + layer3 spatial features).")

    def _preprocess(self, img_path):
        """加载并预处理图片"""
        img = tf.keras.utils.load_img(img_path, target_size=self.input_shape[:2])
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, 0)
        return tf.keras.applications.resnet50.preprocess_input(arr)

    def _extract_patch_features(self, img_path):
        """
        提取单张图片的 patch 特征。
        将两个尺度的特征图对齐后拼接 → 每个空间位置得到一个高维向量
        """
        x = self._preprocess(img_path)
        feat2, feat3 = self.feature_extractor.predict(x, verbose=0)

        # feat2: (1, 28, 28, 512)
        # feat3: (1, 14, 14, 1024) → upsample 到 28x28
        h, w = feat2.shape[1], feat2.shape[2]
        feat3_up = tf.image.resize(feat3, (h, w)).numpy()

        # 拼接: (1, 28, 28, 1536)
        combined = np.concatenate([feat2, feat3_up], axis=-1)

        # reshape 成 (h*w, C) → 每行是一个 patch 的特征向量
        patch_features = combined[0].reshape(h * w, -1)
        return patch_features, h, w

    def build_memory_bank(self, train_good_dir):
        """
        只用好品 (good) 图片构建记忆库。
        """
        print(f"\n[PatchCore] Scanning: {train_good_dir}")
        image_paths = glob.glob(os.path.join(train_good_dir, "*.png"))
        if not image_paths:
            raise ValueError(f"No .png images found in {train_good_dir}")

        print(f"[PatchCore] Extracting patch features from {len(image_paths)} good images...")
        all_patches = []

        for i, path in enumerate(image_paths):
            patches, _, _ = self._extract_patch_features(path)
            all_patches.append(patches)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(image_paths)} images...")

        # 合并所有图片的 patch 特征: (N_images * H*W, 1536)
        all_patches = np.concatenate(all_patches, axis=0)
        print(f"[PatchCore] Total patches before subsampling: {len(all_patches)}")

        # 随机下采样 — 降低内存占用 (工业 PM 关注点: 资源成本)
        n_samples = max(1, int(len(all_patches) * self.subsample_ratio))
        idx = np.random.choice(len(all_patches), n_samples, replace=False)
        self.memory_bank = all_patches[idx]
        print(f"[PatchCore] Memory bank built: {len(self.memory_bank)} patch vectors (subsampled to {self.subsample_ratio*100:.0f}%)")

    def _compute_anomaly_score(self, img_path, query_batch_size=64):
        """
        对单张图片计算异常分数 + 异常热力图。
        使用分批计算 (batched) 避免 OOM: 每次只处理少量 query patches 与整个记忆库的距离。
        """
        patch_features, h, w = self._extract_patch_features(img_path)
        # patch_features: (H*W, C)

        min_distances = np.empty(len(patch_features), dtype=np.float32)

        for start in range(0, len(patch_features), query_batch_size):
            end = min(start + query_batch_size, len(patch_features))
            # batch: (B, 1, C) - memory: (1, M, C)
            batch = patch_features[start:end]  # (B, C)
            # Efficient: use (a-b)^2 = a^2 - 2ab + b^2
            # (B, M) distances
            a2 = (batch ** 2).sum(axis=1, keepdims=True)          # (B, 1)
            b2 = (self.memory_bank ** 2).sum(axis=1, keepdims=True).T  # (1, M)
            ab = batch @ self.memory_bank.T                         # (B, M)
            dist = np.sqrt(np.maximum(a2 - 2 * ab + b2, 0))       # (B, M)
            min_distances[start:end] = dist.min(axis=1)

        # 异常分数 = 所有 patch 中最大的最近邻距离
        anomaly_score = float(min_distances.max())

        # 热力图: 把 min_distances 重排成 (h, w) 的空间图
        heatmap = min_distances.reshape(h, w)
        heatmap = gaussian_filter(heatmap, sigma=4)

        return anomaly_score, heatmap

    def determine_threshold(self, train_good_dir):
        """
        在好品上跑一遍，取 99% 分位数作为判定阈值
        """
        print("\n[PatchCore] Determining threshold using training good images...")
        image_paths = glob.glob(os.path.join(train_good_dir, "*.png"))
        scores = []
        for path in image_paths:
            s, _ = self._compute_anomaly_score(path)
            scores.append(s)
        self.threshold = np.percentile(scores, 99)
        print(f"[PatchCore] Threshold set to: {self.threshold:.4f} (99th percentile of good scores)")

    def train(self, train_good_dir):
        """一键训练: 建记忆库 + 设定阈值"""
        self.build_memory_bank(train_good_dir)
        self.determine_threshold(train_good_dir)
        print("\n[PatchCore] Training complete!")

    def predict(self, img_path):
        """
        推理单张图片
        Returns: (is_pass: bool, score: float, heatmap: np.ndarray, confidence_pct: float)
        """
        if self.memory_bank is None or self.threshold is None:
            raise RuntimeError("Model not trained. Call train() first.")

        score, heatmap = self._compute_anomaly_score(img_path)
        is_pass = (score <= self.threshold)

        # 置信度: 距离阈值有多远 (归一化到 0-100%)
        confidence_pct = max(0, min(100, (1 - score / (self.threshold * 2)) * 100))

        return is_pass, score, heatmap, confidence_pct

    def save(self, save_path):
        """保存模型到磁盘"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state = {
            "memory_bank": self.memory_bank,
            "threshold": self.threshold,
            "subsample_ratio": self.subsample_ratio
        }
        joblib.dump(state, save_path)
        print(f"[PatchCore] Model saved to {save_path}")

    def load(self, save_path):
        """从磁盘加载模型"""
        state = joblib.load(save_path)
        self.memory_bank = state["memory_bank"]
        self.threshold = state["threshold"]
        self.subsample_ratio = state["subsample_ratio"]
        print(f"[PatchCore] Model loaded. Memory bank: {len(self.memory_bank)} vectors, threshold: {self.threshold:.4f}")

    def evaluate(self, test_dir):
        """在测试集上计算 Accuracy / FAR / FRR / AUROC"""
        print(f"\n[PatchCore] Evaluating on test set: {test_dir}")
        y_true = []
        y_scores = []

        # Good images
        good_dir = os.path.join(test_dir, "good")
        for img in glob.glob(os.path.join(good_dir, "*.png")):
            score, _ = self._compute_anomaly_score(img)
            y_true.append(0)      # 0 = normal
            y_scores.append(score)

        # Defect images
        for folder in os.listdir(test_dir):
            if folder == "good":
                continue
            defect_dir = os.path.join(test_dir, folder)
            if os.path.isdir(defect_dir):
                for img in glob.glob(os.path.join(defect_dir, "*.png")):
                    score, _ = self._compute_anomaly_score(img)
                    y_true.append(1)  # 1 = defect
                    y_scores.append(score)

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = (y_scores > self.threshold).astype(int)

        # Metrics
        total = len(y_true)
        total_good = (y_true == 0).sum()
        total_defect = (y_true == 1).sum()

        true_pass = ((y_true == 0) & (y_pred == 0)).sum()
        false_fail = ((y_true == 0) & (y_pred == 1)).sum()   # 过杀 FAR
        false_pass = ((y_true == 1) & (y_pred == 0)).sum()   # 漏杀 FRR
        true_fail  = ((y_true == 1) & (y_pred == 1)).sum()

        accuracy = (true_pass + true_fail) / total * 100
        far       = false_fail / total_good * 100 if total_good > 0 else 0
        frr       = false_pass / total_defect * 100 if total_defect > 0 else 0
        auroc     = roc_auc_score(y_true, y_scores) * 100

        print("\n" + "="*50)
        print("PatchCore - AI PM Evaluation Report")
        print("="*50)
        print(f"Total Test Images : {total}  (Good: {total_good} | Defect: {total_defect})")
        print("-"*50)
        print(f"[+] Accuracy       : {accuracy:.2f}%")
        print(f"[+] AUROC          : {auroc:.2f}%  <- key metric in academia")
        print(f"[!] FAR (False Reject - over-kill): {far:.2f}%")
        print(f"[X] FRR (False Accept - missed!)  : {frr:.2f}%  <- must be near 0")
        print("-"*50)
        print(f"Confusion Matrix:")
        print(f"  {'':20s}  Pred PASS   Pred FAIL")
        print(f"  {'Actual PASS (Good)':20s}  {true_pass:8d}   {false_fail:8d}")
        print(f"  {'Actual FAIL (Defect)':20s}  {false_pass:8d}   {true_fail:8d}")
        print("="*50)

        return {"accuracy": accuracy, "auroc": auroc, "far": far, "frr": frr}


# ── 主程序: 一键训练 + 评估 ──────────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN_DIR = r"D:\industrial_ML\data\mvtec_ad\bottle\train\good"
    TEST_DIR  = r"D:\industrial_ML\data\mvtec_ad\bottle\test"
    MODEL_PATH = r"D:\industrial_ML\models\patchcore_bottle.pkl"

    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Training data not found at {TRAIN_DIR}")
        print("Please extract MVTec AD bottle dataset there first.")
        exit(1)

    model = PatchCore(subsample_ratio=0.1)

    # Check if already trained
    if os.path.exists(MODEL_PATH):
        print(f"[PatchCore] Found existing model at {MODEL_PATH}, loading...")
        model.load(MODEL_PATH)
    else:
        model.train(TRAIN_DIR)
        model.save(MODEL_PATH)

    # Evaluate
    results = model.evaluate(TEST_DIR)
