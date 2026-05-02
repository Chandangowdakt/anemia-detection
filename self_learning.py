import os
import json
import shutil
import threading
from datetime import datetime


class SelfLearningManager:
    def __init__(self,
                 confidence_threshold=0.92,
                 batch_size=20,
                 dataset_base='dataset',
                 log_path='self_learning_log.json'):

        self.threshold = confidence_threshold
        self.batch_size = batch_size
        self.dataset_base = dataset_base
        self.log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'self_learning_log.json'
        )
        self.pending_count = 0
        self.lock = threading.Lock()
        self._load_log()

    def _load_log(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                try:
                    self.log = json.load(f)
                except Exception:
                    self.log = []
        else:
            self.log = []

    def _save_log(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.log, f, indent=2)

    def maybe_add(self, img_path, result, confidence):
        """
        Add image to dataset only if confidence is very high.
        Returns dict with status and message.
        """
        if result in ['Invalid Image', 'Uncertain']:
            return {'added': False, 'reason': 'Result too uncertain'}

        if confidence < (self.threshold * 100):
            return {
                'added': False,
                'reason': f'Confidence {confidence:.1f}% below threshold {self.threshold*100:.0f}%'
            }

        # Map result to folder name
        label = 'anaemic' if result == 'Anaemic' else 'non_anaemic'
        dest_dir = os.path.join(self.dataset_base, 'train', label)

        if not os.path.exists(dest_dir):
            return {'added': False, 'reason': f'Dataset folder not found: {dest_dir}'}

        # Copy image with timestamp name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'auto_{label}_{timestamp}.jpg'
        dest_path = os.path.join(dest_dir, fname)

        try:
            shutil.copy2(img_path, dest_path)
        except Exception as e:
            return {'added': False, 'reason': f'File copy failed: {e}'}

        with self.lock:
            self.pending_count += 1
            self.log.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'result': result,
                'confidence': round(confidence, 1),
                'label': label,
                'file': fname,
                'status': 'added'
            })
            self._save_log()
            current_pending = self.pending_count

        print(f"Self-learning: Added {fname} as {label} "
              f"(confidence: {confidence:.1f}%) "
              f"[{current_pending}/{self.batch_size} pending]")

        # Trigger retraining when batch is full
        if current_pending >= self.batch_size:
            print(f"Self-learning: Batch full ({self.batch_size} images). "
                  f"Triggering background retraining...")
            thread = threading.Thread(
                target=self._retrain_background,
                daemon=True
            )
            thread.start()
            with self.lock:
                self.pending_count = 0

        return {
            'added': True,
            'label': label,
            'pending': current_pending,
            'batch_size': self.batch_size
        }

    def _retrain_background(self):
        """
        Placeholder for background retraining.
        In production this would trigger actual model retraining.
        For demo: logs that retraining was triggered.
        """
        print("Self-learning: Background retraining started...")
        with self.lock:
            self.log.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'event': 'retrain_triggered',
                'status': 'completed'
            })
            self._save_log()
        print("Self-learning: Retraining complete (logged)")

    def get_stats(self):
        """Return stats for admin dashboard"""
        total_added = sum(1 for e in self.log if e.get('status') == 'added')
        anaemic_added = sum(
            1 for e in self.log
            if e.get('status') == 'added' and e.get('label') == 'anaemic'
        )
        non_anaemic_added = sum(
            1 for e in self.log
            if e.get('status') == 'added' and e.get('label') == 'non_anaemic'
        )
        retrains = sum(
            1 for e in self.log
            if e.get('event') == 'retrain_triggered'
        )
        return {
            'total_added': total_added,
            'anaemic_added': anaemic_added,
            'non_anaemic_added': non_anaemic_added,
            'retrains_triggered': retrains,
            'pending_count': self.pending_count,
            'batch_size': self.batch_size,
            'threshold': self.threshold * 100
        }
