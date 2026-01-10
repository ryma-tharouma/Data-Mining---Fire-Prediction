import numpy as np

class ClassificationMetrics:
    """Memory-efficient classification metrics for large datasets using chunked processing"""
    
    def __init__(self, y_true, y_pred, labels=None, chunk_size=100000):
        """
        Initialize metrics calculator with chunked processing
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of labels to include (optional, auto-detected if None)
            chunk_size: Number of samples to process at once
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.chunk_size = chunk_size
        
        if labels is None:
            # Process in chunks to find unique labels
            unique_true = set()
            unique_pred = set()
            for i in range(0, len(self.y_true), chunk_size):
                end = min(i + chunk_size, len(self.y_true))
                unique_true.update(np.unique(self.y_true[i:end]))
                unique_pred.update(np.unique(self.y_pred[i:end]))
            self.labels = np.array(sorted(unique_true | unique_pred))
        else:
            self.labels = np.array(labels)
        
        self.n_classes = len(self.labels)
        self._label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self._confusion_matrix = None
        self._n_samples = len(self.y_true)
    
    def confusion_matrix(self, verbose=False):
        """Compute confusion matrix using chunked processing"""
        if self._confusion_matrix is not None:
            return self._confusion_matrix
        
        cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        
        # Process in chunks to avoid memory issues
        n_chunks = (self._n_samples + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, self._n_samples)
            
            # Get chunk
            y_true_chunk = self.y_true[start:end]
            y_pred_chunk = self.y_pred[start:end]
            
            # Map to indices
            y_true_idx = np.array([self._label_to_idx[y] for y in y_true_chunk])
            y_pred_idx = np.array([self._label_to_idx[y] for y in y_pred_chunk])
            
            # Accumulate
            np.add.at(cm, (y_true_idx, y_pred_idx), 1)
            
            if verbose and (chunk_idx + 1) % 10 == 0:
                print(f"Processed {end:,}/{self._n_samples:,} samples "
                      f"({100*end/self._n_samples:.1f}%)")
        
        self._confusion_matrix = cm
        if verbose:
            print(f"Confusion matrix computed for {self._n_samples:,} samples")
        return cm
    
    def accuracy(self):
        """Calculate accuracy using chunked comparison"""
        correct = 0
        
        for i in range(0, self._n_samples, self.chunk_size):
            end = min(i + self.chunk_size, self._n_samples)
            correct += np.sum(self.y_true[i:end] == self.y_pred[i:end])
        
        return correct / self._n_samples
    
    def _safe_divide(self, numerator, denominator, zero_division=0):
        """Vectorized safe division"""
        result = np.full_like(numerator, zero_division, dtype=np.float64)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        return result
    
    def precision(self, average='macro', zero_division=0):
        """
        Calculate precision - confusion matrix computed once, rest is fast
        
        Args:
            average: 'macro', 'micro', 'weighted', or None (per-class)
            zero_division: Value to return when denominator is zero
        """
        cm = self.confusion_matrix()
        
        # Vectorized: TP for each class (diagonal)
        tp = np.diag(cm).astype(np.float64)
        # Vectorized: Predicted positives (column sum)
        pred_pos = np.sum(cm, axis=0).astype(np.float64)
        
        # Vectorized safe division
        precision = self._safe_divide(tp, pred_pos, zero_division)
        
        if average == 'macro':
            return np.mean(precision)
        elif average == 'micro':
            total_tp = np.sum(tp)
            total_pred = np.sum(pred_pos)
            return total_tp / total_pred if total_pred > 0 else zero_division
        elif average == 'weighted':
            weights = np.sum(cm, axis=1).astype(np.float64)
            return np.average(precision, weights=weights)
        else:
            return precision
    
    def recall(self, average='macro', zero_division=0):
        """
        Calculate recall - confusion matrix computed once, rest is fast
        
        Args:
            average: 'macro', 'micro', 'weighted', or None (per-class)
            zero_division: Value to return when denominator is zero
        """
        cm = self.confusion_matrix()
        
        # Vectorized: TP for each class (diagonal)
        tp = np.diag(cm).astype(np.float64)
        # Vectorized: Actual positives (row sum)
        actual_pos = np.sum(cm, axis=1).astype(np.float64)
        
        # Vectorized safe division
        recall = self._safe_divide(tp, actual_pos, zero_division)
        
        if average == 'macro':
            return np.mean(recall)
        elif average == 'micro':
            total_tp = np.sum(tp)
            total_actual = np.sum(actual_pos)
            return total_tp / total_actual if total_actual > 0 else zero_division
        elif average == 'weighted':
            weights = actual_pos
            return np.average(recall, weights=weights)
        else:
            return recall
    
    def f1_score(self, average='macro', zero_division=0):
        """
        Calculate F1 score using vectorized operations
        
        Args:
            average: 'macro', 'micro', 'weighted', or None (per-class)
            zero_division: Value to return when denominator is zero
        """
        if average in ['macro', 'weighted']:
            prec = self.precision(average=None, zero_division=zero_division)
            rec = self.recall(average=None, zero_division=zero_division)
            
            # Vectorized F1 computation
            denom = prec + rec
            f1 = self._safe_divide(2 * prec * rec, denom, zero_division)
            
            if average == 'macro':
                return np.mean(f1)
            else:  # weighted
                cm = self.confusion_matrix()
                weights = np.sum(cm, axis=1).astype(np.float64)
                return np.average(f1, weights=weights)
        
        elif average == 'micro':
            prec = self.precision(average='micro', zero_division=zero_division)
            rec = self.recall(average='micro', zero_division=zero_division)
            denom = prec + rec
            if denom > 0:
                return 2 * (prec * rec) / denom
            return zero_division
        
        else:  # Return per-class
            prec = self.precision(average=None, zero_division=zero_division)
            rec = self.recall(average=None, zero_division=zero_division)
            
            # Vectorized F1 computation
            denom = prec + rec
            f1 = self._safe_divide(2 * prec * rec, denom, zero_division)
            return f1
    
    def classification_report(self, digits=3):
        """Generate a detailed classification report"""
        cm = self.confusion_matrix()
        support = np.sum(cm, axis=1)
        
        # Vectorized metric computation
        precision = self.precision(average=None)
        recall = self.recall(average=None)
        f1 = self.f1_score(average=None)
        
        # Header
        headers = ['precision', 'recall', 'f1-score', 'support']
        header_fmt = '{:>12} ' * len(headers)
        row_fmt = '{:>12.{digits}f} ' * 3 + '{:>12}'
        
        print()
        print(' ' * 12 + header_fmt.format(*headers))
        print()
        
        # Per-class metrics
        for i, label in enumerate(self.labels):
            values = [precision[i], recall[i], f1[i], support[i]]
            print(f'{str(label):>12}' + ' ' + row_fmt.format(*values, digits=digits))
        
        print()
        
        # Averages
        total_support = np.sum(support)
        print(f'{"accuracy":>12}' + ' ' + '{:>12}'.format('') * 2 + 
              f'{self.accuracy():.{digits}f}' + ' ' + f'{total_support:>12}')
        print(f'{"macro avg":>12} ' + row_fmt.format(
            self.precision(average='macro'),
            self.recall(average='macro'),
            self.f1_score(average='macro'),
            total_support,
            digits=digits
        ))
        print(f'{"weighted avg":>12} ' + row_fmt.format(
            self.precision(average='weighted'),
            self.recall(average='weighted'),
            self.f1_score(average='weighted'),
            total_support,
            digits=digits
        ))
        print()
    
    def print_confusion_matrix(self):
        """Pretty print the confusion matrix"""
        cm = self.confusion_matrix()
        
        # Calculate column widths
        max_label_len = max(len(str(label)) for label in self.labels)
        max_val = np.max(cm)
        max_val_len = len(str(max_val))
        col_width = max(max_label_len, max_val_len) + 2
        
        print(f"\nConfusion Matrix ({self._n_samples:,} samples):")
        print("=" * (col_width * (self.n_classes + 1) + 10))
        
        # Header row
        print(f"{'True/Pred':>{col_width}}", end='')
        for label in self.labels:
            print(f"{str(label):>{col_width}}", end='')
        print()
        print("-" * (col_width * (self.n_classes + 1) + 10))
        
        # Data rows
        for i, label in enumerate(self.labels):
            print(f"{str(label):>{col_width}}", end='')
            row_str = ''.join([f"{cm[i, j]:>{col_width}}" for j in range(self.n_classes)])
            print(row_str)
        print("=" * (col_width * (self.n_classes + 1) + 10))
        print()
    
    def get_all_metrics(self):
        """Return all metrics as a dictionary"""
        return {
            'accuracy': self.accuracy(),
            'precision_macro': self.precision(average='macro'),
            'precision_micro': self.precision(average='micro'),
            'precision_weighted': self.precision(average='weighted'),
            'recall_macro': self.recall(average='macro'),
            'recall_micro': self.recall(average='micro'),
            'recall_weighted': self.recall(average='weighted'),
            'f1_macro': self.f1_score(average='macro'),
            'f1_micro': self.f1_score(average='micro'),
            'f1_weighted': self.f1_score(average='weighted'),
            'confusion_matrix': self.confusion_matrix()
        }


def evaluate_predictions(y_true, y_pred, labels=None, chunk_size=100000, verbose=True):
    """
    Memory-efficient evaluation for large datasets
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels (optional)
        chunk_size: Samples to process at once
        verbose: Print detailed report
    
    Returns:
        ClassificationMetrics object
    """
    metrics = ClassificationMetrics(y_true, y_pred, labels, chunk_size=chunk_size)
    
    if verbose:
        print(f"\nProcessing {len(y_true):,} samples with chunk_size={chunk_size:,}")
        metrics.print_confusion_matrix()
        metrics.classification_report()
    
    return metrics


# Performance test with large datasets
if __name__ == "__main__":
    import time
    
    print("="*70)
    print("=== Memory-Efficient Metrics for LARGE Datasets ===")
    print("="*70)
    
    # Test 1: 1 million samples
    print("\n📊 Test 1: 1 Million Samples")
    print("-" * 70)
    np.random.seed(42)
    
    n_samples = 1_000_000
    n_classes = 10
    
    print(f"Generating {n_samples:,} samples...")
    start_gen = time.time()
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add 20% errors
    error_idx = np.random.choice(n_samples, n_samples // 5, replace=False)
    y_pred[error_idx] = np.random.randint(0, n_classes, len(error_idx))
    print(f"Generation time: {time.time() - start_gen:.2f}s")
    
    print(f"\nComputing metrics with chunk_size=100,000...")
    start = time.time()
    metrics = ClassificationMetrics(y_true, y_pred, chunk_size=100000)
    
    # Compute metrics
    acc = metrics.accuracy()
    cm = metrics.confusion_matrix(verbose=True)
    prec = metrics.precision(average='macro')
    rec = metrics.recall(average='macro')
    f1 = metrics.f1_score(average='macro')
    
    elapsed = time.time() - start
    
    print(f"\n✅ Results:")
    print(f"   Total computation time: {elapsed:.2f}s")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision (macro): {prec:.4f}")
    print(f"   Recall (macro): {rec:.4f}")
    print(f"   F1-score (macro): {f1:.4f}")
    print(f"   Memory-efficient: Processed in chunks ✓")
    
    # Test 2: 5 million samples (if you dare!)
    print("\n" + "="*70)
    print("📊 Test 2: 5 Million Samples (Stress Test)")
    print("-" * 70)
    
    n_samples_large = 5_000_000
    print(f"Generating {n_samples_large:,} samples...")
    start_gen = time.time()
    y_true_large = np.random.randint(0, 5, n_samples_large)
    y_pred_large = y_true_large.copy()
    error_idx = np.random.choice(n_samples_large, n_samples_large // 10, replace=False)
    y_pred_large[error_idx] = np.random.randint(0, 5, len(error_idx))
    print(f"Generation time: {time.time() - start_gen:.2f}s")
    
    print(f"\nComputing metrics with chunk_size=200,000...")
    start = time.time()
    metrics_large = ClassificationMetrics(y_true_large, y_pred_large, chunk_size=200000)
    acc_large = metrics_large.accuracy()
    cm_large = metrics_large.confusion_matrix(verbose=True)
    f1_large = metrics_large.f1_score(average='macro')
    elapsed_large = time.time() - start
    
    print(f"\n✅ Results:")
    print(f"   Total computation time: {elapsed_large:.2f}s")
    print(f"   Accuracy: {acc_large:.4f}")
    print(f"   F1-score (macro): {f1_large:.4f}")
    print(f"   Throughput: {n_samples_large/elapsed_large:,.0f} samples/sec")
    
    # Example with detailed report
    print("\n" + "="*70)
    print("📊 Test 3: Small Example with Full Report")
    print("-" * 70)
    y_true_small = np.random.randint(0, 3, 1000)
    y_pred_small = y_true_small.copy()
    error_idx = np.random.choice(1000, 200, replace=False)
    y_pred_small[error_idx] = np.random.randint(0, 3, 200)
    
    evaluate_predictions(y_true_small, y_pred_small, labels=[0, 1, 2], chunk_size=250)