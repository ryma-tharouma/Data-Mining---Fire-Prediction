import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class MyDecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

class MyDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_bins=100, chunk_size=1000, verbose=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_bins = n_bins
        self.chunk_size = chunk_size
        self.root = None
        self.verbose = verbose
        self._node_count = 0
        self._total_samples = 0

    def _chunked_entropy(self, y):
        counter = collections.Counter()
        total = 0
        for i in range(0, len(y), self.chunk_size):
            y_chunk = y[i:i+self.chunk_size]
            uniques, counts = np.unique(y_chunk, return_counts=True)
            counter.update(dict(zip(uniques, counts)))
            total += len(y_chunk)
        p = np.array(list(counter.values()), dtype=np.float64) / total
        return -np.sum(p * np.log2(p + 1e-12))

    def _chunked_information_gain(self, y, X_col, threshold):
        total_counts = collections.Counter()
        left_counts = collections.Counter()
        right_counts = collections.Counter()
        n, n_left, n_right = 0, 0, 0
        for i in range(0, len(y), self.chunk_size):
            y_chunk = y[i:i+self.chunk_size]
            x_chunk = X_col[i:i+self.chunk_size]
            mask_left = x_chunk <= threshold
            left_chunk = y_chunk[mask_left]
            right_chunk = y_chunk[~mask_left]
            left_counts.update(collections.Counter(left_chunk))
            right_counts.update(collections.Counter(right_chunk))
            n_left += left_chunk.size
            n_right += right_chunk.size
            uniques, counts = np.unique(y_chunk, return_counts=True)
            total_counts.update(dict(zip(uniques, counts)))
            n += len(y_chunk)
        def entropy_from_counts(counts, total_samples):
            if total_samples == 0:
                return 0.0
            probs = np.array(list(counts.values()), dtype=np.float64) / total_samples
            return -np.sum(probs * np.log2(probs + 1e-12))
        parent_entropy = entropy_from_counts(total_counts, n)
        left_entropy = entropy_from_counts(left_counts, n_left)
        right_entropy = entropy_from_counts(right_counts, n_right)
        weighted = 0.0
        if n > 0:
            weighted = (n_left/n)*left_entropy + (n_right/n)*right_entropy
        return parent_entropy - weighted

    def _choose_best_threshold(self, y, X_col):
        n = len(X_col)
        if n > self.n_bins:
            quantiles = np.linspace(0, 1, self.n_bins+2)[1:-1]
            candidates = np.unique(np.quantile(X_col, quantiles))
        else:
            candidates = np.unique(X_col)
        best_ig, best_thresh = -np.inf, None
        for thresh in candidates:
            ig = self._chunked_information_gain(y, X_col, thresh)
            if ig > best_ig:
                best_ig, best_thresh = ig, thresh
        if best_ig == -np.inf:
            return 0.0, None
        return best_ig, best_thresh

    def fit(self, X, y, depth=0):
        if depth == 0:
            self._node_count = 0
            self._total_samples = len(y)
            if self.verbose:
                print(f"\nStarting tree training with {self._total_samples} samples...")
                print("=" * 60)
        
        y = np.array(y)
        unique_labels, label_counts = np.unique(y, return_counts=True)
        
        # Progress indicator
        if self.verbose and depth == 0:
            print(f"[Depth 0] Processing root node with {len(y)} samples")
        elif self.verbose:
            indent = "  " * depth
            print(f"{indent}[Depth {depth}] {len(y)} samples, {len(unique_labels)} classes")
        
        if (depth >= self.max_depth or len(y) < self.min_samples_split or len(unique_labels) == 1):
            majority_label = unique_labels[np.argmax(label_counts)]
            self._node_count += 1
            if self.verbose:
                indent = "  " * depth
                reason = []
                if depth >= self.max_depth:
                    reason.append("max depth")
                if len(y) < self.min_samples_split:
                    reason.append(f"min samples")
                if len(unique_labels) == 1:
                    reason.append("pure")
                print(f"{indent}-> Leaf node #{self._node_count}: class={majority_label} ({', '.join(reason)})")
            
            if depth == 0:
                self.root = MyDecisionTreeNode(prediction=majority_label)
                if self.verbose:
                    print(f"\nTraining complete. Total nodes: {self._node_count}")
                return self
            return MyDecisionTreeNode(prediction=majority_label)
        
        n_features = X.shape[1]
        best_ig, best_feat, best_thresh = -np.inf, None, None
        
        if self.verbose:
            indent = "  " * depth
            print(f"{indent}Evaluating {n_features} features...")
        
        for feat in range(n_features):
            ig, thresh = self._choose_best_threshold(y, X[:, feat])
            if ig > best_ig and thresh is not None:
                best_ig, best_feat, best_thresh = ig, feat, thresh
        
        if best_ig <= 0 or best_thresh is None:
            majority_label = unique_labels[np.argmax(label_counts)]
            self._node_count += 1
            if self.verbose:
                indent = "  " * depth
                print(f"{indent}-> Leaf node #{self._node_count}: class={majority_label} (no valid split)")
            
            if depth == 0:
                self.root = MyDecisionTreeNode(prediction=majority_label)
                if self.verbose:
                    print(f"\nTraining complete. Total nodes: {self._node_count}")
                return self
            return MyDecisionTreeNode(prediction=majority_label)
        
        self._node_count += 1
        if self.verbose:
            indent = "  " * depth
            print(f"{indent}-> Split node #{self._node_count}: X[{best_feat}] <= {best_thresh:.4f} (IG={best_ig:.4f})")
        
        mask = X[:, best_feat] <= best_thresh
        left_child = self.fit(X[mask], y[mask], depth+1)
        right_child = self.fit(X[~mask], y[~mask], depth+1)
        node = MyDecisionTreeNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)
        
        if depth == 0:
            self.root = node
            if self.verbose:
                print("=" * 60)
                print(f"Training complete. Total nodes: {self._node_count}")
            return self
        return node

    def predict_one(self, x, node=None):
        node = node or self.root
        while node.prediction is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        return np.array([self.predict_one(row) for row in X])

    def feature_importances(self):
        """
        Calculate feature importance based on total information gain.
        Returns a dictionary mapping feature indices to their importance scores.
        """
        if self.root is None:
            return {}
        
        importances = collections.defaultdict(float)
        
        def traverse(node, n_samples=1.0):
            """Recursively traverse tree and accumulate importance"""
            if node is None or node.prediction is not None:
                return
            
            # Add this node's contribution (weighted by samples at this node)
            if node.feature is not None:
                importances[node.feature] += n_samples
            
            # Traverse children (approximate equal split for simplicity)
            if node.left is not None:
                traverse(node.left, n_samples * 0.5)
            if node.right is not None:
                traverse(node.right, n_samples * 0.5)
        
        traverse(self.root, 1.0)
        
        # Normalize to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v/total for k, v in importances.items()}
        
        return dict(importances)
    
    def plot_feature_importances(self, feature_names=None):
        """
        Plot feature importances as a bar chart.
        """
        importances = self.feature_importances()
        
        if not importances:
            print("No feature importances available (tree not fitted or has no splits)")
            return
        
        features = sorted(importances.keys())
        values = [importances[f] for f in features]
        
        if feature_names is None:
            labels = [f"X[{f}]" for f in features]
        else:
            labels = [feature_names[f] for f in features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels, values, color='#4CAF50', edgecolor='#2E7D32', linewidth=2)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=10, weight='bold')
        
        ax.set_xlabel('Importance', fontsize=12, weight='bold')
        ax.set_ylabel('Feature', fontsize=12, weight='bold')
        ax.set_title('Feature Importances', fontsize=14, weight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()

    def plot(self, feature_names=None, max_depth=None, filled=True, impurity=False, 
             node_ids=False, precision=3, ax=None):
        """
        Enhanced matplotlib plot inspired by sklearn's implementation.
        
        Parameters
        ----------
        feature_names : list, optional
            Names of features
        max_depth : int, optional
            Maximum depth to display. If None, shows entire tree
        filled : bool, default=True
            Fill nodes with colors
        impurity : bool, default=False
            Show impurity/entropy values (not implemented yet)
        node_ids : bool, default=False
            Show node ID numbers
        precision : int, default=3
            Decimal precision for thresholds
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates new figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        
        if ax is None:
            # Count nodes for optimal sizing
            def count_nodes_and_depth(node, depth=0):
                if node is None:
                    return 0, depth
                if node.prediction is not None or (node.left is None and node.right is None):
                    return 1, depth
                left_count, left_depth = count_nodes_and_depth(node.left, depth+1)
                right_count, right_depth = count_nodes_and_depth(node.right, depth+1)
                return 1 + left_count + right_count, max(left_depth, right_depth)
            
            total_nodes, actual_depth = count_nodes_and_depth(self.root)
            
            # Dynamic figure sizing
            width = max(12, min(actual_depth * 3.5, 50))
            height = max(8, min(actual_depth * 2.5, 40))
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig = ax.figure
            total_nodes = 100  # Default if ax provided
            actual_depth = 10
        
        ax.clear()
        ax.set_axis_off()
        
        # Build tree structure using Reingold-Tilford algorithm (like sklearn)
        class DrawNode:
            def __init__(self, tree_node, node_id, x=0, y=0):
                self.tree_node = tree_node
                self.node_id = node_id
                self.x = x
                self.y = y
                self.left = None
                self.right = None
                self.parent = None
                self.thread = None
                self.mod = 0
                self.ancestor = self
                self.change = 0
                self.shift = 0
                self._lmost_sibling = None
                self.number = 0
        
        # Simplified Reingold-Tilford layout
        def build_tree(node, depth=0, node_id=0):
            if node is None or (max_depth is not None and depth > max_depth):
                return None, node_id
            
            draw_node = DrawNode(node, node_id)
            node_id += 1
            
            if node.left is not None and (max_depth is None or depth < max_depth):
                draw_node.left, node_id = build_tree(node.left, depth+1, node_id)
                if draw_node.left:
                    draw_node.left.parent = draw_node
            
            if node.right is not None and (max_depth is None or depth < max_depth):
                draw_node.right, node_id = build_tree(node.right, depth+1, node_id)
                if draw_node.right:
                    draw_node.right.parent = draw_node
            
            return draw_node, node_id
        
        def layout_tree(root):
            """Simple left-to-right layout"""
            def assign_x(node, x=0):
                if node is None:
                    return x
                x = assign_x(node.left, x)
                node.x = x
                x += 1
                x = assign_x(node.right, x)
                return x
            
            def assign_y(node, y=0):
                if node is None:
                    return
                node.y = y
                assign_y(node.left, y-1)
                assign_y(node.right, y-1)
            
            assign_x(root)
            assign_y(root)
            return root
        
        # Build and layout tree
        draw_root, _ = build_tree(self.root)
        if draw_root is None:
            return []
        
        draw_root = layout_tree(draw_root)
        
        # Get extents for scaling
        def get_extents(node, min_x=[float('inf')], max_x=[float('-inf')], 
                       min_y=[float('inf')], max_y=[float('-inf')]):
            if node is None:
                return
            min_x[0] = min(min_x[0], node.x)
            max_x[0] = max(max_x[0], node.x)
            min_y[0] = min(min_y[0], node.y)
            max_y[0] = max(max_y[0], node.y)
            get_extents(node.left, min_x, max_x, min_y, max_y)
            get_extents(node.right, min_x, max_x, min_y, max_y)
        
        min_x, max_x, min_y, max_y = [float('inf')], [float('-inf')], [float('inf')], [float('-inf')]
        get_extents(draw_root, min_x, max_x, min_y, max_y)
        
        # Scale coordinates to [0, 1]
        width_extent = max_x[0] - min_x[0] + 1
        height_extent = max_y[0] - min_y[0] + 1
        
        # Dynamic font sizing
        scale_factor = max(0.4, min(1.0, 25 / total_nodes))
        fontsize = max(7, int(11 * scale_factor))
        
        annotations = []
        
        def draw_node(node, parent_xy=None):
            if node is None:
                return
            
            # Normalize coordinates
            x = (node.x - min_x[0]) / width_extent
            y = (node.y - min_y[0]) / height_extent
            
            # Draw edge from parent
            if parent_xy is not None:
                ax.annotate(
                    '', xy=(x, y), xytext=parent_xy,
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='<-', color='#444', lw=1.5, 
                                  shrinkA=5, shrinkB=5),
                    zorder=1
                )
            
            # Create node label
            tree_node = node.tree_node
            if tree_node.prediction is not None or (tree_node.left is None and tree_node.right is None):
                # Leaf node
                if node_ids:
                    label = f"node #{node.node_id}\nclass = {tree_node.prediction}"
                else:
                    label = f"class = {tree_node.prediction}"
                box_color = "#e8f5e9" if filled else "white"
                border_color = "#4caf50"
            else:
                # Internal node
                feat_name = f"X[{tree_node.feature}]" if feature_names is None else feature_names[tree_node.feature]
                threshold_str = f"{tree_node.threshold:.{precision}f}"
                if node_ids:
                    label = f"node #{node.node_id}\n{feat_name} <= {threshold_str}"
                else:
                    label = f"{feat_name} <= {threshold_str}"
                box_color = "#fff9e6" if filled else "white"
                border_color = "#ff9800"
            
            # Draw node box
            ann = ax.annotate(
                label,
                xy=(x, y),
                xycoords='axes fraction',
                ha='center', va='center',
                fontsize=fontsize,
                bbox=dict(
                    boxstyle=f'round,pad={0.4*scale_factor}',
                    facecolor=box_color,
                    edgecolor=border_color,
                    linewidth=1.5
                ),
                zorder=3
            )
            annotations.append(ann)
            
            # Draw children
            draw_node(node.left, (x, y))
            draw_node(node.right, (x, y))
        
        draw_node(draw_root)
        
        # Adjust axis limits with padding
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        if ax is None:
            plt.show()
        
        return annotations

# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    np.random.seed(0)
    n_samples = 800
    n_features = 3
    X_train = np.random.randn(n_samples, n_features)
    y_train = (X_train[:,0] + 2*X_train[:,1] - X_train[:,2] > 1).astype(int)
    X_test = np.random.randn(150, n_features)
    y_test = (X_test[:,0] + 2*X_test[:,1] - X_test[:,2] > 1).astype(int)

    tree = MyDecisionTree(max_depth=4, min_samples_split=10, n_bins=20, chunk_size=100, verbose=True)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    print(f"Test set accuracy: {(y_pred == y_test).mean():.4f}")
    
    # Show feature importances
    print("\nFeature Importances:")
    importances = tree.feature_importances()
    for feat_idx, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  X{feat_idx}: {importance:.4f}")
    
    # Plot the tree
    tree.plot(feature_names=[f"X{i}" for i in range(n_features)])  # Shows full tree now!
    
    # Plot feature importances
    tree.plot_feature_importances(feature_names=[f"X{i}" for i in range(n_features)])