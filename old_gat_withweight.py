import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import pickle



# ===================================================================
# 1. DATA PREPARATION (Modified to include node_weights)
# ===================================================================
'''def build_samples(gvariant_status, edge, cancer, node_weights_matrix):
    """
    Builds PyG Data objects for each sample.

    Args:
        gvariant_status (np.array): Shape [num_samples, num_nodes], mutation status.
        edge (list or np.array): Edge list for the graph structure.
        cancer (list or np.array): List of labels for each sample.
        node_weights_matrix (np.array): Shape [num_samples, num_nodes], weights for each node.
    """
    edge_index = torch.tensor(edge, dtype=torch.long)
    samples = []
    for i in range(gvariant_status.shape[0]):
        x_i = torch.tensor(gvariant_status[i], dtype=torch.float32).unsqueeze(1)
        y_i = torch.tensor([cancer[i]], dtype=torch.long)
        
        # Add the node weights for the current sample
        node_weights_i = torch.tensor(node_weights_matrix[i], dtype=torch.float32).unsqueeze(1)
        
        data = Data(x=x_i, edge_index=edge_index, y=y_i, node_weights=node_weights_i)
        samples.append(data)
    return samples'''


# The signature is changed to reflect we're getting a single array of weights
def build_samples(gvariant_status, edge, cancer, node_weights_array):
    """
    Builds PyG Data objects for each sample, applying the same node weights to all.
    """
    edge_index = torch.tensor(edge, dtype=torch.long)
    
    node_weights_tensor = None
    if node_weights_array is not None:
        # This conversion only happens if weights are actually passed in
        node_weights_tensor = torch.tensor(node_weights_array, dtype=torch.float32).unsqueeze(1)

    #node_weights_tensor = torch.tensor(node_weights_array, dtype=torch.float32).unsqueeze(1)
    

    samples = []
    for i in range(gvariant_status.shape[0]):
        x_i = torch.tensor(gvariant_status[i], dtype=torch.float32).unsqueeze(1)
        y_i = torch.tensor([cancer[i]], dtype=torch.long)
        
        # In the Data object, we now assign the pre-made tensor of weights
        data = Data(x=x_i, edge_index=edge_index, y=y_i, node_weights=node_weights_tensor)
        samples.append(data)
        
    return samples


# ===================================================================
# 2. GAT MODEL (Modified to accept and use node_weights)
# ===================================================================
class GATWeightedClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, node_weights=None):
        # --- KEY MODIFICATION ---
        # Apply node weights to the input features if they are provided
        if node_weights is not None:
            x = x * node_weights
        # ------------------------

        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.pool(x, batch)
        return self.fc(x)

    def get_node_embeddings(self, x, edge_index, node_weights=None):
        # Also modify this method for consistency
        if node_weights is not None:
            x = x * node_weights
            
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return x

# ===================================================================
# 3. TRAIN/EVALUATE FUNCTIONS (Modified to pass node_weights)
# ===================================================================
def train(model, loader, optimizer, criterion,device):
    model.train()
    total_loss = 0
    for batch in loader:
        
        batch = batch.to(device)
        
        optimizer.zero_grad()

        if hasattr(batch, 'node_weights'):
            out = model(batch.x, batch.edge_index, batch.batch, node_weights=batch.node_weights).squeeze(-1)
        else:
            # If not, call the model without the node_weights argument
            out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            
        # Pass node_weights from the batch to the model
        #out = model(batch.x, batch.edge_index, batch.batch, batch.node_weights).squeeze()
        loss = criterion(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            
            batch = batch.to(device)
            
            if hasattr(batch, 'node_weights'):
                out = model(batch.x, batch.edge_index, batch.batch, node_weights=batch.node_weights).squeeze(-1)
            else:
                out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            # Pass node_weights from the batch to the model
            #out = model(batch.x, batch.edge_index, batch.batch, batch.node_weights).squeeze()
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).long()
            y_true.extend(batch.y.tolist())
            y_pred.extend(pred.tolist())
            y_prob.extend(prob.tolist())
    
    # Handle cases where a class might not be in y_pred for metric calculations
    # This prevents errors on folds with poor performance
    try:
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except ValueError:
        # Fallback if only one class is present in predictions
        return 0.0, 0.5, 0.0, 0.0, 0.0, np.array([0,1]), np.array([0,1]), y_true, y_pred, y_prob

    return acc, auc, precision, recall, f1, fpr, tpr, y_true, y_pred, y_prob


# ===================================================================
# 4. CROSS-VALIDATION RUNNER (Main logic remains the same)
# ===================================================================
def run_cross_validation(samples, in_channels,device, hidden_channels=32, pos_weight = None, epochs=100, folds=5, task=None):
    labels = [data.y.item() for data in samples]
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(samples, labels)):
        print(f"\n===== Fold {fold+1} =====")
        train_dataset = [samples[i] for i in train_idx]
        test_dataset = [samples[i] for i in test_idx]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = GATWeightedClassifier(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Consider adding pos_weight here if needed

        losses = []
        for epoch in range(1, epochs + 1):
            loss = train(model, train_loader, optimizer, criterion,device)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        acc, auc, precision, recall, f1, fpr, tpr, y_true, y_pred, y_prob = evaluate(model, test_loader,device)
        
        print(f"Fold {fold+1} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        fold_results.append({
            'acc': acc, 
            'auc': auc, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1,
            'test_samples': test_idx,
            'true': y_true,
            'predictions': y_pred, 
            'scores': y_prob
        })
    
    # Calculate and print mean results
    mean_auc = np.mean([res['auc'] for res in fold_results])
    mean_f1 = np.mean([res['f1'] for res in fold_results])
    print(f"\nMean CV Results - AUC: {mean_auc:.4f}, F1: {mean_f1:.4f}")
    return fold_results