import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle


import os 
os.makedirs('out', exist_ok=True)

# create data object
def build_samples(gvariant_status, edge, cancer):
    edge_index = torch.tensor(edge, dtype=torch.long)  # shape [2, num_edges]
    samples = []
    for i in range(gvariant_status.shape[0]):
        x_i = torch.tensor(gvariant_status[i], dtype=torch.float32).unsqueeze(1)  # [4384, 1]
        y_i = torch.tensor([cancer[i]], dtype=torch.long)
        data = Data(x=x_i, edge_index=edge_index, y=y_i)
        samples.append(data)
    return samples
    
# Define GAT model
class GATGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.pool(x, batch)
        return self.fc(x)

    def get_node_embeddings(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return x

# Compute node-level perturbation impact per sample
def compute_node_impact_matrix(samples, model):
    model.eval()
    with torch.no_grad():
        ref_x = torch.zeros((samples[0].num_nodes, 1), dtype=torch.float32)
        ref_edge_index = samples[0].edge_index
        ref_embedding = model.get_node_embeddings(ref_x, ref_edge_index)

        impact_matrix = []
        for sample in samples:
            emb = model.get_node_embeddings(sample.x, sample.edge_index)
            impact = torch.norm(emb - ref_embedding, dim=1)
            impact_matrix.append(impact.cpu().numpy())
    return np.array(impact_matrix)

# Train function
def train(model, loader, optimizer, criterion, epoch, save_embeddings=False):
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch).squeeze()
        loss = criterion(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #if save_embeddings and i == 0:
        #    embeddings = model.get_node_embeddings(batch.x, batch.edge_index).detach().cpu().numpy()
        #    node_scores = batch.x.cpu().numpy()
        #    np.save(f'./data/epoch_{epoch}_train_embeddings.npy', embeddings)
        #    np.save(f'./data/epoch_{epoch}_train_node_scores.npy', node_scores)

    return total_loss / len(loader)

# Evaluate function
def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).long()
            y_true.extend(batch.y.tolist())
            y_pred.extend(pred.tolist())
            y_prob.extend(prob.tolist())
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return acc, auc, precision, recall, f1, fpr, tpr, y_pred, y_prob

# Run cross validation with ROC and loss tracking
def run_cross_validation(samples, in_channels, hidden_channels=32, epochs=100, folds=5, task = None):
    labels = [data.y.item() for data in samples]
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(samples, labels)):
        print(f"\n===== Fold {fold+1} =====")
        train_dataset = [samples[i] for i in train_idx]
        test_dataset = [samples[i] for i in test_idx]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = GATGraphClassifier(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        losses = []

        for epoch in range(1, epochs + 1):
            loss = train(model, train_loader, optimizer, criterion, epoch, save_embeddings=True)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        acc, auc, precision, recall, f1, fpr, tpr, y_pred, y_prob = evaluate(model, test_loader)
        print(f"Fold {fold+1} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        fold_results.append((acc, auc, precision, recall, f1, fpr, tpr, y_pred, y_prob))

        # Plot training loss
        plt.figure()
        plt.plot(range(1, epochs+1), losses, label=f'Train Loss Fold {fold+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Fold {fold+1}')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(f'./out/{task}_Fold{fold+1}_trainingloss.png', dpi=150, bbox_inches='tight')

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Fold {fold+1} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Fold {fold+1}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
        plt.savefig(f'./out/{task}_Fold{fold+1}_test_roc.png', dpi=150, bbox_inches='tight')

        # Visualize node embeddings for first graph in test set
        test_sample = test_dataset[0]
        embeddings = model.get_node_embeddings(test_sample.x, test_sample.edge_index).detach().cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(6, 6))
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=test_sample.x.squeeze().numpy(), cmap='coolwarm', s=10)
        plt.title(f'TSNE of Node Embeddings (Fold {fold+1}')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.colorbar(label='Input Variant')
        plt.grid(True)
        plt.show()
        plt.savefig(f'./out/{task}_Fold{fold+1}_embending_tsne.png', dpi=150, bbox_inches='tight')

        # save node importance
        impact_matrix_train = compute_node_impact_matrix(train_dataset, model)
        impact_matrix_test = compute_node_impact_matrix(test_dataset, model)
        np.save(f'./out/{task}_Fold{fold}_train_impact_matrix.npy', impact_matrix_train)
        np.save(f'./out/{task}_Fold{fold}_test_impact_matrix.npy', impact_matrix_test)
        

    mean_acc = sum([a for a, _, _, _, _, _, _, _, _ in fold_results]) / folds
    mean_auc = sum([a for _, a, _, _, _, _, _, _, _ in fold_results]) / folds
    mean_precision = sum([a for _, _, a, _, _, _, _, _, _ in fold_results]) / folds
    mean_recall = sum([a for _, _, _, a, _, _, _, _, _ in fold_results]) / folds
    mean_f1 = sum([a for _, _, _, _, a, _, _, _, _ in fold_results]) / folds

    print(f"\nMean Accuracy: {mean_acc:.4f}, Mean AUC: {mean_auc:.4f}, Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}, Mean F1: {mean_f1:.4f}")

    # Save the results with pickle
    with open(f'./data/{task}_prediction_results.pkl', 'wb') as f:
        pickle.dump(fold_results, f)

    return fold_results


germline_input = np.load('./data/tcga_germlinevariants_7861patients.npy', allow_pickle= True).item()
germline_input.keys()


gcancer = ['PAAD','PRAD','LUSC','THCA','BLCA','LUSC','LUAD','COAD','BRCA','OV','LGG']
scancer = ['STAD','SKCM','GBM']

output_all = {}


for g in gcancer:
    for s in scancer:
        cancer_index = np.where(pd.Series(germline_input['cancer']) == g)
        control_index = np.where(pd.Series(germline_input['cancer']) == s)
        all_index = np.concatenate((cancer_index[0], control_index[0]))
        
        g_input = np.concatenate((germline_input['gvariant_status'][cancer_index],germline_input['gvariant_status'][control_index]),axis = 0)
        edge_input = germline_input['edge']
        label = [1] * len(cancer_index[0]) + [0] * len(control_index[0]) 
        
        input_object = build_samples(g_input, 
                        edge_input, 
                        label)

        output = run_cross_validation(input_object, in_channels=1, epochs=100, folds = 5, task = f'{g}_vs_{s}')
        #np.save(f'./out/{g}_vs_{s}_prediction_results.npy', output, allow_pickle=True)

        output_all[f'{g}_vs_{s}'] = output


# Save the results with pickle
with open(f'./data/all_prediction_results.pkl', 'wb') as f:
    pickle.dump(output_all, f)


#np.save('./out/final_prediction_results.npy', output_all, allow_pickle=True)

