import os
import json
import glob
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from networkx.readwrite import json_graph
import networkx as nx

class GBMJsonObsDataset(InMemoryDataset):
    """
    Dataset for loading Geometric Block Model observed subgraphs from JSON files.
    Expects a directory containing JSON files where each JSON has:
      - 'graph': node-link dict with 'coords' and 'comm' node attrs
      - 'observations': dict with 'PairSamplingObservation'
    """
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # Process if not already
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        # Load with weights_only=False to allow PyG classes
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # All JSONs in the directory
        return [os.path.basename(p)
                for p in glob.glob(os.path.join(self.root, '*.json'))]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # No download
        pass

    def process(self):
        data_list = []
        # Iterate over JSON files
        for fname in self.raw_file_names:
            path = os.path.join(self.root, fname)
            with open(path, 'r') as f:
                raw = json.load(f)
            G_full = json_graph.node_link_graph(raw['graph'])
            obs = raw.get('observations', {}).get('PairSamplingObservation', {})
            edge_list = []
            if isinstance(obs, dict):
                for key in obs:
                    if key == 'sparsity':
                        continue
                    try:
                        u, v = json.loads(key)
                        edge_list.extend([(u, v), (v, u)])
                    except Exception:
                        continue
            # Build subgraph with all nodes and only observed edges
            subG = nx.Graph()
            subG.add_nodes_from(G_full.nodes(data=True))
            subG.add_edges_from(edge_list)
            # Convert to PyG Data
            data = from_networkx(subG)
            data.x = torch.tensor(
                [subG.nodes[n]['coords'] for n in subG.nodes()], dtype=torch.float)
            data.y = torch.tensor(
                [subG.nodes[n]['comm'] for n in subG.nodes()], dtype=torch.long)
            data_list.append(data)
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

if __name__ == '__main__':
    # Adjust path to JSON directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(
        'datasets', 'observations_generation', 'gbm_observation_01'
    )
    # Hyperparameters
    hidden = 64
    epochs = 100
    lr = 0.01
    batch_size = 16

    # Load observed subgraphs
    dataset = GBMJsonObsDataset(json_dir)
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_len = int(0.8 * len(dataset))
    train_dataset = dataset[:train_len]
    test_dataset = dataset[train_len:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Determine number of classes globally to match all targets
    num_classes = int(dataset.data.y.max().item() + 1)
    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=hidden,
        out_channels=num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes
        return total_loss / len(train_dataset)

    def test(loader):
        model.eval()
        correct = 0
        total = 0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.num_nodes
        return correct / total

    for epoch in range(1, epochs + 1):
        loss = train()
        if epoch % 10 == 0:
            acc = test(test_loader)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    final_acc = test(test_loader)
    print(f'Final Test Accuracy: {final_acc:.4f}')
