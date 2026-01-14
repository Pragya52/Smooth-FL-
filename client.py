# client.py
import torch
import torch.nn as nn

class Client:
    def __init__(self, client_id, client_model, dataloader, device='cuda', lr=1e-3):
        self.id = client_id
        self.model = client_model.to(device)
        self.dataloader = dataloader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def get_state(self):
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_state(self, state):
        self.model.load_state_dict(state)

    def local_train_one_epoch(self, server):
        self.model.train()
        total_loss = 0

        for x, y in self.dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # ---- CLIENT FORWARD ----
            A1 = self.model.forward_client(x)
            A1_send = A1.detach().cpu().clone()

            # ---- SERVER FORWARD ----
            A2_recv = server.forward_store(self.id, A1_send).to(self.device)

            # ---- CLIENT LOSS ----
            A2_local = A2_recv.detach().clone().requires_grad_(True)
            logits = self.model.compute_logits_from_A2(A2_local)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()   # produces grad on A2_local

            grad_A2 = A2_local.grad.detach().cpu().clone()

            # ---- SERVER BACKPROP ----
            grad_A1 = server.backward_from_gradA2(self.id, grad_A2).to(self.device)

            # ---- CLIENT BACKPROP ----
            torch.autograd.backward(A1, grad_tensors=grad_A1)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)
