"""
MLP baseline for ACDC cardiac pathology, evaluated with exactly the same protocol as the Tsetlin Machine
notebook (see acdc_common.py for the protocol; only the model differs).

Architecture:
    Linear(d, h) -> BatchNorm -> ReLU -> Dropout(p)
    Linear(h, h/2) -> BatchNorm -> ReLU -> Dropout(p)
    Linear(h/2, 5)

Model-specific choices
* Features standardised with the mean / SD of the training part of each split only.
* Search space: hidden size, dropout, learning rate, weight decay, batch size (Adam, cross-entropy loss).
* Number of epochs: as for the TM, the argmax of the fold-averaged validation-accuracy curve smoothed over
  +/-2 epochs (MAX_EPOCHS explored instead of the TM's 60, because one MLP epoch is only a few
  mini-batch updates).
* Scores for temperature scaling: the logits.

Run:
    python acdc_dnn.py --train features_train.csv --test features_test.csv --out results_dnn
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

import acdc_common as C


class MLP(nn.Module):
    def __init__(self, d, h, dropout, n_classes=len(C.CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def suggest(trial):
    return {"hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32, 64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.6),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32])}


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return [torch.tensor((X - mu) / sd, dtype=torch.float32) for X in (Xtr, *others)]


def make_model(d, p, seed):
    torch.manual_seed(int(seed))        # fixes initialisation, dropout masks and mini-batch order of this run
    model = MLP(d, p["hidden_dim"], p["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    return model, opt


def train_epoch(model, opt, X, y, batch_size, loss_fn=nn.CrossEntropyLoss()):
    model.train()
    perm = torch.randperm(len(y))
    for i in range(0, len(y), batch_size):
        idx = perm[i:i + batch_size]
        if len(idx) < 2:  # BatchNorm requires more than one sample
            continue
        opt.zero_grad()
        loss_fn(model(X[idx]), y[idx]).backward()
        opt.step()


@torch.no_grad()
def logits_of(model, X):
    model.eval()
    return model(X).numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features_train.csv")
    ap.add_argument("--test", default="features_test.csv")
    ap.add_argument("--out", default="results_dnn")
    ap.add_argument("--max_epochs", type=int, default=200)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    D = C.load_data(args.train, args.test)

    def curve(Xtr, ytr, Xva, yva, p, seed):
        Xtr_t, Xva_t = standardise(Xtr, Xva)
        ytr_t = torch.tensor(ytr)
        model, opt = make_model(Xtr.shape[1], p, seed)
        accs = []
        for _ in range(args.max_epochs):
            train_epoch(model, opt, Xtr_t, ytr_t, p["batch_size"])
            accs.append(np.mean(logits_of(model, Xva_t).argmax(1) == yva))
        return np.array(accs)

    def fit_scores(Xtr, ytr, Xev, p, seed):
        Xtr_t, Xev_t = standardise(Xtr, Xev)
        ytr_t = torch.tensor(ytr)
        model, opt = make_model(Xtr.shape[1], p, seed)
        for _ in range(p["epochs"]):
            train_epoch(model, opt, Xtr_t, ytr_t, p["batch_size"])
        return logits_of(model, Xev_t), model

    C.run_protocol("MLP", D, suggest, curve, fit_scores, args.out, tau_bounds=(0.05, 20.0),
                   extra_config={"model": "MLP (2 hidden layers, BatchNorm, dropout, Adam)",
                                 "epochs_max": args.max_epochs, "score": "logits"},
                   pkgs=["torch", "optuna", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib"])


if __name__ == "__main__":
    main()
