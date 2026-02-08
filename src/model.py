import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, input_length, d_model):
        super().__init__()
        self.position_embedding = nn.Embedding(input_length, d_model)

    def forward(self, x):
        B, T, _ = x.size()

        positions = torch.arange(T, device=x.device)
        positions = positions.unsqueeze(0).expand(B, T)

        pos_embed = self.position_embedding(positions)

        return x + pos_embed


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.input_length = model_config["input_length"]
        self.horizon = model_config["horizon"]
        self.d_model = model_config["d_model"]
        self.use_id_embedding = model_config["use_id_embedding"]
        self.output_mode = model_config["output_mode"]

        self.value_embedding = nn.Linear(1, self.d_model)

        self.position_embedding = nn.Embedding(
            self.input_length,
            self.d_model
        )

        if self.use_id_embedding:
            self.id_embedding = nn.Embedding(
                model_config["num_households"],
                model_config["id_embedding_dim"]
            )
            self.id_projection = nn.Linear(
                model_config["id_embedding_dim"],
                self.d_model
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=model_config["n_heads"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config["n_layers"],
        )

        self.head = nn.Linear(self.d_model, self.horizon)

    def forward(self, x, h_id=None):

        B, T, _ = x.size()

        x = self.value_embedding(x)

        positions = torch.arange(T, device=x.device)
        positions = positions.unsqueeze(0).expand(B, T)
        pos_embed = self.position_embedding(positions)

        x = x + pos_embed

        if self.use_id_embedding and h_id is not None:
            id_embed = self.id_embedding(h_id)
            id_embed = self.id_projection(id_embed)
            id_embed = id_embed.unsqueeze(1)
            x = x + id_embed

        x = self.encoder(x)

        if self.output_mode == "last_token":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)

        out = self.head(x)

        return out
