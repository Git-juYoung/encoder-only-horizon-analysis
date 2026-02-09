data_config = {
    "input_length": 672,
    "horizon": 96,
    "stride": 96,

    "train_start": "2011-01-01",
    "train_end": "2012-12-31",

    "val_start": "2013-01-01",
    "val_end": "2013-12-31",

    "test_start": "2014-01-01",
    "test_end": "2014-12-31",

    "normalization": "per_household_zscore"
}


model_config = {
    "input_length": 672,
    "horizon": 96,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,

    "positional_encoding": "learnable",

    "use_id_embedding": False, 
    "num_households": None,
    "id_embedding_dim": 16,

    "output_mode": "last_token"
}


train_config = {
    "epochs": 40,
    "batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,

    "lr": 1e-3,
    "weight_decay": 1e-4,

    "scheduler_mode": "min",
    "factor": 0.5,
    "scheduler_patience": 3,

    "early_patience": 7,
    "save_path": "models/best_model.pt"
}
