import pytorch_lightning as pl
import yaml
from model import RoBERTaClassification
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import get_project_root

PROJECT_ROOT = get_project_root()

if __name__ == "__main__":
    with open(PROJECT_ROOT / "config.yml") as fin:
        params = yaml.load(fin, Loader=yaml.FullLoader)
    classifier = RoBERTaClassification(params)

    checkpoint_callback = ModelCheckpoint(params["training"]["logdir"])
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=params["training"]["num_epochs"],
    )

    trainer.fit(classifier)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path)
