import glob
import json
import os
import logging
import hydra
import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from inferencer import Inferencer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed, Trainer, TrainingArguments
from src.metrics import get_metric
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.utils.statistics import show_statistics
from src.models.api_client import run_api
from src.utils.misc import parallel_run, save_json
from src.models.model import ppl_generate

logger = logging.getLogger(__name__)

class Prefix_Inferencer(Inferencer):
    def init_model_dataloader(self, cfg):
        # same as class:Inferencer
        self.dataset_reader.shard(self.accelerator)

        if self.accelerator.is_main_process:
            logger.info(f"Statistics after sharding: ")
            show_statistics(self.dataset_reader.encoded_dataset, "main dataset")
            show_statistics(self.dataset_reader.index_reader.encoded_dataset, "index dataset")

        # different DataCollator and model
        # to be fix: rewrite DataCollator
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.accelerator.device)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model = hu.instantiate(cfg.model_config.model)
        if cfg.train:
            vocab_size = model.config.vocab_size
            new_vocab_size = vocab_size + cfg.new_tokens
            model.resize_token_embeddings(new_vocab_size)
        model = self.accelerator.prepare(model)
        return model, dataloader
    
    def forward(self):
        # to be fix: split dataset into train/eval, refer to scorer
        trainer = Trainer(
            model = self.model,
            args = hu.instantiate(self.cfg.training_args),
            data_collator=self.dataloader.collate_fn,
            train_dataset=None,
            eval_dataset=None,
        )
        pass

    def eval_forward(self):
        super().forward()

@hydra.main(config_path="configs", config_name="prefix_inferencer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    if cfg.model_config.model_type == 'hf':
        accelerator = Accelerator()
        inferencer = Prefix_Inferencer(cfg, accelerator)
        # to be fix
        if cfg.is_train:
            inferencer.forward()
            pass
        else:
            inferencer.eval_forward()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                inferencer.write_results()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
