import torch
from datetime import datetime
from transformers import TrainingArguments
from transformers import Trainer
from transformers.modeling_utils import ModelOutput
import wandb

class QuickDrawTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(inputs["pixel_values"])
        labels = inputs.get("labels")

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, ModelOutput(logits=logits, loss=loss)) if return_outputs else loss

timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
training_args = TrainingArguments(
    output_dir=f'./outputs_20k_{timestamp}',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    report_to=['wandb', 'tensorboard'],  # Update to just tensorboard if not using wandb
    logging_strategy='steps',
    logging_steps=100,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    learning_rate=0.003,
    fp16=torch.cuda.is_available(),
    num_train_epochs=20,
    run_name=f"quickdraw-med-{timestamp}",  # Can remove if not using wandb
    warmup_steps=10000,
    save_total_limit=5,
)

trainer = QuickDrawTrainer(
    model,
    training_args,
    data_collator=ds.collate_fn,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=None,
    compute_metrics=quickdraw_compute_metrics,
)

try:
    # Training
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Evaluation
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
except:
    pass
finally:
    wandb.finish()