# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from args import *
from utils import *
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer,
    TrainerCallback
)
from datasets import load_dataset, Dataset
import evaluate


torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        generation_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )


    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print("Detected that training was already completed!")

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print("loaded model")
    set_seed(args.seed)


    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    import threading

    class SyncModelsCallbackA(TrainerCallback):
        def __init__(self, modelA, modelB, condition, global_step):
            self.modelA = modelA
            self.modelB = modelB
            self.outputB = None
            self.condition = condition
            self.global_step = global_step

        def on_step_begin(self, args, state, control, model=None, **kwargs):
            with self.condition:
                # Ensure that modelA does not start the next step without modelB finishing the last step
                while self.global_step.value != state.global_step:
                    self.condition.wait()

                outputA = self.modelA(**kwargs)
                kwargs['input_ids'] = torch.cat([kwargs['input_ids'], outputA[0]], dim=-1)
                self.condition.notify()  # Notify modelB that it can proceed

            return kwargs

        def on_step_end(self, args, state, control, **kwargs):
            self.outputB = kwargs['outputs']
            with self.condition:
                self.condition.wait()  # Wait for modelA to finish its step

    class SyncModelsCallbackB(TrainerCallback):
        def __init__(self, modelA, modelB, condition, global_step):
            self.modelA = modelA
            self.modelB = modelB
            self.condition = condition
            self.global_step = global_step

        def on_step_end(self, args, state, control, **kwargs):
            with self.condition:
                # Ensure that modelB does not start until modelA finishes the current step
                while self.global_step.value != state.global_step:
                    self.condition.wait()

    # Initialize your trainers
    condition = threading.Condition()
    global_step = threading.Value('i', 0)
    modelA = model
    modelB = model
    class CustomTrainerA(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputA = model(**inputs)
            # Model A's loss is determined by the output of model B plus a loss calculated by other means
            lossA = some_loss_function(outputA, self.callback_handler.callbacks[0].outputB) + 1  # Assuming SyncModelsCallbackA is the first callback

            with self.callback_handler.callbacks[0].condition:
                self.callback_handler.callbacks[0].global_step.value += 1
                self.callback_handler.callbacks[0].condition.notify()  # Notify modelB that it can proceed

            return (lossA, outputA) if return_outputs else lossA

    class CustomTrainerB(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Model B's loss is determined normally
            outputs = model(**inputs)
            if "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = inputs.pop("lm_labels")
            lossB = model(**inputs)[0]

            with self.callback_handler.callbacks[0].condition:
                self.callback_handler.callbacks[0].condition.notify()  # Notify modelA that it can proceed

            return (lossB, outputs) if return_outputs else lossB
    trainerA = CustomTrainerA(
        callbacks=[SyncModelsCallbackA(modelA, modelB, condition, global_step)],
        model=modelA,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    

    trainerB = CustomTrainerB(
        model=modelB,
        args=training_args,
        callbacks=[SyncModelsCallbackB(modelA, modelB, condition, global_step)],
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},    )


    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == "mmlu-zs":
            mmlu_dataset = load_dataset(
                "json",
                data_files={
                    "eval": "data/mmlu/zero_shot_mmlu_val.json",
                    "test": "data/mmlu/zero_shot_mmlu_test.json",
                },
            )
            mmlu_dataset = mmlu_dataset.remove_columns("subject")
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == "mmlu" or args.mmlu_dataset == "mmlu-fs":
            mmlu_dataset = load_dataset(
                "json",
                data_files={
                    "eval": "data/mmlu/five_shot_mmlu_val.json",
                    "test": "data/mmlu/five_shot_mmlu_test.json",
                },
            )
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")

        # TODO: Fix the mmlu file,
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(
                        trainer.model,
                        batch,
                        prediction_loss_only=False,
                    )
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch["labels"][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id - 1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {"mmlu_loss": loss_mmlu / len(data_loader)}
                subject = mmlu_dataset["subject"]
                subjects = {s: {"refs": [], "preds": []} for s in set(subject)}
                for s, p, r in zip(subject, preds, refs):
                    subjects[s]["preds"].append(p)
                    subjects[s]["refs"].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]["refs"],
                        predictions=subjects[subject]["preds"],
                    )["accuracy"]
                    results[
                        f"mmlu_{args.mmlu_split}_accuracy_{subject}"
                    ] = subject_score
                    subject_scores.append(subject_score)
                results[f"mmlu_{args.mmlu_split}_accuracy"] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    print_dtypes_parameters(model)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        def train_launch(trainers):
            logger.info("*** Train ***")
            # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
            # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
            train_result = trainers.train()
            metrics = train_result.metrics
            trainers.log_metrics("train", metrics)
            trainers.save_metrics("train", metrics)
            trainers.save_state()
            all_metrics.update(metrics)
                # Train your models in separate threads
        threadA = threading.Thread(target=train_launch(trainerA))
        threadB = threading.Thread(target=train_launch(trainerB))
        threadA.start()
        threadB.start()
        threadA.join()
        threadB.join()

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(
            test_dataset=data_module["predict_dataset"], metric_key_prefix="predict"
        )
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as fout:
            for i, example in enumerate(data_module["predict_dataset"]):
                example["prediction_with_input"] = predictions[i].strip()
                example["prediction"] = (
                    predictions[i].replace(example["input"], "").strip()
                )
                fout.write(json.dumps(example) + "\n")
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
