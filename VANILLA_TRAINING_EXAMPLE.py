#!/usr/bin/env python3
"""
VANILLA TRAINING SCRIPT - Demonstrates Zero-Code-Change JIT Checkpointing

This is exactly what a user's training script looks like.
NO explicit JIT checkpoint configuration!
Just set environment variables and it works.
"""

import ray
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def train_func(config):
    """
    User's normal training function.
    No JIT checkpoint code - just normal training!
    """
    import torch

    # Get starting checkpoint if available (for recovery)
    checkpoint = train.get_checkpoint()

    if checkpoint:
        # Resume from checkpoint
        model_state = checkpoint.to_dict()
        start_epoch = model_state["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        # Start fresh
        model_state = {"weights": torch.randn(10, 10), "epoch": 0}
        start_epoch = 0
        print("Starting fresh training")

    # Normal training loop
    for epoch in range(start_epoch, 10):
        # Training work
        model_state["weights"] = model_state["weights"] + torch.randn(10, 10) * 0.1
        model_state["epoch"] = epoch

        # Report metrics
        loss = 1.0 / (epoch + 1)
        train.report({"epoch": epoch, "loss": loss})

        print(f"Epoch {epoch}: loss={loss:.4f}")


def main():
    """Main training entry point."""

    # Initialize Ray
    ray.init()

    # ========================================================================
    # VANILLA TRAINER - No JIT checkpoint config!
    # ========================================================================
    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2),
        run_config=RunConfig(
            storage_path="/mnt/ray-checkpoints",
            name="my_training_job"
            # ⭐ NO jit_checkpoint_config parameter! ⭐
            # JIT checkpointing enabled via environment variables!
        ),
    )

    # Run training
    result = trainer.fit()

    print(f"Training completed! Final metrics: {result.metrics}")

    ray.shutdown()


if __name__ == "__main__":
    """
    To enable JIT checkpointing, just set environment variables:

    $ export RAY_TRAIN_JIT_CHECKPOINT_ENABLED=1
    $ export RAY_TRAIN_JIT_CHECKPOINT_KILL_WAIT=5.0
    $ python VANILLA_TRAINING_EXAMPLE.py

    That's it! No code changes needed!

    For KubeRay, add annotation to your RayJob:

    annotations:
      ray.io/jit-checkpoint-enabled: "true"
      ray.io/jit-checkpoint-kill-wait: "5.0"

    KubeRay operator will inject the environment variables automatically!
    """
    main()
