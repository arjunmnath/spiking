from training.engine.checkpoint_manager import CheckpointManager


with CheckpointManager(bucket_mame='ckpt-spikign') as manager:
    manager.save_checkpoint(
        step=2,
        model_data={"a": 1},
        optimizer_data={"b": 2},
        meta_data={"step": 2},
        rank=0,
    )
    manager.save_checkpoint(
        step=2,
        model_data={"a": 1},
        optimizer_data={"b": 2},
        meta_data={"step": 2},
        rank=1,
    )