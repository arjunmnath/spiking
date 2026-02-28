import pytest
from pathlib import Path
from unittest.mock import MagicMock

from training.engine.checkpoint_manager import CheckpointManager


def test_initialization(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.get_base_dir",
        lambda: tmp_path
    )

    mock_s3 = MagicMock()
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.boto3.client",
        lambda _: mock_s3
    )

    manager = CheckpointManager("test-bucket")

    assert manager.bucket_name == "test-bucket"
    assert manager.checkpoints_dir == tmp_path / "checkpoints"
    assert manager.checkpoints_dir.exists()


def test_upload_to_s3(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.get_base_dir",
        lambda: tmp_path
    )

    mock_s3 = MagicMock()
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.boto3.client",
        lambda _: mock_s3
    )

    manager = CheckpointManager("bucket")

    file_path = tmp_path / "test.tar.gz"
    file_path.write_bytes(b"data")

    manager._upload_to_s3(file_path)

    mock_s3.upload_fileobj.assert_called_once()

def test_upload_to_s3_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.get_base_dir",
        lambda: tmp_path
    )

    manager = CheckpointManager("bucket")

    missing_file = tmp_path / "does_not_exist.tar.gz"

    with pytest.raises(FileNotFoundError):
        manager._upload_to_s3(missing_file)

def test_save_checkpoint_rank0(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.get_base_dir",
        lambda: tmp_path
    )

    mock_s3 = MagicMock()
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.boto3.client",
        lambda _: mock_s3
    )

    manager = CheckpointManager("bucket")

    # prevent real threading
    manager._executor.submit = MagicMock()

    manager.save_checkpoint(
        step=1,
        model_data={"a": 1},
        optimizer_data={"b": 2},
        meta_data={"step": 1},
        rank=0,
    )

    ckpt_path = tmp_path / "checkpoints" / "ckpt_000001"

    assert ckpt_path.exists()
    assert (ckpt_path / "model.pt").exists()
    assert (tmp_path / "checkpoints" / "meta.json").exists()

    manager._executor.submit.assert_called_once()

def test_save_checkpoint_non_zero_rank(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.get_base_dir",
        lambda: tmp_path
    )

    manager = CheckpointManager("bucket")
    manager._executor.submit = MagicMock()

    manager.save_checkpoint(
        step=2,
        model_data={"a": 1},
        optimizer_data={"b": 2},
        meta_data={"step": 2},
        rank=1,
    )

    ckpt_path = tmp_path / "checkpoints" / "ckpt_000002"

    assert not (ckpt_path / "model.pt").exists()
    assert (tmp_path / "checkpoints" / "optim_rank1.pt").exists()

    manager._executor.submit.assert_not_called()

def test_archive_and_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.get_base_dir",
        lambda: tmp_path
    )

    mock_s3 = MagicMock()
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.boto3.client",
        lambda _: mock_s3
    )

    manager = CheckpointManager("bucket")

    ckpt_dir = tmp_path / "checkpoints" / "ckpt_000003"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model.pt").write_bytes(b"model")

    manager._archive_and_upload(ckpt_dir)

    # verify upload called
    assert mock_s3.upload_fileobj.called

    # archive should be deleted
    archive_path = ckpt_dir.with_suffix(".tar.gz")
    assert not archive_path.exists()


def test_context_manager_success(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.get_base_dir",
        lambda: tmp_path
    )

    manager = CheckpointManager("bucket")

    manager._archive_and_upload = MagicMock()

    with manager as m:
        m._created_checkpoints.append(tmp_path / "dummy")

    manager._archive_and_upload.assert_called_once()


def test_context_manager_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "training.engine.checkpoint_manager.get_base_dir",
        lambda: tmp_path
    )

    manager = CheckpointManager("bucket")
    manager._archive_and_upload = MagicMock()

    try:
        with manager:
            manager._created_checkpoints.append(tmp_path / "dummy")
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    manager._archive_and_upload.assert_not_called()