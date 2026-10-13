import numpy as np
import pytest

from core_modules.flow_retrieval import FlowMemoryBank
from core_modules.protocol import validate_three_way_split


def test_split_guard_rejects_overlap():
    with pytest.raises(ValueError):
        validate_three_way_split(np.array([0, 1]), np.array([1, 2]), np.array([3]), n_obs=4)


def test_validation_query_is_not_a_bank_member():
    rng = np.random.default_rng(0)
    bank_z = rng.normal(size=(10, 4)).astype(np.float32)
    bank_p = rng.normal(size=(10, 3)).astype(np.float32)
    val_z = rng.normal(size=(3, 4)).astype(np.float32)
    bank = FlowMemoryBank(use_faiss=False, temperature=0.5)
    bank.fit(bank_z, bank_p)
    result = bank.retrieve(val_z, k=3)
    assert result.indices.min() >= 0
    assert result.indices.max() < len(bank_z)
    with pytest.raises(ValueError):
        bank.retrieve(val_z, k=3, query_bank_indices=np.array([10, 11, 12]))


def test_training_self_exclusion():
    z = np.eye(6, dtype=np.float32)
    p = np.arange(12, dtype=np.float32).reshape(6, 2)
    bank = FlowMemoryBank(use_faiss=False)
    bank.fit(z, p)
    result = bank.retrieve(z, k=2, query_bank_indices=np.arange(6))
    for i in range(6):
        assert i not in result.indices[i]
