"""Tests for geometry-based train/validation split helpers."""

import pytest

from util.data_processing.generate_split_indices import (
    build_geometry_index_map,
    build_split_dict,
    generate_geometry_split,
    generate_split_indices,
    resolve_flat_indices,
)


GEOMETRIES = ["0075_1001", "0076_1001", "0094_0001", "0095_0001", "0105_0001"]


class TestGenerateGeometrySplit:
    def test_reproducible_with_seed(self):
        train_a, val_a = generate_geometry_split(0.8, seed=42, geometry_names=GEOMETRIES)
        train_b, val_b = generate_geometry_split(0.8, seed=42, geometry_names=GEOMETRIES)
        assert train_a == train_b
        assert val_a == val_b

    def test_preserves_all_geometries(self):
        train, val = generate_geometry_split(0.8, seed=0, geometry_names=GEOMETRIES)
        assert sorted(train + val) == sorted(GEOMETRIES)
        assert len(train) >= 1
        assert len(val) >= 1

    def test_empty_geometry_list_raises(self):
        with pytest.raises(ValueError, match="geometry_names must be non-empty"):
            generate_geometry_split(0.8, seed=0, geometry_names=[])


class TestGenerateSplitIndices:
    def test_geometry_level_split_matches_row_ranges(self):
        row_ranges = [(0, 10), (10, 25), (25, 30), (30, 40), (40, 50)]
        train_idx, val_idx, train_geo_idx, val_geo_idx = generate_split_indices(
            num_pts=50,
            percent_train=0.8,
            seed=7,
            geometry_row_ranges=row_ranges,
        )
        assert train_geo_idx is not None
        assert val_geo_idx is not None
        assert len(train_idx) + len(val_idx) == 50
        assert set(train_idx).isdisjoint(set(val_idx))

        # Whole geometries: indices within one range stay together.
        for start, end in row_ranges:
            block = set(range(start, end))
            in_train = block <= set(train_idx)
            in_val = block <= set(val_idx)
            assert in_train or in_val

    def test_row_range_sum_mismatch_raises(self):
        with pytest.raises(ValueError, match="Geometry row ranges sum"):
            generate_split_indices(
                num_pts=10,
                percent_train=0.8,
                geometry_row_ranges=[(0, 3), (3, 6)],
            )


class TestBuildGeometryIndexMap:
    def test_merges_junction_and_vessel_ranges(self):
        junction_dict = {
            "geometry_row_ranges": [(0, 4), (4, 9)],
            "geometry_names_order": ["g1", "g2"],
        }
        vessel_dict = {
            "geometry_row_ranges": [(0, 2), (2, 5)],
            "geometry_names_order": ["g1", "g2"],
        }
        index_map = build_geometry_index_map(junction_dict, vessel_dict)
        assert index_map["g1"] == {"junction": (0, 4), "vessel": (0, 2)}
        assert index_map["g2"] == {"junction": (4, 9), "vessel": (2, 5)}


class TestResolveFlatIndices:
    def test_flattens_train_junction_indices(self):
        geometry_indices = {
            "g1": {"junction": (0, 3), "vessel": (0, 1)},
            "g2": {"junction": (3, 7), "vessel": (1, 4)},
        }
        split_dict = build_split_dict(
            train_geometries=["g1"],
            val_geometries=["g2"],
            geometry_indices=geometry_indices,
        )
        train_junction = resolve_flat_indices(split_dict, "junction", "train")
        val_junction = resolve_flat_indices(split_dict, "junction", "val")
        assert list(train_junction) == [0, 1, 2]
        assert list(val_junction) == [3, 4, 5, 6]

    def test_missing_geometry_in_split_raises(self):
        geometry_indices = {"g1": {"junction": (0, 2)}}
        with pytest.raises(KeyError, match="Geometry 'g2'"):
            build_split_dict(
                train_geometries=["g1"],
                val_geometries=["g2"],
                geometry_indices=geometry_indices,
            )
