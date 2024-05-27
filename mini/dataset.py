from typing import Sequence
from functools import cached_property
from pathlib import Path
from dataclasses import dataclass
import json

from pydantic import BaseModel, computed_field, Field
import torchaudio
from torch import Tensor
from torch.nn.functional import interpolate
import torch
from einops import reduce
import lightning

import tqdm


import numpy.random as rng
from torch import Tensor, einsum
import torch.utils
import torchaudio.functional as F
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader





class ActivityDetector(BaseModel):
    min_power: float
    min_samples_activity: int
    min_samples_silence: int

    def _get_raw_activity_mask(self, wave: Tensor) -> Tensor:
        spec_transform = torchaudio.transforms.Spectrogram()
        spec = spec_transform(wave)
        is_active = reduce(spec > self.min_power, 'c f t -> t', 'max') * 1.0
        # not exact, but close enough for this purpose
        mask = interpolate(is_active[None, None, :], size=(wave.shape[-1]), mode='nearest')[0, 0]
        mask = torch.concat([mask, Tensor([0])])  # add a backstop for argmax lookups
        return mask
    
    def get_active_segments(self, wave: Tensor) -> list[tuple[int, int]]:
        mask = self._get_raw_activity_mask(wave).numpy()
        segments = []
        start = mask.argmax()  # start at first activation
        if mask[start] == 0:
            # edge case: no audio in file -> skip everything
            print('Found a file with no voice activations!')
            return []
        lookup = start
        while lookup < wave.shape[-1]:
            stop = lookup + (-mask[lookup:]).argmax()
            lookup = mask[stop:].argmax() + stop
            if lookup > stop + self.min_samples_silence or lookup == stop:
                if stop - start >= self.min_samples_activity:
                    segments.append((start, stop))
                if lookup == stop:
                    return segments
                start = lookup
            
        return segments



class FileMetaData(BaseModel):
    file_path: str
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str
    audio_segments: dict[str, list[tuple[int, int]]]
    activity_detector: ActivityDetector | None = Field(None)

    # @computed_field
    # @cached_property
    # def audio_segments(self) -> dict[str, list[tuple[int, int]]]:
    #     # the str key is meant to be used as a tag
    #     # segments must be non-overlapping between all keys
        
    @cached_property
    def audio_segments_flat(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for segment in self.audio_segments.values():
            out += segment
        return out

    @classmethod
    def from_path(cls, root: Path, path: Path, activity_detector: ActivityDetector | None):
        meta = torchaudio.info(root / path)
        if activity_detector:
            wav = torchaudio.load(root / path)[0]
            segments = activity_detector.get_active_segments(wav)
            segments = {
                'active': segments
            }
        else:
            segments = {
                'all': [(0, meta.num_frames)]
            }

        return cls(
            file_path=path.as_posix(),
            sample_rate=meta.sample_rate,
            num_frames=meta.num_frames,
            num_channels=meta.num_channels,
            bits_per_sample=meta.bits_per_sample,
            encoding=meta.encoding,
            activity_detector=activity_detector,
            audio_segments=segments,
        )
    
    def model_post_init(self, _) -> None:
        # validate non-overlapping segments and force segment calculation after init
        segments = self.audio_segments
        if len(segments) <= 1:
            return
        segments = sorted(segments)
        for first, second in zip(segments[:-1], segments[1:]):
            assert first[-1] < second[0], 'Overlapping segments'

class Mapper:
    """Map indices to files and staring locations."""
    def __init__(self, sample_len: int, segments: Sequence[tuple[str, int, int]]) -> None:
        self.sample_len = sample_len
        self.sequences = segments
        self.lens = [
            max(0, (end - start) // sample_len) for _, start, end in segments
        ]
        self.length = sum(self.lens)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int) -> tuple[str, int]:
        # linear search good enough
        # binary search only if number of segments is actually large (>1000)
        if index < 0 or index >= self.length:
            raise IndexError(f'Index out of bounds ({index}), should be 0 <= i < {self.length}')
        total = 0
        for i, l in enumerate(self.lens):
            if total <= index and index < total + l:
                filename, start, _ = self.sequences[i]
                return filename, start + self.sample_len * (index - total)
            total += l
        raise RuntimeError('Mismatch in segment lengths!')


class TracksMetaData(BaseModel):
    files: list[FileMetaData]
    root: Path = Field(exclude=True)

    @property
    def total_seconds(self) -> float:
        duration = 0
        for file in self.files:
            for start, end in file.audio_segments_flat:
                duration += (end - start) / file.sample_rate
        return duration

    @property
    def total_seconds_raw(self) -> float:
        duration = 0
        for file in self.files:
            duration += file.num_frames / file.sample_rate
        return duration

    @classmethod
    def from_paths(cls, root: Path, paths: Sequence[Path], detector: ActivityDetector | None):
        if detector:
            paths_wrapped = tqdm.tqdm(paths, desc='Processing files for audio activity')
        else:
            paths_wrapped = paths
        metas = [FileMetaData.from_path(root, path, detector) for path in paths_wrapped]
        return cls(root=root, files=metas)
    
    @classmethod
    def from_cache(cls, root: Path, path: str | Path):
        with open(root/path) as json_file:
            data = json_file.read()
            data = json.loads(data)
            data['root'] = root.as_posix()
            return cls.model_validate(data)
            return cls.model_validate_json(json_data=data, strict=True)
    
    def to_cache(self, path: str | Path):
        data = self.model_dump_json(indent=2)
        with open(path, 'w') as json_file:
            json_file.write(data)
    
    def get_index_map(self, sample_len: int, max_index_jitter: int):
        # filename, start, end - jitter
        all_segments: list[tuple[str, int, int]] = []
        # TODO: include tags, exclude files, etc
        for file in self.files:
            for _, segments in file.audio_segments.items():
                for start, end in segments:
                    all_segments.append((file.file_path, start, end - max_index_jitter))
        return Mapper(sample_len=sample_len, segments=all_segments)


@dataclass
class AudioAugment:
    polarity: bool = True
    adjust_db_lb: float = -5
    adjust_db_ub: float = 5

    def __call__(self, wave: Tensor) -> Tensor:
        n_batch = wave.shape[0]
        mults = self.get_mult(n_batch)
        return wave * mults[:, None].to(wave.device)

    def get_mult(self, n_batch) -> Tensor:
        polarity = torch.randint(0, 2, (n_batch,)) * 2 - 1
        r = torch.rand((n_batch,))
        db = self.adjust_db_lb + (self.adjust_db_ub - self.adjust_db_lb) * r
        amplitude = 10 ** (db / 10)
        return polarity * amplitude


@dataclass
class Degradation:
    min_len: int
    avg_len: int
    max_len: int
    min_count: int
    max_count: int
    target_share: float = 0.05  # TODO: better parametrization

    def __call__(self, wave: Tensor):
        batches, samples = wave.shape
        mask = torch.zeros_like(wave, dtype=torch.bool)
        n_blocks = torch.randint(
            low=self.min_count,
            high=self.max_count+1,
            size=(batches,)
        )
        for batch, blocks in enumerate(n_blocks):
            blocks = blocks.item()
            durations = -self.avg_len * (1 - torch.rand((blocks,))).log()
            durations = durations.clip(self.min_len, self.max_len).to(torch.int32)
            # normalize mask duration variance
            durations = durations / (self.target_share * samples) * durations.sum() 
            durations = durations.to(torch.int32) 

            starts = torch.randint(0, samples, (blocks,))
            ends = starts + durations
            for start, end in zip(starts, ends):
                mask[batch, start:end] = True
        return wave * ~mask, mask


class TracksDataset(Dataset[Tensor]):
    """Dataset of wav files with random sampling of position."""
    _cache_filename = 'cache.json'

    def __init__(
        self,
        root: str | Path,
        samples: int,
        activity_detector: ActivityDetector,
        use_cached: bool = True,
        write_cache: bool = True,
        include: str | None = None,
    ) -> None:
        self.samples_per_item = samples
        self.max_jitter = samples
        root = root if isinstance(root, Path) else Path(root)
        self.root = root.expanduser()

        # fetch file information
        cache_filename = self.root / self._cache_filename
        if use_cached and cache_filename.exists():
            self.meta_data = TracksMetaData.from_cache(root, cache_filename)
        else:
            all_wavs = sorted(
                path.relative_to(self.root)
                for path in self.root.glob(f'**/*.wav')
            )
            if include is not None:
                all_wavs = [
                    path for path in all_wavs
                    if include in path.as_posix()
                ]
            self.meta_data = TracksMetaData.from_paths(self.root, all_wavs, activity_detector)
            if write_cache:
                self.meta_data.to_cache(cache_filename)
        self.mapper = self.meta_data.get_index_map(samples, self.max_jitter)
    
    def __len__(self):
        return len(self.mapper)

    def __getitem__(self, index: int):
        filename, start = self.mapper[index]
        start += rng.randint(0, self.max_jitter)
        sample = torchaudio.load(
            uri=self.root/filename,
            frame_offset=start,
            num_frames=self.samples_per_item,
            normalize=True,
        )[0]
        channels = sample.shape[0]
        channel = rng.randint(0, channels)
        sample = sample[channel]
        assert isinstance(sample, Tensor)
        return sample


class TrackDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            dataset: TracksDataset,
            batch_size: int,
            train_share: float = 0.8,
            val_share: float = 0.1,
            test_share: float = 0.1,
        ) -> None:
        super().__init__()
        used_share = train_share + val_share + test_share
        assert used_share <= 1.0
        unused_share = 1 - used_share
        self.shares = (train_share, val_share, test_share, unused_share)

        self._dataset = dataset
        self.batch_size = batch_size
    
    def setup(self, stage: str):
        train, val, test, _ = random_split(self._dataset, self.shares)

        self.data_train = train
        self.data_val = val
        self.data_test = test
    
    def train_dataloader(self) :
        return DataLoader(self.data_train, self.batch_size, num_workers=0, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, self.batch_size, num_workers=0, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, self.batch_size, num_workers=0, shuffle=True)
