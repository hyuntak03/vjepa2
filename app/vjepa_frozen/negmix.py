"""negimp 학습에 음의 손실을 안 거는 데이터셋 (자연 영상 등) 을 섞기 (2026-11-06, 자동 탐색 후보 — 데이터 원칙: 합성만으로 학습하지 않는다).

data.datasets[] 항목에 `negimp_off: true` 를 주면 train.py 가
  - 그 항목에는 불가능 짝 정보를 붙이지 않고 (mp4_list 가 아니어도 된다),
  - 만든 데이터셋을 ZeroCode 로 감싸 둘째 값 (불가능 종류 코드) 을 늘 0 (가능) 으로 바꾼다.
왜 감싸나: video_csv (src VideoDataset) 의 둘째 값은 행동 라벨이다. 그대로 두면 TrainCollator(with_imp) 가 라벨을 불가능 코드로 읽는다.
짝 손실 (negimp.mode=paired) 과는 같이 못 쓴다 — collator 가 배치 안 모든 샘플에 불가능 클립이 있다고 본다.
"""
import torch


class ZeroCode(torch.utils.data.Dataset):
    """(clips, 아무 값, idx) → (clips, 0, idx)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        out = self.ds[i]
        return out[0], 0, out[2]
