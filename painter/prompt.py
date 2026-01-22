from itertools import product
from typing import Iterable, List
import random
import logging


class Prompt:
    def __init__(
        self,
        positive: str,
        negative: str | None,
        width: int,
        height: int,
        seed: int | None = None,
        cfg: float | None = None,
        steps: int | None = None,
        name: str = "",
    ):
        self.positive = positive
        self.negative = negative
        self.width = width
        self.height = height
        self.seed = seed
        self.cfg = cfg
        self.steps = steps

        if name == "":
            self.name = str(self)
        else:
            self.name = name

    def __str__(self):
        name = self.positive.translate(
            str.maketrans({" ": "", ",": "", ":": "", "(": "", ")": "", ".": ""})
        )
        while len((name).encode("utf-8")) > 240:
            name = name[:: random.randint(4, 9)]  # drop every n-th char
        return "IMAGE_" + name


class PromptGrid:
    def __init__(
        self, width: int, height: int, pos_dropout: int = 0, neg_dropout: int = 0
    ):
        self.width = width
        self.height = height
        self.prompts: list[Prompt] = []

        self.pos_dropout = pos_dropout
        self.neg_dropout = neg_dropout

    def __iter__(self):
        return iter(self.prompts)

    def __len__(self):
        return len(self.prompts)

    def add(
        self,
        positive: str | Iterable[str],
        negative: str | Iterable[str] | None = None,
        seed: int | Iterable[int] | None = 1234,
        cfg: float = 6.0,
        steps: int = 30,
        name: str = "",
    ) -> None:
        if seed is None:
            seed = random.randint(0, 2 ^ 63 - 1)
        pos, neg, seeds = self._normalize(positive, negative, seed)

        pos, neg = self._dropout(pos, neg)

        self.prompts.extend(
            Prompt(p, n, self.width, self.height, seed=s, name=name, cfg=cfg, steps=steps)
            for p, n, s in zip(pos, neg, seeds, strict=True)
        )

    def add_variations(
        self,
        positives: list[str],
        negatives: list[str] | list[None],
        seeds: list[int] | list[None] | None,
        cfgs: list[float] | list[None] = [None],
        steps: list[int] | list[None] = [None],
        names: list[str] = [""],
    ) -> None:
        
        if seeds == None:
            seeds = [None]

        for (name, pos), neg, seed, cfg, steps in product(
            zip(names, positives), negatives, seeds, cfgs, steps
        ):
            if seed is None:
                seed = random.randint(0, 2**63 - 1)

            pos_list, neg_list = self._dropout([pos], [neg])

            self.prompts.append(
                Prompt(
                    positive=pos_list[0],
                    negative=neg_list[0],
                    width=self.width,
                    height=self.height,
                    seed=seed,
                    cfg=cfg,
                    steps=steps,
                    name=name,
                )
            )

    @staticmethod
    def construct_prompt_dimensions(
        variations: Iterable[Iterable[str]],
    ) -> List[List[str]]:

        lists = [list(dimension) for dimension in variations]
        return [", ".join(combination) for combination in product(*lists)]

    @staticmethod
    def construct_prompt_dimensions_from_files(files: Iterable[str]) -> List[List[str]]:
        dimensions = []
        for file in files:
            with open(file) as f:
                dimension = f.readlines()
                dimension = [d.replace("\n", "") for d in dimension]
                dimensions.append(dimension)
        return dimensions

    @staticmethod
    def _normalize(
        positive: str | Iterable[str],
        negative: str | Iterable[str] | None,
        seed: int | Iterable[int],
    ):
        pos = [positive] if isinstance(positive, str) else list(positive)
        neg = (
            [negative]
            if (isinstance(negative, str) or negative is None)
            else list(negative)
        )
        seeds = [seed] if not isinstance(seed, Iterable) else list(seed)

        max_len = max(len(pos), len(neg), len(seeds))
        if max_len == 0:
            return [], [], []

        def extend_list(lst, target_len):
            if not lst:
                return []
            return (lst * (target_len // len(lst) + 1))[:target_len]

        pos = extend_list(pos, max_len)
        neg = extend_list(neg, max_len)
        seeds = extend_list(seeds, max_len)

        print(max_len, pos, neg, seeds)

        return pos, neg, seeds

    def _dropout(self, pos: List[str], neg: List[str]):
        new_pos = []
        for p_str in pos:
            if self.pos_dropout > 0:
                parts = p_str.split(", ")
                n_drop = int(len(parts) * self.pos_dropout)
                if n_drop > 0:
                    drop_indices = set(random.sample(range(len(parts)), n_drop))
                    parts = [p for i, p in enumerate(parts) if i not in drop_indices]
                p_str = ", ".join(parts)
            new_pos.append(p_str)

        new_neg = []
        for n_str in neg:
            if self.neg_dropout > 0:
                parts = n_str.split(", ")
                n_drop = int(len(parts) * self.neg_dropout)
                if n_drop > 0:
                    drop_indices = set(random.sample(range(len(parts)), n_drop))
                    parts = [p for i, p in enumerate(parts) if i not in drop_indices]
                n_str = ", ".join(parts)
            new_neg.append(n_str)

        return new_pos, new_neg
