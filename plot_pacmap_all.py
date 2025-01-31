# -*- coding: utf-8 -*-
"""A script for visualization of xvectors via PaCMAP.

Copyright (C) 2025 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pacmap
from hydra import compose, initialize
from omegaconf import DictConfig


def get_emotion_type(actor: str, emotion: str):
    """Return emotion type."""
    emotion_type = -1
    if emotion == "angry" and actor == "tsuchiya":
        emotion_type = 0
    elif emotion == "happy" and actor == "tsuchiya":
        emotion_type = 1
    elif emotion == "normal" and actor == "tsuchiya":
        emotion_type = 2
    elif emotion == "angry" and actor == "fujitou":
        emotion_type = 3
    elif emotion == "happy" and actor == "fujitou":
        emotion_type = 4
    elif emotion == "normal" and actor == "fujitou":
        emotion_type = 5
    elif emotion == "angry" and actor == "uemura":
        emotion_type = 6
    elif emotion == "happy" and actor == "uemura":
        emotion_type = 7
    elif emotion == "normal" and actor == "uemura":
        emotion_type = 8

    return emotion_type


def main(cfg: DictConfig):
    """Make PaCMAP plot in 2D space."""
    feat_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.feat_dir)
    feat_list = []
    emotion_type = []
    for actor in cfg.actor:
        for emotion in cfg.emotion:
            feats_files = glob.glob(feat_dir + actor + "/" + f"{actor}_{emotion}_*.npy")
            for feats in feats_files:
                xvector = np.load(feats)
                xvector = np.expand_dims(xvector, axis=0)
                feat_list.append(xvector)
                emotion_type.append(get_emotion_type(actor, emotion))

    feat_array = np.concatenate(feat_list)
    mapper = pacmap.PaCMAP(n_components=2, n_neighbors=15)
    embedding = mapper.fit_transform(feat_array)
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]
    for n in np.unique(emotion_type):
        if n == 0:  # angry tsuchiya
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="tsuchiya-angry",
                s=10,
            )
        elif n == 1:  # happy tsuchiya
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="tsuchiya-happy",
                s=10,
            )
        elif n == 2:  # normal tsuchiya
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="tsuchiya-normal",
                s=10,
            )
        elif n == 3:  # angry fujitou
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="fujitou-angry",
                s=10,
            )
        elif n == 4:  # happy fujitou
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="fujitou-happy",
                s=10,
            )
        elif n == 5:  # normal fujitou
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="fujitou-normal",
                s=10,
            )
        elif n == 6:  # angry uemura
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="uemura-angry",
                s=10,
            )
        elif n == 7:  # happy uemura
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="uemura-happy",
                s=10,
            )
        elif n == 8:  # normal uemura
            _ = plt.scatter(
                embedding_x[emotion_type == n],
                embedding_y[emotion_type == n],
                label="uemura-normal",
                s=10,
            )

    plt.grid()
    _ = plt.legend()
    plt.tight_layout()
    plt.savefig("PaCMAP_all.png")
    plt.show()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
