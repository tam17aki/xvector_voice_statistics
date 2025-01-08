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


def main(cfg: DictConfig):
    """Make PaCMAP plot in 2D space."""
    feat_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.feat_dir)
    for actor in cfg.actor:
        feat_list = []
        emotion_type = []
        for emotion in cfg.emotion:
            feats_files = glob.glob(feat_dir + actor + "/" + f"{actor}_{emotion}_*.npy")
            for feats in feats_files:
                xvector = np.load(feats)
                xvector = np.expand_dims(xvector, axis=0)
                feat_list.append(xvector)
                if emotion == "angry":
                    emotion_type.append(0)
                elif emotion == "happy":
                    emotion_type.append(1)
                else:  # normal
                    emotion_type.append(2)

        feat_array = np.concatenate(feat_list)
        mapper = pacmap.PaCMAP(n_components=2, n_neighbors=15)
        embedding = mapper.fit_transform(feat_array)
        embedding_x = embedding[:, 0]
        embedding_y = embedding[:, 1]
        for n in np.unique(emotion_type):
            if n == 0:  # angry
                _ = plt.scatter(
                    embedding_x[emotion_type == n],
                    embedding_y[emotion_type == n],
                    label="angry",
                    s=10,
                )
            elif n == 1:  # happy
                _ = plt.scatter(
                    embedding_x[emotion_type == n],
                    embedding_y[emotion_type == n],
                    label="happy",
                    s=10,
                )
            else:  # normal
                _ = plt.scatter(
                    embedding_x[emotion_type == n],
                    embedding_y[emotion_type == n],
                    label="normal",
                    s=10,
                )

        plt.grid()
        _ = plt.legend(title=actor)
        plt.tight_layout()
        plt.savefig("PaCMAP_{}.png".format(actor))
        plt.show()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
