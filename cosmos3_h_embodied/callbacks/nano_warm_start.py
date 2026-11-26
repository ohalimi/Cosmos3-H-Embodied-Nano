# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.utils.callback import Callback


class InitEmaFromNano(Callback):
    """The Nano checkpoint has regular weights only. Seed EMA after the warm load,
    before iteration-zero validation; a resumed training checkpoint keeps its own EMA.
    """

    def on_train_start(self, model, iteration: int = 0) -> None:
        if iteration == 0:
            model.net_ema_worker.copy_to(src_model=model.net, tgt_model=model.net_ema)
