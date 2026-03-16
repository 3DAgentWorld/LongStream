import torch


class StreamSession:
    def __init__(
        self,
        model,
        mode: str,
        window_size: int = 5,
        keep_first_frame_anchor: bool = True,
    ):
        self.model = model
        self.core_model = getattr(model, "longstream", model)
        self.mode = mode
        self.window_size = window_size
        self.keep_first_frame_anchor = keep_first_frame_anchor

        if self.mode not in ["causal", "window"]:
            raise ValueError(f"Unsupported attention mode: {self.mode}")

        self.aggregator_kv_cache_depth = self.core_model.aggregator.depth
        self.use_camera_head = self.core_model.camera_head is not None
        if self.use_camera_head:
            self.camera_head_kv_cache_depth = self.core_model.camera_head.trunk_depth
            self.camera_head_iterations = 4
        else:
            self.camera_head_kv_cache_depth = 0
            self.camera_head_iterations = 0

        self.use_rel_pose_head = (
            hasattr(self.core_model, "rel_pose_head")
            and self.core_model.rel_pose_head is not None
        )
        if self.use_rel_pose_head:
            self.rel_pose_head_trunk_depth = self.core_model.rel_pose_head.trunk_depth
            self.rel_pose_head_iterations = 4

        self.clear()

    def _clear_predictions(self):
        self.sequence_predictions = {}
        self.scalar_predictions = {}

    def _update_predictions(self, predictions):
        sequence_keys = [
            "pose_enc",
            "rel_pose_enc",
            "world_points",
            "world_points_conf",
            "depth",
            "depth_conf",
        ]
        scalar_keys = ["predicted_scale_factor", "global_scale"]

        for k in sequence_keys:
            if k in predictions:
                self.sequence_predictions.setdefault(k, []).append(
                    predictions[k].detach().cpu()
                )

        for k in scalar_keys:
            if k in predictions:
                value = predictions[k]
                self.scalar_predictions[k] = (
                    value.detach().cpu() if isinstance(value, torch.Tensor) else value
                )

    def _clear_cache(self):
        self.aggregator_kv_cache_list = [
            [None, None] for _ in range(self.aggregator_kv_cache_depth)
        ]
        if self.use_camera_head:
            self.camera_head_kv_cache_list = [
                [[None, None] for _ in range(self.camera_head_kv_cache_depth)]
                for _ in range(self.camera_head_iterations)
            ]
        else:
            self.camera_head_kv_cache_list = None
        if self.use_rel_pose_head:
            self.rel_pose_kv_cache_list = [
                [[None, None] for _ in range(self.rel_pose_head_trunk_depth)]
                for _ in range(self.rel_pose_head_iterations)
            ]
        else:
            self.rel_pose_kv_cache_list = None

    def _update_cache(
        self, aggregator_kv_cache_list, camera_head_kv_cache_list, frame_hw
    ):
        if self.mode == "causal":
            self.aggregator_kv_cache_list = aggregator_kv_cache_list
            if self.use_camera_head:
                self.camera_head_kv_cache_list = camera_head_kv_cache_list
            return

        if self.mode == "window":
            h, w = frame_hw
            P = (
                h
                * w
                // self.core_model.aggregator.patch_size
                // self.core_model.aggregator.patch_size
                + self.core_model.aggregator.patch_start_idx
            )

            for k in range(2):
                for i in range(self.aggregator_kv_cache_depth):
                    cache_size = aggregator_kv_cache_list[i][k].size(2)
                    if self.keep_first_frame_anchor:
                        if cache_size <= P:
                            self.aggregator_kv_cache_list[i][
                                k
                            ] = aggregator_kv_cache_list[i][k].contiguous()
                        elif cache_size <= self.window_size * P:
                            self.aggregator_kv_cache_list[i][
                                k
                            ] = aggregator_kv_cache_list[i][k].contiguous()
                        else:
                            anchor = aggregator_kv_cache_list[i][k][:, :, :P]
                            recent_start = cache_size - (self.window_size - 1) * P
                            recent = aggregator_kv_cache_list[i][k][:, :, recent_start:]
                            self.aggregator_kv_cache_list[i][k] = torch.cat(
                                [anchor, recent], dim=2
                            ).contiguous()
                    else:
                        start_idx = max(0, cache_size - self.window_size * P)
                        self.aggregator_kv_cache_list[i][k] = aggregator_kv_cache_list[
                            i
                        ][k][:, :, start_idx:].contiguous()

            if camera_head_kv_cache_list is not None:
                for k in range(2):
                    for i in range(self.camera_head_iterations):
                        for j in range(self.camera_head_kv_cache_depth):
                            cache_size = camera_head_kv_cache_list[i][j][k].size(2)
                            if self.keep_first_frame_anchor:
                                if cache_size <= 1:
                                    self.camera_head_kv_cache_list[i][j][
                                        k
                                    ] = camera_head_kv_cache_list[i][j][k].contiguous()
                                elif cache_size <= self.window_size:
                                    self.camera_head_kv_cache_list[i][j][
                                        k
                                    ] = camera_head_kv_cache_list[i][j][k].contiguous()
                                else:
                                    anchor = camera_head_kv_cache_list[i][j][k][
                                        :, :, :1
                                    ]
                                    recent_start = cache_size - (self.window_size - 1)
                                    recent = camera_head_kv_cache_list[i][j][k][
                                        :, :, recent_start:
                                    ]
                                    self.camera_head_kv_cache_list[i][j][k] = torch.cat(
                                        [anchor, recent], dim=2
                                    ).contiguous()
                            else:
                                start_idx = max(0, cache_size - self.window_size)
                                self.camera_head_kv_cache_list[i][j][
                                    k
                                ] = camera_head_kv_cache_list[i][j][k][
                                    :, :, start_idx:
                                ].contiguous()
            return

        raise ValueError(f"Unsupported attention mode: {self.mode}")

    def _get_cache(self):
        return self.aggregator_kv_cache_list, self.camera_head_kv_cache_list

    def get_all_predictions(self):
        predictions = {}
        for key, chunks in self.sequence_predictions.items():
            if not chunks:
                continue
            predictions[key] = (
                chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
            )
        predictions.update(self.scalar_predictions)
        return predictions

    def get_last_prediction(self):
        last_predictions = {}
        keys_to_extract = [
            "pose_enc",
            "rel_pose_enc",
            "world_points",
            "world_points_conf",
            "depth",
            "depth_conf",
            "predicted_scale_factor",
        ]
        for k in keys_to_extract:
            if k in self.sequence_predictions and self.sequence_predictions[k]:
                last_predictions[k] = self.sequence_predictions[k][-1][:, -1:]
            elif k in self.scalar_predictions:
                last_predictions[k] = self.scalar_predictions[k]
        return last_predictions

    def clear(self):
        self._clear_predictions()
        self._clear_cache()
        if self.use_rel_pose_head:
            if hasattr(self.core_model.rel_pose_head, "_keyframe_tokens_cache"):
                self.core_model.rel_pose_head._keyframe_tokens_cache = {}
            if hasattr(self.core_model.rel_pose_head, "_current_frame_id"):
                self.core_model.rel_pose_head._current_frame_id = 0
            if hasattr(self.core_model.rel_pose_head, "_frame_info"):
                self.core_model.rel_pose_head._frame_info = []

    def clear_cache_only(self):
        self._clear_cache()
        if self.use_rel_pose_head:
            if hasattr(self.core_model.rel_pose_head, "_keyframe_tokens_cache"):
                self.core_model.rel_pose_head._keyframe_tokens_cache = {}
            if hasattr(self.core_model.rel_pose_head, "_current_frame_id"):
                self.core_model.rel_pose_head._current_frame_id = 0
            if hasattr(self.core_model.rel_pose_head, "_frame_info"):
                self.core_model.rel_pose_head._frame_info = []

    def forward_stream(
        self, images, is_keyframe=None, keyframe_indices=None, record: bool = True
    ):
        aggregator_kv_cache_list, camera_head_kv_cache_list = self._get_cache()

        rel_pose_inputs = None
        if (
            self.use_rel_pose_head
            and is_keyframe is not None
            and keyframe_indices is not None
        ):
            rel_pose_inputs = {
                "is_keyframe": is_keyframe,
                "keyframe_indices": keyframe_indices,
                "kv_cache_list": self.rel_pose_kv_cache_list,
            }

        outputs = self.model(
            images=images,
            mode=self.mode,
            aggregator_kv_cache_list=aggregator_kv_cache_list,
            camera_head_kv_cache_list=camera_head_kv_cache_list,
            rel_pose_inputs=rel_pose_inputs,
            is_keyframe=is_keyframe,
        )

        if record:
            self._update_predictions(outputs)

        camera_head_kv_cache_list = outputs.get("camera_head_kv_cache_list", None)
        depth_hw = (
            outputs["depth"].shape[2:4] if "depth" in outputs else images.shape[-2:]
        )
        self._update_cache(
            outputs["aggregator_kv_cache_list"], camera_head_kv_cache_list, depth_hw
        )

        if self.use_rel_pose_head and "rel_pose_kv_cache_list" in outputs:
            rel_pose_kv_cache = outputs["rel_pose_kv_cache_list"]
            if self.mode == "causal":
                self.rel_pose_kv_cache_list = rel_pose_kv_cache
            elif self.mode == "window":
                for k in range(2):
                    for i in range(self.rel_pose_head_iterations):
                        for j in range(self.rel_pose_head_trunk_depth):
                            if rel_pose_kv_cache[i][j][k] is None:
                                continue
                            cache_len = rel_pose_kv_cache[i][j][k].size(2)
                            if self.keep_first_frame_anchor:
                                if cache_len <= 1:
                                    self.rel_pose_kv_cache_list[i][j][
                                        k
                                    ] = rel_pose_kv_cache[i][j][k].contiguous()
                                elif cache_len <= self.window_size:
                                    self.rel_pose_kv_cache_list[i][j][
                                        k
                                    ] = rel_pose_kv_cache[i][j][k].contiguous()
                                else:
                                    anchor = rel_pose_kv_cache[i][j][k][:, :, :1]
                                    recent_start = cache_len - (self.window_size - 1)
                                    recent = rel_pose_kv_cache[i][j][k][
                                        :, :, recent_start:
                                    ]
                                    self.rel_pose_kv_cache_list[i][j][k] = torch.cat(
                                        [anchor, recent], dim=2
                                    ).contiguous()
                            else:
                                start_idx = max(0, cache_len - self.window_size)
                                self.rel_pose_kv_cache_list[i][j][
                                    k
                                ] = rel_pose_kv_cache[i][j][k][
                                    :, :, start_idx:
                                ].contiguous()

        return outputs
