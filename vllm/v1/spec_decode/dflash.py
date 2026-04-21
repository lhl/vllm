# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        self.runner = runner
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None
        self._draft_kv_cache_group_ids: tuple[int, ...] = ()
        self._draft_layer_names_by_gid: dict[int, tuple[str, ...]] = {}
        self._draft_block_sizes_by_gid: dict[int, int] = {}
        self._dflash_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._dflash_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._dflash_group_common_attn_metadata_by_gid: dict[
            int, CommonAttentionMetadata
        ] = {}

    @override
    def _raise_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        # Support for multimodal inputs has not been tested.
        pass

    def _use_torch_dflash_setup(self) -> bool:
        group_ids = self._draft_kv_cache_group_ids or (
            (0,) if self.block_size > 0 else ()
        )
        device_type = (
            self.device.type
            if isinstance(self.device, torch.device)
            else torch.device(self.device).type
        )
        return device_type == "cpu" or len(group_ids) > 1

    def _get_draft_group_ids(self) -> tuple[int, ...]:
        if self._draft_kv_cache_group_ids:
            return self._draft_kv_cache_group_ids
        if self.kv_cache_gid >= 0:
            return (self.kv_cache_gid,)
        return (0,)

    def _ensure_group_slot_mapping_buffers(self) -> None:
        for gid in self._get_draft_group_ids():
            self._dflash_context_slot_mapping_buffers.setdefault(
                gid,
                torch.zeros(
                    self.max_num_tokens,
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
            self._dflash_query_slot_mapping_buffers.setdefault(
                gid,
                torch.zeros(
                    self.max_query_tokens,
                    dtype=torch.int64,
                    device=self.device,
                ),
            )

    def _get_block_size_for_gid(self, gid: int) -> int:
        block_size = self._draft_block_sizes_by_gid.get(gid, self.block_size)
        assert block_size > 0, f"block_size has not been initialized for gid={gid}"
        return block_size

    def _get_block_table_for_gid(
        self,
        gid: int,
        cad: CommonAttentionMetadata,
    ) -> torch.Tensor:
        if gid == self.kv_cache_gid or self.runner is None:
            return cad.block_table_tensor
        return self.runner.input_batch.block_table[gid].get_device_tensor(cad.num_reqs)

    def _compute_slot_mapping(
        self,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        block_table_tensor: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        batch_size = query_start_loc.shape[0] - 1
        seq_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.long)
        req_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device, dtype=torch.long),
            seq_lens,
        )
        block_numbers = torch.div(
            positions.to(torch.long),
            block_size,
            rounding_mode="floor",
        ).clamp(max=block_table_tensor.shape[1] - 1)
        block_ids = block_table_tensor[req_ids, block_numbers].to(torch.long)
        return block_ids * block_size + positions.to(torch.long).remainder(block_size)

    def _build_group_common_attn_metadata(
        self,
        cad: CommonAttentionMetadata,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        num_query_total: int,
        num_query_per_req: int,
    ) -> CommonAttentionMetadata:
        return cad.replace(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            causal=False,
        )

    def _set_inputs_first_pass_torch(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req
        self._ensure_group_slot_mapping_buffers()
        self._dflash_group_common_attn_metadata_by_gid = {}

        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states

        query_start_loc = cad.query_start_loc
        if num_rejected_tokens_gpu is not None:
            num_rejected = num_rejected_tokens_gpu.to(torch.long)
        else:
            num_rejected = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        valid_query_ends = query_start_loc[1:].to(torch.long) - num_rejected
        last_pos = target_positions[valid_query_ends - 1]

        query_offsets = torch.arange(
            num_query_per_req, device=self.device, dtype=torch.int64
        )
        query_positions = last_pos.unsqueeze(1) + 1 + query_offsets.unsqueeze(0)
        query_positions_flat = query_positions.reshape(-1)

        self._context_positions_buffer[:num_context].copy_(target_positions)
        self.positions[:num_query_total].copy_(query_positions_flat)

        query_input_ids = self.input_ids[:num_query_total].view(
            batch_size, num_query_per_req
        )
        query_input_ids.fill_(self.parallel_drafting_token_id)
        query_input_ids[:, 0].copy_(next_token_ids)

        token_indices_to_sample = (
            torch.arange(batch_size, device=self.device, dtype=torch.int32).unsqueeze(1)
            * num_query_per_req
            + torch.arange(
                1,
                num_query_per_req,
                device=self.device,
                dtype=torch.int32,
            ).unsqueeze(0)
        ).reshape(-1)

        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req
        query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
            * num_query_per_req
        )

        effective_seq_lens = cad.seq_lens
        if num_rejected_tokens_gpu is not None:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
        new_seq_lens = effective_seq_lens + num_query_per_req

        for gid in self._get_draft_group_ids():
            block_table_tensor = self._get_block_table_for_gid(gid, cad)
            block_size = self._get_block_size_for_gid(gid)

            context_slot_mapping = self._compute_slot_mapping(
                target_positions,
                query_start_loc,
                block_table_tensor,
                block_size,
            )
            query_slot_mapping = self._compute_slot_mapping(
                query_positions_flat,
                new_query_start_loc,
                block_table_tensor,
                block_size,
            )

            self._dflash_context_slot_mapping_buffers[gid][:num_context].copy_(
                context_slot_mapping
            )
            self._dflash_query_slot_mapping_buffers[gid][:num_query_total].copy_(
                query_slot_mapping
            )

            self._dflash_group_common_attn_metadata_by_gid[gid] = (
                self._build_group_common_attn_metadata(
                    cad,
                    query_start_loc=new_query_start_loc,
                    query_start_loc_cpu=query_start_loc_cpu,
                    seq_lens=new_seq_lens,
                    num_query_total=num_query_total,
                    num_query_per_req=num_query_per_req,
                ).replace(
                    block_table_tensor=block_table_tensor,
                    slot_mapping=self._dflash_query_slot_mapping_buffers[gid][
                        :num_query_total
                    ],
                )
            )

        primary_gid = self._get_draft_group_ids()[0]
        return (
            num_query_total,
            token_indices_to_sample,
            self._dflash_group_common_attn_metadata_by_gid[primary_gid],
        )

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Store for build_model_inputs_first_pass to use
        self._dflash_num_context = num_context
        self._dflash_group_common_attn_metadata_by_gid = {}

        # We don't need to copy into a buffer here since the context preprocessing
        # does not run in a CUDA graph
        self._dflash_hidden_states = target_hidden_states

        if self._use_torch_dflash_setup():
            return self._set_inputs_first_pass_torch(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                cad=cad,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=False,  # Non-causal attention is required for DFlash
        )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        attention_groups: dict[tuple[int, str], AttentionGroup] = {}
        draft_layer_names_by_gid: dict[int, list[str]] = {}

        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_layer_names = sorted(
                self._draft_attn_layer_names & set(group.layer_names)
            )
            if not group_layer_names:
                continue
            draft_layer_names_by_gid[gid] = group_layer_names
            for layer_name in group_layer_names:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                key = (gid, attn_backend.full_cls_name())
                if key not in attention_groups:
                    layer_kv_cache_spec = group.kv_cache_spec
                    if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                        layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                            layer_name
                        ]
                    kernel_block_size = (
                        kernel_block_sizes[gid]
                        if kernel_block_sizes is not None
                        and gid < len(kernel_block_sizes)
                        else None
                    )
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[key] = attn_group
                else:
                    attention_groups[key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self._draft_layer_names_by_gid = {
            gid: tuple(layer_names)
            for gid, layer_names in draft_layer_names_by_gid.items()
        }
        self._draft_kv_cache_group_ids = tuple(sorted(self._draft_layer_names_by_gid))
        self.kv_cache_gid = (
            self._draft_kv_cache_group_ids[0] if self._draft_kv_cache_group_ids else -1
        )
        self._draft_block_sizes_by_gid = {}
        for gid in self._draft_kv_cache_group_ids:
            for attn_group in self.draft_attn_groups:
                if attn_group.kv_cache_group_id == gid:
                    self._draft_block_sizes_by_gid[gid] = (
                        attn_group.get_metadata_builder().kv_cache_spec.block_size
                    )
                    break

        self._ensure_group_slot_mapping_buffers()

        if self.kv_cache_gid >= 0:
            self.block_size = self._get_block_size_for_gid(self.kv_cache_gid)
            logger.debug(
                "Using DFlash drafting KV cache groups %s with primary block size %d",
                self._draft_kv_cache_group_ids,
                self.block_size,
            )

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written to separate
        # buffers by the kernel — no copy needed.
        num_context = self._dflash_num_context

        context_slot_mapping: torch.Tensor | dict[str, torch.Tensor] | None = None
        if self._draft_layer_names_by_gid and len(self._get_draft_group_ids()) > 1:
            context_slot_mapping = {}
            for gid, layer_names in self._draft_layer_names_by_gid.items():
                slot_mapping = self._dflash_context_slot_mapping_buffers[gid][
                    :num_context
                ]
                for layer_name in layer_names:
                    context_slot_mapping[layer_name] = slot_mapping
        elif self._get_draft_group_ids():
            primary_gid = self._get_draft_group_ids()[0]
            context_slot_mapping = self._dflash_context_slot_mapping_buffers.get(
                primary_gid, self._context_slot_mapping_buffer
            )[:num_context]

        # Pre-insert context KVs directly into cache
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            context_slot_mapping,
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group: list[object] = []
        per_layer: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            group_cad = self._dflash_group_common_attn_metadata_by_gid.get(
                attn_group.kv_cache_group_id, cad
            )
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=group_cad,
                draft_index=draft_index,
            )
            per_group.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = attn_metadata
        for layer_name, attn_metadata in per_layer.items():
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} does not have"
                " non-causal support, which is required for DFlash."
                " Consider using a different attention backend, such as FlashAttention."
            )
        return per_group, per_layer

    @override
    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(self._get_draft_group_ids()) <= 1:
            return super()._get_slot_mapping(num_tokens, slot_mapping)

        result: dict[str, torch.Tensor] = {}
        for gid, layer_names in self._draft_layer_names_by_gid.items():
            view = self._dflash_query_slot_mapping_buffers[gid][:num_tokens]
            for layer_name in layer_names:
                result[layer_name] = view
        return result

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
