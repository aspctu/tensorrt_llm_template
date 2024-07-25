from typing import Optional, List
import numpy as np
import tritonclient
import tritonclient.grpc.aio as grpcclient

class ModelInput:
    def __init__(
        self,
        prompt: str,
        request_id: int,
        max_tokens: int = 50,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        beam_width: int = 1,
        bad_words_list: Optional[List[str]] = None,
        stop_words_list: Optional[List[str]] = None,
        repetition_penalty: float = 1.0,
        ignore_eos: bool = False,
        stream: bool = True,
        eos_token_id: Optional[int] = None,
        len_penalty: float = 1.0,
        early_stopping: bool = False,
        min_length: int = 0,
        beam_search_diversity_rate: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        random_seed: Optional[int] = None,
        return_log_probs: bool = False,
        return_context_logits: bool = False,
        return_generation_logits: bool = False,
        lora_task_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.stream = stream
        self.request_id = request_id
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._beam_width = beam_width
        self._bad_words_list = [""] if bad_words_list is None else bad_words_list
        self._stop_words_list = [""] if stop_words_list is None else stop_words_list
        self._repetition_penalty = repetition_penalty
        self._eos_token_id = eos_token_id
        self._ignore_eos = ignore_eos
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._len_penalty = len_penalty
        self._early_stopping = early_stopping
        self._min_length = min_length
        self._beam_search_diversity_rate = beam_search_diversity_rate
        self._presence_penalty = presence_penalty
        self._frequency_penalty = frequency_penalty
        self._random_seed = random_seed
        self._return_log_probs = return_log_probs
        self._return_context_logits = return_context_logits
        self._return_generation_logits = return_generation_logits
        self._lora_task_id = lora_task_id

    def _prepare_grpc_tensor(
        self, name: str, input_data: np.ndarray
    ) -> grpcclient.InferInput:
        tensor = grpcclient.InferInput(
            name,
            input_data.shape,
            tritonclient.utils.np_to_triton_dtype(input_data.dtype),
        )
        tensor.set_data_from_numpy(input_data)
        return tensor

    def to_tensors(self):
        if self._eos_token_id is None and self._ignore_eos:
            raise ValueError("eos_token_id is required when ignore_eos is True")

        inputs = [
            self._prepare_grpc_tensor("input_ids", np.array([self._prompt], dtype=np.int32)),
            self._prepare_grpc_tensor("input_lengths", np.array([len(self._prompt)], dtype=np.int32)),
            self._prepare_grpc_tensor("request_output_len", np.array([self._max_tokens], dtype=np.int32)),
            self._prepare_grpc_tensor("beam_width", np.array([self._beam_width], dtype=np.int32)),
            self._prepare_grpc_tensor("temperature", np.array([self._temperature], dtype=np.float32)),
            self._prepare_grpc_tensor("runtime_top_k", np.array([self._top_k], dtype=np.int32)),
            self._prepare_grpc_tensor("runtime_top_p", np.array([self._top_p], dtype=np.float32)),
            self._prepare_grpc_tensor("len_penalty", np.array([self._len_penalty], dtype=np.float32)),
            self._prepare_grpc_tensor("repetition_penalty", np.array([self._repetition_penalty], dtype=np.float32)),
            self._prepare_grpc_tensor("min_length", np.array([self._min_length], dtype=np.int32)),
            self._prepare_grpc_tensor("presence_penalty", np.array([self._presence_penalty], dtype=np.float32)),
            self._prepare_grpc_tensor("frequency_penalty", np.array([self._frequency_penalty], dtype=np.float32)),
            self._prepare_grpc_tensor("early_stopping", np.array([self._early_stopping], dtype=bool)),
            self._prepare_grpc_tensor("beam_search_diversity_rate", np.array([self._beam_search_diversity_rate], dtype=np.float32)),
            self._prepare_grpc_tensor("return_log_probs", np.array([self._return_log_probs], dtype=bool)),
            self._prepare_grpc_tensor("return_context_logits", np.array([self._return_context_logits], dtype=bool)),
            self._prepare_grpc_tensor("return_generation_logits", np.array([self._return_generation_logits], dtype=bool)),
            self._prepare_grpc_tensor("streaming", np.array([self.stream], dtype=bool)),
        ]

        if self._random_seed is not None:
            inputs.append(self._prepare_grpc_tensor("random_seed", np.array([self._random_seed], dtype=np.uint64)))

        if self._bad_words_list:
            inputs.append(self._prepare_grpc_tensor("bad_words_list", np.array(self._bad_words_list, dtype=np.int32)))

        if self._stop_words_list:
            inputs.append(self._prepare_grpc_tensor("stop_words_list", np.array(self._stop_words_list, dtype=np.int32)))

        if not self._ignore_eos and self._eos_token_id is not None:
            inputs.append(self._prepare_grpc_tensor("end_id", np.array([self._eos_token_id], dtype=np.int32)))

        if self._lora_task_id is not None:
            inputs.append(self._prepare_grpc_tensor("lora_task_id", np.array([self._lora_task_id], dtype=np.uint64)))

        return inputs