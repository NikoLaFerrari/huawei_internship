import torch  
import torch.nn as nn  
from typing import Optional, List  
import numpy as np  
  
class SkyLadder(nn.Module):  
    """  
    SkyLadder: Context Window Scheduling for Efficient LLM Pretraining  
      
    Implements progressive context window expansion during training with  
    fragment-based attention masking.  
    """  
      
    def __init__(self, config):  
        super().__init__()  
        self.config = config  
        self.gpt_model = GPT(config)  # Uses the existing GPT implementation  
        self.current_iter = 0  
          
        # Scheduling parameters  
        self.init_mask_length = getattr(config, 'init_mask_length', 512)  
        self.final_mask_length = config.block_size  
        self.schedule_type = getattr(config, 'schedule_type', 'linear')  
        self.alpha = getattr(config, 'alpha', 8)  # 1/alpha is the rate  
          
    def calculate_current_mask_length(self, iter_num: int) -> int:  
        """Calculate current context window size based on iteration."""  
        changing_point = self.final_mask_length * self.alpha  
          
        if iter_num >= changing_point:  
            return self.final_mask_length  
          
        if self.schedule_type == 'linear':  
            return self._linear_schedule(iter_num, changing_point)  
        elif self.schedule_type == 'sin':  
            return self._sin_schedule(iter_num, changing_point)  
        elif self.schedule_type == 'exp':  
            return self._exp_schedule(iter_num, changing_point)  
        else:  
            return self._linear_schedule(iter_num, changing_point)  
      
    def _linear_schedule(self, iter_num: int, changing_point: int) -> int:  
        """Linear context window scheduling."""  
        progress = iter_num / changing_point  
        curr_length = self.init_mask_length + int(  
            (self.final_mask_length - self.init_mask_length) * progress  
        )  
        return curr_length  
      
    def _sin_schedule(self, iter_num: int, changing_point: int) -> int:  
        """Sinusoidal context window scheduling."""  
        progress = iter_num / changing_point  
        sin_progress = (1 - np.cos(np.pi * progress)) / 2  
        curr_length = self.init_mask_length + int(  
            (self.final_mask_length - self.init_mask_length) * sin_progress  
        )  
        return curr_length  
      
    def _exp_schedule(self, iter_num: int, changing_point: int) -> int:  
        """Exponential context window scheduling."""  
        progress = iter_num / changing_point  
        exp_progress = (np.exp(progress) - 1) / (np.exp(1) - 1)  
        curr_length = self.init_mask_length + int(  
            (self.final_mask_length - self.init_mask_length) * exp_progress  
        )  
        return curr_length  
      
    def get_fragment_lens_fixed_length(self, chunk: torch.Tensor, fixed_length: int):  
        """Calculate fragment lengths for variable-length attention."""  
        chunk_len = len(chunk)  
        if fixed_length >= chunk_len:  
            return [chunk_len], 1  
          
        filtered_indices = np.arange(fixed_length, chunk_len, fixed_length) - 1  
          
        if filtered_indices.size > 0:  
            fragment_lengths = []  
            prev = 0  
            for idx in filtered_indices:  
                fragment_lengths.append(int(idx - prev + 1))  
                prev = idx + 1  
            if prev < chunk_len:  
                fragment_lengths.append(chunk_len - prev)  
        else:  
            fragment_lengths = [chunk_len]  
          
        return fragment_lengths, len(fragment_lengths)  
      
    def forward(self, input_ids: torch.Tensor, iter_num: Optional[int] = None):  
        """Forward pass with context window scheduling."""  
        if iter_num is not None:  
            self.current_iter = iter_num  
          
        # Calculate current mask length based on training iteration  
        curr_mask_length = self.calculate_current_mask_length(self.current_iter)  
          
        # Generate fragment information for variable-length attention  
        batch_size = input_ids.size(0)  
        fragment_lens_batch = []  
        fragment_nums_batch = []  
          
        for i in range(batch_size):  
            sequence = input_ids[i]  
            fragment_lens, fragment_nums = self.get_fragment_lens_fixed_length(  
                sequence, curr_mask_length  
            )  
            fragment_lens_batch.append(fragment_lens)  
            fragment_nums_batch.append(fragment_nums)  
          
        # Call the underlying GPT model with fragment information  
        return self.gpt_model(  
            input_ids,   
            fragment_lens=fragment_lens_batch,   
            fragment_nums=fragment_nums_batch  
        )  
      
    def step(self):  
        """Increment iteration counter for scheduling."""  
        self.current_iter += 1
