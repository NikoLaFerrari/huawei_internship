import torch  
import torch.nn as nn  
from typing import Optional, List, Any, Dict  
import numpy as np  
from transformers import AutoModel, AutoTokenizer  
  
class GeneralSkyLadder(nn.Module):  
    """  
    Model-agnostic SkyLadder implementation that can wrap any transformer model  
    """  
      
    def __init__(self, model_name_or_path: str, **model_kwargs):  
        super().__init__()  
          
        # Load any HuggingFace model  
        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  
          
        # SkyLadder scheduling parameters  
        self.current_iter = 0  
        self.init_mask_length = 512  
        self.final_mask_length = getattr(self.model.config, 'max_position_embeddings', 4096)  
        self.schedule_type = 'linear'  # 'linear', 'sin', 'exp'  
        self.alpha = 8  # 1/alpha is the rate  
          
        # Store original forward method  
        self._original_forward = self.model.forward  
          
        # Replace model's forward with our scheduling-aware version  
        self.model.forward = self._skyladder_forward  
      
    def calculate_current_mask_length(self, iter_num: int) -> int:  
        """Calculate current context window size based on iteration"""  
        changing_point = self.final_mask_length * self.alpha  
          
        if iter_num >= changing_point:  
            return self.final_mask_length  
          
        progress = iter_num / changing_point  
          
        if self.schedule_type == 'linear':  
            curr_length = self.init_mask_length + int(  
                (self.final_mask_length - self.init_mask_length) * progress  
            )  
        elif self.schedule_type == 'sin':  
            sin_progress = (1 - np.cos(np.pi * progress)) / 2  
            curr_length = self.init_mask_length + int(  
                (self.final_mask_length - self.init_mask_length) * sin_progress  
            )  
        elif self.schedule_type == 'exp':  
            exp_progress = (np.exp(progress) - 1) / (np.exp(1) - 1)  
            curr_length = self.init_mask_length + int(  
                (self.final_mask_length - self.init_mask_length) * exp_progress  
            )  
        else:  
            curr_length = self.init_mask_length + int(  
                (self.final_mask_length - self.init_mask_length) * progress  
            )  
          
        return curr_length  
      
    def get_fragment_lens_fixed_length(self, sequence_length: int, fixed_length: int):  
        """Calculate fragment lengths for variable-length attention"""  
        if fixed_length >= sequence_length:  
            return [sequence_length], 1  
          
        filtered_indices = np.arange(fixed_length, sequence_length, fixed_length) - 1  
          
        if filtered_indices.size > 0:  
            fragment_lengths = []  
            prev = 0  
            for idx in filtered_indices:  
                fragment_lengths.append(int(idx - prev + 1))  
                prev = idx + 1  
            if prev < sequence_length:  
                fragment_lengths.append(sequence_length - prev)  
        else:  
            fragment_lengths = [sequence_length]  
          
        return fragment_lengths, len(fragment_lengths)  
      
    def create_fragment_attention_mask(self, input_ids: torch.Tensor, curr_mask_length: int):  
        """Create attention mask based on fragment lengths"""  
        batch_size, seq_len = input_ids.shape  
        attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)  
          
        for i in range(batch_size):  
            fragment_lens, _ = self.get_fragment_lens_fixed_length(seq_len, curr_mask_length)  
              
            # Create block diagonal attention pattern  
            start_idx = 0  
            for frag_len in fragment_lens:  
                end_idx = start_idx + frag_len  
                # Only attend within each fragment  
                attention_mask[i, start_idx:end_idx] = 1  
                start_idx = end_idx  
          
        return attention_mask  
      
    def _skyladder_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):  
        """Modified forward pass with SkyLadder scheduling"""  
        if self.training:  
            # Calculate current mask length based on training iteration  
            curr_mask_length = self.calculate_current_mask_length(self.current_iter)  
              
            # Create fragment-based attention mask  
            fragment_attention_mask = self.create_fragment_attention_mask(input_ids, curr_mask_length)  
              
            # Combine with existing attention mask if provided  
            if attention_mask is not None:  
                attention_mask = attention_mask * fragment_attention_mask  
            else:  
                attention_mask = fragment_attention_mask  
          
        # Call original forward method  
        return self._original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)  
      
    def forward(self, input_ids: torch.Tensor, iter_num: Optional[int] = None, **kwargs):  
        """Forward pass with optional iteration number update"""  
        if iter_num is not None:  
            self.current_iter = iter_num  
          
        return self.model(input_ids=input_ids, **kwargs)  
      
    def step(self):  
        """Increment iteration counter for scheduling"""  
        self.current_iter += 1  
      
    def set_schedule_params(self, init_length: int = None, final_length: int = None,   
                           schedule_type: str = None, alpha: int = None):  
        """Update scheduling parameters"""  
        if init_length is not None:  
            self.init_mask_length = init_length  
        if final_length is not None:  
            self.final_mask_length = final_length  
        if schedule_type is not None:  
            self.schedule_type = schedule_type  
        if alpha is not None:  
            self.alpha = alpha  
  
# Usage example for any model  
def create_skyladder_model(model_name: str, **model_kwargs):  
    """Factory function to create SkyLadder wrapper for any model"""  
    return GeneralSkyLadder(model_name, **model_kwargs)  
  
# Example usage:  
# qwen_skyladder = create_skyladder_model("Qwen/Qwen2.5-7B")  
# qwen3_skyladder = create_skyladder_model("Qwen/Qwen3-32B")
