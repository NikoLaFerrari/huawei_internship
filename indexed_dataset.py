import os
import shutil
import struct                                                                                                                                
import glob
import re
from enum import Enum                                                                                                                        
from functools import lru_cache                                                                                                              
from abc import ABC, abstractmethod                                                                                                          
from itertools import accumulate
from types import TracebackType                                                                                                              
from typing import Optional, Tuple, Type, Union, List                                                                                        

import torch
import numpy
    
_INDEX_HEADER = b"MMIDIDX\x00\x00"                                                                                                           

                                                                                                                                             
def get_packed_indexed_dataset(data_prefix: str, filter_length: Optional[int] = None):                                                       
    index_dataset_name = f"{data_prefix}_packed_*_document*"                                                                                 
    names = glob.glob(index_dataset_name)                                                                                                    
    template = f"{data_prefix}_packed_(.*)_document(.*)"                                                                                     
    all_field = set()                                                                                                                        
    for name in names:                                                                                                                       
        fields = re.match(template, name)
        all_field.add(fields.group(1))
    packed_dataset = dict()

    for field in all_field:                                                                                                                  
        # We only do filter for input_ids when filter_length is specified                                                                    
        max_len = filter_length if filter_length and field == 'input_ids' else None                                                          
        packed_dataset[field] = IndexedDataset(f"{data_prefix}_packed_{field}_document", max_len=max_len)                                    
                 
    if filter_length:
        filter_mask = packed_dataset['input_ids'].get_filter_mask()                                                                          
        for field in packed_dataset:                                                                                                         
            packed_dataset[field].do_filter(filter_mask)                                                                                     
                 
    combine_dataset = CombinedDataset(packed_dataset)                                                                                        
    return combine_dataset
