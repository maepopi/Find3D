"""
Compatibility shim for torch_scatter using torch_geometric
Falls back to torch_geometric.utils.scatter when torch_scatter is not available
"""

import sys
import torch
from torch_geometric.utils import scatter as tg_scatter

# Try to import real torch_scatter first
try:
    import torch_scatter as _real_torch_scatter
    # If successful, use the real module
    segment_csr = _real_torch_scatter.segment_csr
except ImportError:
    # Fallback: implement segment_csr using torch_geometric
    def segment_csr(src, indptr, out_size=None, reduce='add'):
        """
        Segment CSR (from torch_scatter API) implemented using torch_geometric.utils.scatter
        
        Args:
            src: Tensor of shape (N, *feature_shape)
            indptr: 1D tensor with segment boundaries  
            out_size: Number of output segments
            reduce: 'sum', 'add', 'mean', 'min', 'max'
        
        Returns:
            Segmented tensor of shape (out_size, *feature_shape)
        """
        if out_size is None:
            out_size = len(indptr) - 1
        
        # Convert CSR indptr to segment indices
        index = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
        for i in range(len(indptr) - 1):
            index[indptr[i]:indptr[i + 1]] = i
        
        # Use torch_geometric's scatter
        return tg_scatter(src, index, dim=0, dim_size=out_size, reduce=reduce)


# Make torch_scatter available as a module
class TorchScatterCompat:
    """Compatibility wrapper for torch_scatter module"""
    
    def segment_csr(self, src, indptr, out_size=None, reduce='add'):
        """Segment CSR wrapper"""
        return segment_csr(src, indptr, out_size, reduce)


# If torch_scatter wasn't found, inject our compat version
if 'torch_scatter' not in sys.modules:
    sys.modules['torch_scatter'] = TorchScatterCompat()
    torch_scatter_available = False
else:
    torch_scatter_available = True


def get_segment_csr():
    """Get the segment_csr function"""
    if torch_scatter_available:
        import torch_scatter
        return torch_scatter.segment_csr
    else:
        return segment_csr


__all__ = ['segment_csr', 'get_segment_csr', 'torch_scatter_available']
