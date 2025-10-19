from .aspp_head import ASPPHead
from .fcn_head import FCNHead
from .psp_head import PSPHead
from .uper_head import UPerHead
from .afficd_uper_head import AffiCDUPerHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'UPerHead', 'AffiCDUPerHead'
]
