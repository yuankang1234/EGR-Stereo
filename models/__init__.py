from models.cfnet import CFNet_G, CFNet_GC
from models.cfnet_yk import CFNet_yk
from models.relu.cfnet_yk import CFNet_GC_yk
from models.relu.cfnet_relu_yk import CFNet_GC_Relu

from models.loss import model_loss

__models__ = {
    "gwcnet-g": CFNet_G,
    "gwcnet-gc": CFNet_GC,
    "cfnet-gc": CFNet_GC_yk,
    "gwcnet-gc-relu": CFNet_GC_Relu,
    "cfnet": CFNet_yk,
}
