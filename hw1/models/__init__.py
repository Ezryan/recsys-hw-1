"""Five own recommenders sharing one evaluation protocol."""

from models.base import BaseRecommender
from models.item_knn import ItemKNN
from models.ials import IALS
from models.ease import EASE
from models.fm import FM
from models.custom_ials import CustomIALS

__all__ = ["BaseRecommender", "ItemKNN", "IALS", "EASE", "FM", "CustomIALS"]
