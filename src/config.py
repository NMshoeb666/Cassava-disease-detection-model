import os
import platform
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

MAIN_DATA_DIR = os.path.join(RAW_DATA_DIR, 'cassava_main')
MAIN_IMAGES_DIR = os.path.join(MAIN_DATA_DIR, 'train_images')
MAIN_FOLDS_CSV = os.path.join(PROCESSED_DATA_DIR, 'cassava_5folds.csv')

CROSS_DATA_DIR = os.path.join(RAW_DATA_DIR, 'cassava_cross')
CROSS_META_CSV = os.path.join(PROCESSED_DATA_DIR, 'cassava_cross_meta.csv')

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 5
CLASS_NAMES = [
    'cbb',      # 0: Cassava Bacterial Blight
    'cbsd',     # 1: Cassava Brown Streak Disease
    'cgm',      # 2: Cassava Green Mottle
    'cmd',      # 3: Cassava Mosaic Disease
    'healthy'   # 4: Healthy
]

CLASS_MAP = {
    0: 'Cassava Bacterial Blight (CBB)',
    1: 'Cassava Brown Streak Disease (CBSD)',
    2: 'Cassava Green Mottle (CGM)',
    3: 'Cassava Mosaic Disease (CMD)',
    4: 'Healthy'
}

CROSS_CLASSES_SUBSET = [0, 3, 4]  # cbb, cmd, healthy

IMG_SIZE = 224          # 224x224 for fast training (can be overridden via CLI)
BATCH_SIZE = 16         # 16 or 32 with AMP and FlashAttention for high GPU utilization
# On Windows, PyTorch DataLoader requires 0 workers to avoid multiprocessing spawn hangs
NUM_WORKERS = 0 if platform.system() == 'Windows' else 2
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42

CB_BETA = 0.999
FOCAL_GAMMA = 2.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
