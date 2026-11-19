import argparse
import sys
import os
import torch

from src.config import MAIN_FOLDS_CSV, CROSS_META_CSV, NUM_CLASSES, CLASS_NAMES, DEVICE, OUTPUT_DIR, IMG_SIZE
from src.train_cv import run_cross_validation
from src.evaluate_cross import evaluate_cross_dataset
from src.models.egwt import EGWT
from src.models.egwt_improved import EGWT_Improved
from src.models.egwt_pretrained import EGWT_Pretrained
from src.gradcam import GradCAM, overlay_cam_on_image
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Cassava Disease Detection & Generalization Pipeline")
    parser.add_argument("--mode", type=str, default="sanity", choices=["sanity", "train", "eval_cross", "gradcam"],
                        help="Execution mode: sanity (fast test), train (5-fold CV), eval_cross (India dataset), gradcam (visualization)")
    parser.add_argument("--model", type=str, default="pretrained", choices=["pretrained", "improved", "baseline"],
                        help="Model architecture: pretrained (EGWT-Net++ with ImageNet weights), improved (EGWT_Improved from scratch), or baseline (EGWT)")
    parser.add_argument("--attention_mode", type=str, default="spatial_se", choices=["spatial_se", "se_only", "none"],
                        help="Attention ablation mode for EGWT_Improved: spatial_se (Spatial->SE), se_only (SE only), none (backbone only)")
    parser.add_argument("--use_mixup", action="store_true",
                        help="Enable CutMix / MixUp batch augmentation with minority-class-weighted partner sampling")
    parser.add_argument("--fold", type=int, default=None, help="Train a specific fold (0-4), or None for all 5 folds")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16 for fast GPU training)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="cb_focal", choices=["cb_focal", "ce"], help="Loss function")
    parser.add_argument("--sampler", type=str, default="balanced", choices=["balanced", "random"], help="Sampler type")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to saved model checkpoint for eval or gradcam")
    parser.add_argument("--image_path", type=str, default=None, help="Path to leaf image for Grad-CAM generation")
    
    args = parser.parse_args()
    
    if args.mode == "sanity":
        import run_sanity_check
    elif args.mode == "train":
        use_cb = (args.loss == "cb_focal")
        use_bal = (args.sampler == "balanced")
        run_cross_validation(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_cb_loss=use_cb,
            use_balanced_sampler=use_bal,
            specific_fold=args.fold,
            model_type=args.model,
            attention_mode=args.attention_mode,
            use_mixup=args.use_mixup
        )
    elif args.mode == "eval_cross":
        if args.model == "pretrained":
            model = EGWT_Pretrained(num_classes=NUM_CLASSES, pretrained=False, attention_mode=args.attention_mode).to(DEVICE)
        elif args.model == "improved":
            model = EGWT_Improved(num_classes=NUM_CLASSES, attention_mode=args.attention_mode).to(DEVICE)
        else:
            model = EGWT(num_classes=NUM_CLASSES).to(DEVICE)
            
        if args.checkpoint and os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
            print(f"Loaded checkpoint from: {args.checkpoint}")
        else:
            print("Warning: No checkpoint provided or not found, using randomly initialized model.")
        evaluate_cross_dataset(model, cross_csv=CROSS_META_CSV, device=DEVICE)
    elif args.mode == "gradcam":
        if not args.image_path or not os.path.exists(args.image_path):
            print("Error: --image_path is required and must exist for gradcam mode.")
            return
        if args.model == "pretrained":
            model = EGWT_Pretrained(num_classes=NUM_CLASSES, pretrained=False, attention_mode=args.attention_mode).to(DEVICE)
        elif args.model == "improved":
            model = EGWT_Improved(num_classes=NUM_CLASSES, attention_mode=args.attention_mode).to(DEVICE)
        else:
            model = EGWT(num_classes=NUM_CLASSES).to(DEVICE)
            
        if args.checkpoint and os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        
        if args.model == "pretrained":
            target_layer = model.stage3_cnn[-1]
        elif args.model == "improved":
            target_layer = model.stage3_proj.conv_dw[3]
        else:
            target_layer = model.stage3_proj.conv_dw[3]
        cam_gen = GradCAM(model, target_layer)
        
        from src.dataset import get_transforms
        _, val_trans = get_transforms(IMG_SIZE)
        raw_pil = Image.open(args.image_path).convert("RGB")
        tensor_in = val_trans(raw_pil).unsqueeze(0).to(DEVICE)
        
        cam, pred_cls, logits = cam_gen.generate_cam(tensor_in)
        overlay_pil = overlay_cam_on_image(raw_pil, cam)
        
        out_save = os.path.join(OUTPUT_DIR, "gradcam_output.jpg")
        overlay_pil.save(out_save)
        print(f"Grad-CAM Heatmap saved to: {out_save} (Predicted: {CLASS_NAMES[pred_cls]})")

if __name__ == "__main__":
    main()
