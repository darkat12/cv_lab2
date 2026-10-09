import os, glob, torch, numpy as np, imageio.v3 as iio
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

os.makedirs("frames_detection", exist_ok=True)
os.makedirs("frames_segmentation", exist_ok=True)

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()
categories = weights.meta["categories"]

frame_paths = sorted(glob.glob("frames/*.png"))
print("Найдено кадров:", len(frame_paths))

for i, path in enumerate(frame_paths):
    print(f"Обработка {i+1}/{len(frame_paths)}...")
    img = Image.open(path).convert("RGB")
    tensor = F.to_tensor(img)
    with torch.no_grad():
        pred = model([tensor])[0]
    keep = pred["scores"] > 0.5
    boxes = pred["boxes"][keep]
    labels = pred["labels"][keep]
    scores = pred["scores"][keep]
    masks = pred["masks"][keep]

    img_det = img.copy()
    draw = ImageDraw.Draw(img_det)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        name = categories[label]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1-12)), f"{name} {score:.2f}", fill="red")
    img_det.save(f"frames_detection/{os.path.basename(path)}")

    img_seg = np.array(img).copy()
    if len(masks) > 0:
        masks_np = masks.squeeze(1).cpu().numpy() > 0.5
        rng = np.random.default_rng(42)
        for mask in masks_np:
            color = rng.integers(0, 255, 3)
            img_seg[mask] = (0.5 * img_seg[mask] + 0.5 * color).astype(np.uint8)
    Image.fromarray(img_seg).save(f"frames_segmentation/{os.path.basename(path)}")

print("Обработка завершена! Собираем GIF...")
for folder, gif_name in [("frames_detection", "detection.gif"), ("frames_segmentation", "segmentation.gif")]:
    imgs = [iio.imread(p) for p in sorted(glob.glob(f"{folder}/*.png"))]
    iio.imwrite(gif_name, imgs, duration=333)
    print(f"GIF сохранен: {gif_name}")

print("ВСЁ ГОТОВО!")
