import torch
import torch.nn.functional as F
from attention import Attention_pooling
from torchvision import transforms
from PIL import Image

def get_score(sample_tensor, eval_tensor):
    """
    Đánh giá độ chênh lệch giữa eval_tensor và sample_tensor.

    Args:
        eval_tensor (torch.Tensor): Tensor đánh giá có kích thước [1, 1280].
        sample_tensor (torch.Tensor): Tensor mẫu có kích thước [1, 1280].

    Returns:
        float: Điểm đánh giá, giá trị nằm trong khoảng [0, 1], với 1 là giống 100%.
    """
    # Chuẩn hóa tensor để sử dụng trong khoảng cách cosine
    eval_tensor = F.normalize(eval_tensor, dim=1)
    sample_tensor = F.normalize(sample_tensor, dim=1)

    # Tính khoảng cách cosine
    cosine_similarity = torch.sum(eval_tensor * sample_tensor, dim=1)

    # Chuyển đổi từ độ tương đồng sang đánh giá (1 là giống nhất)
    score = ((cosine_similarity + 1) / 2) ** 2
    return score.item()

def evaluate_image(image_path):
    # Cố định kết quả qua các lần chạy
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # Khởi tạo mô hình
    ap = Attention_pooling()
    state_dict = torch.load(r'weights\1211_202056_MobileNetV2_Lite_CX2.pth', map_location=torch.device('cpu'), weights_only=True)
    ap.load_state_dict(state_dict['model'])

    # Định nghĩa transform cho ảnh
    transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sample_image_path = r"test_image_casia\sample_image_casia.jpg" # 1 ảnh mẫu để làm gốc

    # Đặt mô hình ở chế độ đánh giá
    ap.eval()

    sample_image = transform(Image.open(sample_image_path).convert("RGB")).unsqueeze(0)
    print(sample_image.shape)
    sample_output = ap(sample_image)

    eval_image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    eval_output = ap(eval_image)

    # Đánh giá ảnh đánh giá dựa trên ảnh mẫu
    score = get_score(sample_output, eval_output)
    print(f"Score for {image_path}: {score}")

    return round(score * 100, 2)