import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np
import math
import os
import urllib.request
import tarfile
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# MPS 디바이스 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class FFTAttention(nn.Module):
    """FFT 기반 어텐션 메커니즘"""
    def __init__(self, d_model, n_heads=8):
        super(FFTAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # FFT를 위한 학습 가능한 필터
        self.freq_filter = nn.Parameter(torch.ones(self.n_heads) * 0.5)
        
    def forward(self, x, y=None, z=None, attn_mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # QKV 계산
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2), qkv)
        
        # FFT 기반 어텐션 계산
        attn_output = self.fft_attention(q, k, v)
        
        # 차원 재정렬 및 결합
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 출력 프로젝션
        return self.out_proj(attn_output)
    
    def fft_attention(self, q, k, v):
        batch_size, n_heads, seq_len, head_dim = q.shape
        
        outputs = []
        for h in range(n_heads):
            q_h = q[:, h, :, :].to(torch.complex64)  # (batch, seq, head_dim)
            k_h = k[:, h, :, :].to(torch.complex64)
            v_h = v[:, h, :, :]  # 실수로 유지
            
            # FFT 적용
            q_fft = torch.fft.fft(q_h, dim=1)
            k_fft = torch.fft.fft(k_h, dim=1)
            
            # 주파수 도메인에서 상관관계 계산
            attn_fft = q_fft * k_fft.conj()
            
            # 학습 가능한 필터 적용
            filter_weight = torch.sigmoid(self.freq_filter[h])
            attn_fft = attn_fft * filter_weight
            
            # IFFT로 시간 도메인으로 복원
            attn_weights = torch.fft.ifft(attn_fft, dim=1).real
            
            # 스케일링 및 소프트맥스
            attn_weights = F.softmax(attn_weights * self.scale, dim=1)
            
            # Value와 곱하기
            output_h = torch.einsum('bsi,bsi->bsi', attn_weights, v_h)
            outputs.append(output_h.unsqueeze(1))
        
        # 모든 헤드 결합
        return torch.cat(outputs, dim=1)

class PositionalEncoding2D(nn.Module):
    """2D 위치 인코딩 - 수정된 버전"""
    def __init__(self, d_model):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 간단한 위치 인코딩 생성
        pos_embed = torch.zeros(batch_size, self.d_model, height, width, device=x.device)
        
        # Y 위치 인코딩
        y_pos = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        y_pos = y_pos / height
        
        # X 위치 인코딩  
        x_pos = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        x_pos = x_pos / width
        
        # 사인/코사인 인코딩
        for i in range(self.d_model // 4):
            div_term = 10000 ** (2 * i / self.d_model)
            
            if 4 * i < self.d_model:
                pos_embed[:, 4*i, :, :] = torch.sin(y_pos.squeeze() * div_term)
            if 4 * i + 1 < self.d_model:
                pos_embed[:, 4*i + 1, :, :] = torch.cos(y_pos.squeeze() * div_term)
            if 4 * i + 2 < self.d_model:
                pos_embed[:, 4*i + 2, :, :] = torch.sin(x_pos.squeeze() * div_term)
            if 4 * i + 3 < self.d_model:
                pos_embed[:, 4*i + 3, :, :] = torch.cos(x_pos.squeeze() * div_term)
        
        return pos_embed

class FFT_DETR(nn.Module):
    """FFT 어텐션을 사용하는 DETR 모델"""
    def __init__(self, num_classes, d_model=256, n_heads=8, num_encoder_layers=2, 
                 num_decoder_layers=2, num_queries=25):
        super(FFT_DETR, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        
        # CNN 백본 - 더 가벼운 모델 사용
        backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1').features
        self.backbone = backbone
        
        # 특징 차원 조정 (MobileNetV2는 1280 채널 출력)
        self.input_proj = nn.Conv2d(1280, d_model, kernel_size=1)
        
        # 위치 인코딩
        self.pos_embedding = PositionalEncoding2D(d_model)
        
        # 표준 트랜스포머 인코더 (안정성을 위해)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 표준 트랜스포머 디코더
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 객체 쿼리
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 예측 헤드
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = nn.Linear(d_model, 4)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # CNN 백본으로 특징 추출
        features = self.backbone(x)
        features = self.input_proj(features)
        
        # 위치 인코딩 추가
        pos_embed = self.pos_embedding(features)
        features_with_pos = features + pos_embed
        
        # 특징맵을 시퀀스로 변환
        h, w = features.shape[-2:]
        features_flat = features_with_pos.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 인코더 통과
        memory = self.encoder(features_flat)
        
        # 디코더 통과
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        decoder_output = self.decoder(query_embed, memory)
        
        # 예측
        class_logits = self.class_head(decoder_output)
        bbox_pred = self.bbox_head(decoder_output).sigmoid()
        
        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_pred
        }

class OxfordPetDataset(Dataset):
    """Oxford Pet 데이터셋"""
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # 이미지와 어노테이션 경로
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations', 'xmls')
        
        # 파일 리스트 생성
        self.image_files = []
        self.annotation_files = []
        
        if os.path.exists(self.images_dir) and os.path.exists(self.annotations_dir):
            for filename in os.listdir(self.images_dir):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(self.images_dir, filename)
                    annotation_path = os.path.join(self.annotations_dir, 
                                                 filename.replace('.jpg', '.xml'))
                    
                    if os.path.exists(annotation_path):
                        self.image_files.append(image_path)
                        self.annotation_files.append(annotation_path)
        
        print(f"총 {len(self.image_files)}개의 이미지를 찾았습니다.")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image = Image.open(self.image_files[idx]).convert('RGB')
        
        # 어노테이션 파싱
        boxes = []
        labels = []
        
        try:
            tree = ET.parse(self.annotation_files[idx])
            root = tree.getroot()
            
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is not None:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # 정규화된 좌표로 변환
                    img_width, img_height = image.size
                    x_center = (xmin + xmax) / 2.0 / img_width
                    y_center = (ymin + ymax) / 2.0 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    boxes.append([x_center, y_center, width, height])
                    labels.append(1)  # pet 클래스
        
        except Exception as e:
            pass
        
        # 기본 박스 설정 (어노테이션이 없는 경우)
        if len(boxes) == 0:
            boxes.append([0.5, 0.5, 1.0, 1.0])
            labels.append(1)
        
        # 패딩 (25개로 고정)
        while len(boxes) < 25:
            boxes.append([0, 0, 0, 0])
            labels.append(0)  # 배경 클래스
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(boxes[:25], dtype=torch.float32), torch.tensor(labels[:25], dtype=torch.long)

def download_oxford_pet_dataset(data_dir):
    """Oxford Pet 데이터셋 다운로드"""
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print("Oxford Pet 데이터셋이 이미 존재합니다.")
        return
    
    print("Oxford Pet 데이터셋을 다운로드 중...")
    os.makedirs(data_dir, exist_ok=True)
    
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    try:
        # 이미지 파일 다운로드
        images_path = os.path.join(data_dir, "images.tar.gz")
        print("이미지 파일 다운로드 중...")
        urllib.request.urlretrieve(images_url, images_path)
        
        # 어노테이션 파일 다운로드
        annotations_path = os.path.join(data_dir, "annotations.tar.gz")
        print("어노테이션 파일 다운로드 중...")
        urllib.request.urlretrieve(annotations_url, annotations_path)
        
        # 압축 해제
        print("파일 압축 해제 중...")
        with tarfile.open(images_path, 'r:gz') as tar:
            tar.extractall(data_dir, filter='data')
        
        with tarfile.open(annotations_path, 'r:gz') as tar:
            tar.extractall(data_dir, filter='data')
        
        # 압축 파일 삭제
        os.remove(images_path)
        os.remove(annotations_path)
        
        print("Oxford Pet 데이터셋 다운로드 완료!")
        
    except Exception as e:
        print(f"다운로드 오류: {e}")

def collate_fn(batch):
    """배치 콜레이트 함수"""
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, 0)
    boxes = torch.stack(boxes, 0)
    labels = torch.stack(labels, 0)
    return images, boxes, labels

class DETRLoss(nn.Module):
    """DETR 손실 함수"""
    def __init__(self, num_classes, weight_dict):
        super(DETRLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.class_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.L1Loss()
        
    def forward(self, outputs, targets_boxes, targets_labels):
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # 분류 손실
        class_loss = self.class_loss(pred_logits.view(-1, self.num_classes + 1), 
                                   targets_labels.view(-1))
        
        # 바운딩 박스 손실 (유효한 객체에 대해서만)
        valid_mask = targets_labels > 0
        if valid_mask.sum() > 0:
            bbox_loss = self.bbox_loss(pred_boxes[valid_mask], targets_boxes[valid_mask])
        else:
            bbox_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        total_loss = (self.weight_dict['class'] * class_loss + 
                     self.weight_dict['bbox'] * bbox_loss)
        
        return {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'bbox_loss': bbox_loss
        }

def train_model():
    """모델 학습 함수"""
    # 데이터셋 다운로드
    data_dir = "./oxford_pet_data"
    download_oxford_pet_dataset(data_dir)
    
    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 및 데이터로더 생성
    dataset = OxfordPetDataset(data_dir, transform=transform)
    
    if len(dataset) == 0:
        print("데이터셋이 비어있습니다.")
        return
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, 
                          collate_fn=collate_fn, num_workers=0)
    
    # 모델 생성
    model = FFT_DETR(num_classes=1, d_model=256, n_heads=8, 
                     num_encoder_layers=2, num_decoder_layers=2, num_queries=25).to(device)
    
    # 손실 함수 및 옵티마이저
    weight_dict = {'class': 2.0, 'bbox': 5.0}
    criterion = DETRLoss(num_classes=1, weight_dict=weight_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 학습 루프
    num_epochs = 15
    model.train()
    
    print("FFT-DETR 모델 학습 시작...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, boxes, labels) in enumerate(dataloader):
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            try:
                # 순전파
                outputs = model(images)
                
                # 손실 계산
                loss_dict = criterion(outputs, boxes, labels)
                loss = loss_dict['total_loss']
                
                # 역전파
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], '
                          f'Loss: {loss.item():.4f}, '
                          f'Class Loss: {loss_dict["class_loss"].item():.4f}, '
                          f'BBox Loss: {loss_dict["bbox_loss"].item():.4f}')
                        
            except Exception as e:
                print(f"배치 {batch_idx}에서 오류 발생: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}] 완료, Average Loss: {avg_loss:.4f}')
        
        scheduler.step()
        
        # 모델 저장
        if (epoch + 1) % 3 == 0:
            torch.save(model.state_dict(), f'fft_detr_epoch_{epoch+1}.pth')
            print(f'모델 저장: fft_detr_epoch_{epoch+1}.pth')
    
    print("학습 완료!")
    
    # 최종 모델 저장
    torch.save(model.state_dict(), 'fft_detr_final.pth')
    print("최종 모델 저장: fft_detr_final.pth")

if __name__ == "__main__":
    train_model()
