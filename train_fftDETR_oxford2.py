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
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFT를 위한 학습 가능한 필터
        self.freq_filter = nn.Parameter(torch.randn(n_heads, self.head_dim))
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Q, K, V 프로젝션
        Q = self.query_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.key_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.value_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # FFT 기반 어텐션 계산
        attention_output = self.fft_attention(Q, K, V, mask)
        
        # 출력 프로젝션
        output = self.out_proj(attention_output)
        return output
    
    def fft_attention(self, Q, K, V, mask=None):
        batch_size, seq_len, n_heads, head_dim = Q.shape
        
        # 복소수 변환을 위해 실수부만 사용
        Q_complex = Q.to(torch.complex64)
        K_complex = K.to(torch.complex64)
        
        # FFT 적용
        Q_fft = torch.fft.fft(Q_complex, dim=1)
        K_fft = torch.fft.fft(K_complex, dim=1)
        
        # 주파수 도메인에서 어텐션 계산
        freq_attention = torch.zeros_like(Q_fft)
        for h in range(n_heads):
            # 학습 가능한 주파수 필터 적용
            filter_weight = torch.sigmoid(self.freq_filter[h]).unsqueeze(0).unsqueeze(0)
            freq_attention[:, :, h, :] = Q_fft[:, :, h, :] * K_fft[:, :, h, :].conj() * filter_weight
        
        # IFFT로 시간 도메인으로 복원
        attention_real = torch.fft.ifft(freq_attention, dim=1).real
        
        # Value와 결합
        output = torch.sum(attention_real * V, dim=2)
        
        # 정규화
        output = output / math.sqrt(head_dim)
        
        return output

class FFTTransformerBlock(nn.Module):
    """FFT 어텐션을 사용하는 트랜스포머 블록"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(FFTTransformerBlock, self).__init__()
        self.attention = FFTAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class FFT_DETR(nn.Module):
    """FFT 어텐션을 사용하는 DETR 모델"""
    def __init__(self, num_classes, d_model=256, n_heads=8, num_encoder_layers=6, 
                 num_decoder_layers=6, num_queries=100):
        super(FFT_DETR, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        
        # CNN 백본 (ResNet-50 사용)
        resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 특징 차원 조정
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # 위치 인코딩
        self.pos_embedding = PositionalEncoding2D(d_model)
        
        # FFT 트랜스포머 인코더
        self.encoder_layers = nn.ModuleList([
            FFTTransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(num_encoder_layers)
        ])
        
        # FFT 트랜스포머 디코더
        self.decoder_layers = nn.ModuleList([
            FFTTransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(num_decoder_layers)
        ])
        
        # 객체 쿼리
        self.query_pos = nn.Parameter(torch.randn(num_queries, d_model))
        
        # 예측 헤드
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(d_model, 4)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # CNN 백본으로 특징 추출
        features = self.backbone(x)
        features = self.input_proj(features)
        
        # 특징맵을 시퀀스로 변환
        h, w = features.shape[-2:]
        features_flat = features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        
        # 위치 인코딩 추가
        pos_embed = self.pos_embedding(features).flatten(2).permute(2, 0, 1)
        features_flat = features_flat + pos_embed
        
        # 인코더 통과
        memory = features_flat.permute(1, 0, 2)  # (B, H*W, C)
        for layer in self.encoder_layers:
            memory = layer(memory)
        
        # 디코더에서 메모리와 쿼리 결합
        queries = self.query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention을 위해 메모리와 쿼리를 결합
        decoder_input = torch.cat([queries, memory], dim=1)
        
        decoder_output = decoder_input
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output)
        
        # 쿼리 부분만 추출
        decoder_output = decoder_output[:, :self.num_queries, :]
        
        # 예측
        class_logits = self.class_head(decoder_output)
        bbox_pred = self.bbox_head(decoder_output).sigmoid()
        
        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_pred
        }

class PositionalEncoding2D(nn.Module):
    """2D 위치 인코딩"""
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Y 방향 위치 인코딩
        y_pos = torch.arange(height, dtype=torch.float32, device=x.device)
        y_pos = y_pos.unsqueeze(1).repeat(1, width)
        
        # X 방향 위치 인코딩
        x_pos = torch.arange(width, dtype=torch.float32, device=x.device)
        x_pos = x_pos.unsqueeze(0).repeat(height, 1)
        
        # 사인/코사인 인코딩
        div_term = torch.exp(torch.arange(0, self.d_model//2, 2, dtype=torch.float32, device=x.device) *
                           -(math.log(10000.0) / (self.d_model//2)))
        
        pos_embed = torch.zeros(height, width, self.d_model, device=x.device)
        
        # 차원 검사 및 조정
        y_embed_dim = min(len(div_term), self.d_model//4)
        x_embed_dim = min(len(div_term), self.d_model//4)
        
        if y_embed_dim > 0:
            pos_embed[:, :, 0::4][:, :, :y_embed_dim] = torch.sin(y_pos.unsqueeze(-1) * div_term[:y_embed_dim])
            pos_embed[:, :, 1::4][:, :, :y_embed_dim] = torch.cos(y_pos.unsqueeze(-1) * div_term[:y_embed_dim])
        
        if x_embed_dim > 0:
            pos_embed[:, :, 2::4][:, :, :x_embed_dim] = torch.sin(x_pos.unsqueeze(-1) * div_term[:x_embed_dim])
            pos_embed[:, :, 3::4][:, :, :x_embed_dim] = torch.cos(x_pos.unsqueeze(-1) * div_term[:x_embed_dim])
        
        return pos_embed.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

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
        try:
            tree = ET.parse(self.annotation_files[idx])
            root = tree.getroot()
            
            boxes = []
            labels = []
            
            for obj in root.findall('object'):
                # 바운딩 박스 좌표
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
                    labels.append(1)  # 모든 객체를 pet으로 분류
        
        except Exception as e:
            print(f"어노테이션 파싱 오류: {e}")
            boxes = []
            labels = []
        
        # 최소 1개의 박스는 있어야 함
        if len(boxes) == 0:
            boxes.append([0.5, 0.5, 1.0, 1.0])  # 전체 이미지를 커버하는 박스
            labels.append(1)
        
        # 패딩 (최대 100개 객체로 고정)
        while len(boxes) < 100:
            boxes.append([0, 0, 0, 0])
            labels.append(0)  # 배경 클래스
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(boxes[:100], dtype=torch.float32), torch.tensor(labels[:100], dtype=torch.long)

def download_oxford_pet_dataset(data_dir):
    """Oxford Pet 데이터셋 다운로드"""
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print("Oxford Pet 데이터셋이 이미 존재합니다.")
        return
    
    print("Oxford Pet 데이터셋을 다운로드 중...")
    os.makedirs(data_dir, exist_ok=True)
    
    # 이미지 다운로드
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
            tar.extractall(data_dir)
        
        with tarfile.open(annotations_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        # 압축 파일 삭제
        os.remove(images_path)
        os.remove(annotations_path)
        
        print("Oxford Pet 데이터셋 다운로드 완료!")
        
    except Exception as e:
        print(f"다운로드 오류: {e}")
        print("수동으로 데이터셋을 다운로드해주세요.")

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
        print("데이터셋이 비어있습니다. 데이터셋 경로를 확인해주세요.")
        return
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, 
                          collate_fn=collate_fn, num_workers=0)
    
    # 모델 생성
    model = FFT_DETR(num_classes=1, d_model=256, n_heads=8).to(device)
    
    # 손실 함수 및 옵티마이저
    weight_dict = {'class': 1.0, 'bbox': 5.0}
    criterion = DETRLoss(num_classes=1, weight_dict=weight_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 학습 루프
    num_epochs = 20
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
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
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
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'fft_detr_epoch_{epoch+1}.pth')
            print(f'모델 저장: fft_detr_epoch_{epoch+1}.pth')
    
    print("학습 완료!")
    
    # 최종 모델 저장
    torch.save(model.state_dict(), 'fft_detr_final.pth')
    print("최종 모델 저장: fft_detr_final.pth")

if __name__ == "__main__":
    train_model()
