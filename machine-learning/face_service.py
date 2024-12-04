from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import requests
import json
import os
from sklearn.metrics.pairwise import cosine_distances

# 特征提取相关的模型
class ImageRequest(BaseModel):
    filePath: str
    fileCode: str

class ImageResponse(BaseModel):
    fileType: int
    fileCode: str
    ndarray: Optional[List[float]] = None

class BatchImageRequest(BaseModel):
    images: List[ImageRequest]

class BatchImageResponse(BaseModel):
    results: List[ImageResponse]

# 聚类相关的模型
class ClusterRequest(BaseModel):
    ndarray: List[float]
    fileCode: str

class ClusterResponse(BaseModel):
    faces: str
    cluster: List[str]

class BatchClusterRequest(BaseModel):
    faces: List[ClusterRequest]

class BatchClusterResponse(BaseModel):
    results: List[ClusterResponse]

# 人脸服务
class FaceService:
    def __init__(self, 
                 api_url: str = "http://home.plantplanethome.com:3003/predict",
                 max_distance: float = 0.6,
                 min_faces: int = 2):
        self.api_url = api_url
        self.max_distance = max_distance
        self.min_faces = min_faces
    
    def get_face_embedding(self, image_path: str) -> Optional[List[float]]:
        """获取单个图片的人脸特征向量"""
        config = {
            "facial-recognition": {
                "detection": {
                    "modelName": "buffalo_l",
                    "options": {"minScore": 0.7}
                },
                "recognition": {
                    "modelName": "buffalo_l"
                }
            }
        }
        
        try:
            if not os.path.exists(image_path):
                print(f"图片不存在: {image_path}")
                return None
                
            with open(image_path, 'rb') as img_file:
                files = {
                    'image': ('image.jpg', img_file, 'image/jpeg'),
                    'entries': (None, json.dumps(config), 'application/json')
                }
                
                response = requests.post(self.api_url, files=files)
                response.raise_for_status()
                
                if response.status_code == 200:
                    result = response.json()
                    faces = result.get("facial-recognition", [])
                    if faces:
                        return faces[0]["embedding"]
                    return None
                    
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return None
            
        return None
    
    def cluster_faces(self, faces: List[ClusterRequest]) -> List[ClusterResponse]:
        """聚类人脸"""
        if not faces:
            return []
            
        # 准备数据
        embeddings = np.array([face.ndarray for face in faces])
        file_codes = [face.fileCode for face in faces]
        
        # 计算距离矩阵
        distances = cosine_distances(embeddings)
        
        # 打印距离矩阵用于调试
        print("\n人脸距离矩阵:")
        for i in range(len(faces)):
            for j in range(len(faces)):
                print(f"{distances[i][j]:.3f} ", end="")
            print()
        
        # 存储聚类结果
        clusters = []  # [(person_id, [file_codes])]
        used_indices = set()
        
        # 遍历所有人脸
        for i in range(len(faces)):
            if i in used_indices:
                continue
                
            # 找到与当前人脸相似的其他人脸
            similar_indices = {i}  # 包含自己
            for j in range(len(faces)):
                if j != i and j not in used_indices and distances[i][j] < self.max_distance:
                    similar_indices.add(j)
            
            # 如果相似人脸数量达到阈值，创建新的簇
            if len(similar_indices) >= self.min_faces:
                cluster_codes = [file_codes[idx] for idx in similar_indices]
                clusters.append((f"person_{len(clusters)}", cluster_codes))
                used_indices.update(similar_indices)
        
        # 转换为响应格式
        return [
            ClusterResponse(faces=person_id, cluster=cluster_codes)
            for person_id, cluster_codes in clusters
        ]

# FastAPI应用
app = FastAPI(
    title="人脸识别服务",
    description="提供人脸特征提取和聚类功能",
    version="1.0.0"
)

# 创建服务实例
face_service = FaceService()

@app.post("/extract_faces", response_model=BatchImageResponse)
async def extract_faces(request: BatchImageRequest):
    """
    批量提取图片中的人脸特征
    
    Args:
        request: 包含图片路径和编码的请求
        
    Returns:
        包含处理结果的响应
    """
    try:
        results = []
        for image in request.images:
            embedding = face_service.get_face_embedding(image.filePath)
            
            response = ImageResponse(
                fileType=1 if embedding is not None else 0,
                fileCode=image.fileCode,
                ndarray=embedding
            )
            results.append(response)
            
        return BatchImageResponse(results=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster_faces", response_model=BatchClusterResponse)
async def cluster_faces(request: BatchClusterRequest):
    """
    对人脸特征向量进行聚类
    
    Args:
        request: 包含人脸特征向量和文件编码的请求
        
    Returns:
        聚类结果
    """
    try:
        results = face_service.cluster_faces(request.faces)
        return BatchClusterResponse(results=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "healthy"}