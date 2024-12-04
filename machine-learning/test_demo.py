import numpy as np
import requests
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_distances
import os

@dataclass
class Face:
    """人脸数据结构"""
    id: str  # 图片ID
    embedding: np.ndarray  # 512维特征向量
    score: float  # 人脸检测置信度
    person_id: Optional[str] = None  # 所属人物ID

class FaceCluster:
    def __init__(self, 
                 max_recognition_distance: float = 0.5,
                 min_recognized_faces: int = 3):
        """
        初始化人脸聚类器
        
        Args:
            max_recognition_distance: 最大识别距离(0.3-0.7)
            min_recognized_faces: 创建新人物所需的最小人脸数(默认3)
        """
        self.max_distance = max_recognition_distance
        self.min_faces = min_recognized_faces
        self.people: Dict[str, List[Face]] = {}  # 人物ID -> 人脸列表
        
    def get_face_embedding(self, image_path: str, api_url: str) -> List[Face]:
        """从API获取人脸特征向量"""
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
            # 确保图片文件存在且可读
            with open(image_path, 'rb') as img_file:
                # 使用multipart/form-data格式
                files = {
                    'image': ('image.jpg', img_file, 'image/jpeg'),  # 显式指定文件名和MIME类型
                    'entries': (None, json.dumps(config), 'application/json')
                }
                
                response = requests.post(api_url, files=files)
                response.raise_for_status()
                
                if response.status_code == 200:
                    result = response.json()
                    faces = []
                    for i, face_data in enumerate(result.get("facial-recognition", [])):
                        face = Face(
                            id=f"{image_path}_{i}",
                            embedding=np.array(face_data["embedding"]),
                            score=face_data["score"]
                        )
                        faces.append(face)
                    return faces
                else:
                    print(f"API请求失败: {response.status_code}")
                    return []
            
        except FileNotFoundError:
            print(f"找不到图片文件: {image_path}")
            return []
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return []
    
    def find_similar_faces(self, face: Face) -> List[Face]:
        """查找与给定人脸相似的人脸"""
        all_faces = []
        for faces in self.people.values():
            all_faces.extend(faces)
            
        if not all_faces:
            return []
            
        # 计算余弦距离
        embeddings = np.vstack([f.embedding for f in all_faces])
        distances = cosine_distances([face.embedding], embeddings)[0]
        
        # 获取距离小于阈值的人脸
        similar_indices = np.where(distances < self.max_distance)[0]
        similar_faces = [all_faces[i] for i in similar_indices]
        
        return similar_faces
    
    def is_core_point(self, face: Face, similar_faces: List[Face]) -> bool:
        """判断是否为核心点"""
        return len(similar_faces) + 1 >= self.min_faces
    
    def process_face(self, face: Face):
        """处理单个人脸"""
        # 1. 查找相似人脸
        similar_faces = self.find_similar_faces(face)
        
        # 2. 检查相似人脸是否属于某个人物
        person_counts = {}
        for similar_face in similar_faces:
            if similar_face.person_id:
                person_counts[similar_face.person_id] = \
                    person_counts.get(similar_face.person_id, 0) + 1
        
        # 3. 如果有相似人脸属于某个人物，分配到最多相似人脸的人物
        if person_counts:
            most_common_person = max(person_counts.items(), key=lambda x: x[1])[0]
            face.person_id = most_common_person
            self.people[most_common_person].append(face)
            return
            
        # 4. 如果是核心点，创建新人物
        if self.is_core_point(face, similar_faces):
            new_person_id = f"person_{len(self.people)}"
            face.person_id = new_person_id
            self.people[new_person_id] = [face]
            
            # 同时将未分配的相似人脸也加入该人物
            for similar_face in similar_faces:
                if not similar_face.person_id:
                    similar_face.person_id = new_person_id
                    self.people[new_person_id].append(similar_face)
    
    def process_faces(self, faces: List[Face]):
        """批量处理人脸"""
        if not faces:
            return
        
        # 打印所有人脸之间的距离矩阵
        print("\n人脸距离矩阵:")
        n = len(faces)
        embeddings = np.vstack([f.embedding for f in faces])
        distances = cosine_distances(embeddings)
        
        # 打印表头
        print("     ", end="")
        for i in range(n):
            print(f"Face{i:2d}  ", end="")
        print("\n" + "-" * (8 * n + 5))
        
        # 打印距离矩阵
        for i in range(n):
            print(f"Face{i:2d}", end=" ")
            for j in range(n):
                print(f"{distances[i][j]:6.3f}", end=" ")
            print()
        print()
        
        # 打印可能属于同一个人的人脸对
        print("\n可能属于同一个人的人脸对 (距离 < 0.6):")
        for i in range(n):
            for j in range(i+1, n):
                if distances[i][j] < self.max_distance:
                    print(f"Face {i} 和 Face {j}: 距离 = {distances[i][j]:.3f}")
                    print(f"  - {faces[i].id}")
                    print(f"  - {faces[j].id}")
        
        # 修改聚类逻辑
        # 1. 找出所有未分配的人脸
        unassigned = faces.copy()
        person_id = 0
        
        # 2. 当还有未分配的人脸时，继续处理
        while unassigned:
            # 取第一个未分配的人脸作为新簇的种子
            seed_face = unassigned[0]
            
            # 找出与种子人脸相似的所有人脸
            similar_faces = []
            for face in unassigned[1:]:
                distance = cosine_distances([seed_face.embedding], [face.embedding])[0][0]
                if distance < self.max_distance:
                    similar_faces.append(face)
            
            # 如果找到足够多的相似人脸，创建新的人物簇
            if len(similar_faces) + 1 >= self.min_faces:
                new_person_id = f"person_{person_id}"
                person_id += 1
                
                # 将种子人脸和相似人脸加入新簇
                self.people[new_person_id] = [seed_face]
                seed_face.person_id = new_person_id
                unassigned.remove(seed_face)
                
                for face in similar_faces:
                    face.person_id = new_person_id
                    self.people[new_person_id].append(face)
                    unassigned.remove(face)
                    
                print(f"\n创建新人物簇 {new_person_id} 包含 {len(self.people[new_person_id])} 个人脸")
            else:
                # 如果没有足够的相似人脸，将这个人脸标记为未分配
                unassigned.remove(seed_face)
                print(f"\n人脸 {seed_face.id} 没有足够的相似人脸，暂不分配")

def main():
    """主函数"""
    # API配置
    api_url = "http://home.plantplanethome.com:3003/predict"
    
    # 图片路径列表
    image_paths = [
        "/Users/yuzheng/Downloads/pic/黄1.jpg",
        "/Users/yuzheng/Downloads/pic/黄2.jpg",
        "/Users/yuzheng/Downloads/pic/涛1.jpg",
        "/Users/yuzheng/Downloads/pic/涛2.jpg",
        "/Users/yuzheng/Downloads/pic/张1.jpg",
        "/Users/yuzheng/Downloads/pic/张2.jpg"
    ]
    
    # 初始化聚类器，调整参数
    cluster = FaceCluster(
        max_recognition_distance=0.6,  # 距离阈值
        min_recognized_faces=2  # 最小人脸数
    )
    
    # 获取所有人脸特征
    all_faces = []
    for image_path in image_paths:
        faces = cluster.get_face_embedding(image_path, api_url)
        print(f"图片 {image_path} 检测到 {len(faces)} 个人脸")
        all_faces.extend(faces)
    
    # 进行聚类
    cluster.process_faces(all_faces)
    
    # 打印结果
    print("\n聚类结果:")
    for person_id, faces in cluster.people.items():
        print(f"\n人物 {person_id}:")
        for face in faces:
            print(f"  - 图片: {face.id}")
    
    # 打印未分配的人脸
    unassigned = [f for f in all_faces if not f.person_id]
    if unassigned:
        print("\n未分配的人脸:")
        for face in unassigned:
            print(f"  - 图片: {face.id}")

if __name__ == "__main__":
    main()
