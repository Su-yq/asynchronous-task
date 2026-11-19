"""
Copyright (C) 2025 bio_GUO_LAB

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
import uuid
from typing import Dict, Optional, Any, List
import time
from enum import Enum
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors, Lipinski
from admet_ai import ADMETModel
from email_sender import send_email_file
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional
from math import ceil
from model_output import send_result_email
import models
# from database import SessionLocal, engine

import os
from typing import Optional
import logging
import threading

from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
import Protein_Process

import SMILES_Process
import kcat_Process
import numpy as np
import json
import torch

import csv
import io
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import base64
from io import BytesIO
from database_do3 import router as data_router  # 导入路由器
from kg_router import router as kg_router
from workflow_router import router as workflow_router
import crud
import schemas
from database1 import SessionLocal, engine
from utils import Item, FileItem, EmailRequest, Item2
import base64
from io import BytesIO
import warnings
from starlette.responses import JSONResponse, FileResponse
from tempfile import NamedTemporaryFile
from celery_worker import process_task, celery_app
import pandas as pd

warnings.filterwarnings("ignore")

# 创建 FastAPI 应用实例
app = FastAPI(title="分子属性 API", description="基于 SMILES 计算分子属性的 API")

# 添加 CORS 中间件，解决跨域问题
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应限制特定来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头部
)
# 在创建 app 后，包含数据浏览路由
app.include_router(data_router)
app.include_router(kg_router)
app.include_router(workflow_router)

# 初始化 ADMET 模型
admet_model = ADMETModel()
app.state.admet_model = admet_model
app.state.admet_predict_lock = threading.Lock()
# ==================== 本地模型配置 ====================
BASE_DIR = "/home/liuxiaoyi/bio_chem_crow_fastapi-new"
MODELS_DIR = os.path.join("./models")
address = "D/graduate/Backend/bio_chem_crow_fastapi"

# 设备配置
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# 全局缓存 - 存储加载的模型和数据
loaded_cpi_models: Dict[str, Any] = {}
protein_sequences_cache: Dict[str, str] = {}
tape_model_cache = None
tape_tokenizer_cache = None

kcat_lock = threading.Lock()


# ==================== 数据模型定义 ====================

class TaskStatus(str, Enum):
    """
    任务状态枚举类

    PENDING: 等待处理
    PROCESSING: 正在处理
    COMPLETED: 已完成
    FAILED: 处理失败
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MoleculeRequest(BaseModel):
    """
    分子请求模型

    smi: SMILES 字符串，表示分子结构
    task_name: 可选的任务名称
    """
    smi: str = Field(..., description="分子的 SMILES 表示")
    task_name: Optional[str] = Field(None, description="任务名称（可选）")


class TaskResponse(BaseModel):
    """
    任务响应模型

    taskid: 任务唯一标识符
    message: 响应消息
    """
    taskid: str
    message: str = "任务已成功提交"


class Protein1Input(BaseModel):
    """
    分子输入模型，用于存储输入数据

    smi: SMILES 字符串
    """
    smi: str


class Protein2Input(BaseModel):
    """
    蛋白质验证输入模型，用于存储输入数据

    smi: SMILES 字符串
    protein_sequence: 蛋白质序列
    download_file_path: 需要下载的远程文件路径
    """
    smi: str
    protein_sequence: str
    download_file_path: Optional[str] = None


class ProteinInput(BaseModel):
    """
    蛋白质输入模型，用于存储输入数据

    smi: SMILES 字符串
    protein_sequence: 蛋白质序列
    """
    smi: str
    protein_sequence: str


class MoleculeInput(BaseModel):
    """
    分子输入模型，用于存储输入数据

    smi: SMILES 字符串
    """
    smi: str


class TaskResult(BaseModel):
    """
    任务结果模型，包含任务状态和结果

    taskid: 任务ID
    status: 任务状态
    input: 输入的分子数据
    results: 计算的分子属性结果
    error: 如果有错误，存储错误信息
    progress: 执行进度（0-100）
    progress_message: 进度描述信息
    """
    taskid: str
    status: TaskStatus
    input: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[int] = 0
    progress_message: Optional[str] = "Waiting to start"


# 数据库相关
class TCMDataItem(BaseModel):
    smiles: str
    cid: str
    english: Optional[str] = ""

    class Config:
        orm_mode = True


class TCMListResponse(BaseModel):
    items: List[TCMDataItem]
    total: int
    pages: int
    page: int
    limit: int


# 依赖项，用于获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_unique_task_id():
    task_id = uuid.uuid4()
    return str(task_id)


def remove_file(file_path: str):
    os.remove(file_path)
# ==================== TAPE模型加载和蛋白质嵌入函数 ====================

def load_tape_model():
    """
    加载TAPE预训练模型
    TAPE模型输出768维的蛋白质嵌入，与训练时使用的方法一致
    """
    global tape_model_cache, tape_tokenizer_cache

    if tape_model_cache is None:
        try:
            from tape import ProteinBertModel, TAPETokenizer

            print("Loading TAPE model...")
            tape_model_cache = ProteinBertModel.from_pretrained('bert-base')
            tape_tokenizer_cache = TAPETokenizer(vocab='iupac')
            tape_model_cache.eval()

            # 如果有GPU，将模型移到GPU
            if USE_CUDA:
                tape_model_cache = tape_model_cache.to(DEVICE)

            print("TAPE model loaded successfully")
        except ImportError as e:
            raise ImportError(
                "TAPE package not installed. Please install it with: "
                "pip install tape-proteins"
            ) from e

    return tape_model_cache, tape_tokenizer_cache


def get_protein_embedding_tape(sequence: str, max_length: int = 1500) -> tuple:
    """
    使用TAPE模型获取蛋白质embedding (768维)

    这是正确的方法，与训练时data_process.py中的tape_embedding函数一致:
    - 使用ProteinBertModel编码蛋白质序列
    - 输出维度为768
    - 需要去除首尾的特殊token ([CLS] 和 [SEP])
    - padding到max_length

    参数:
        sequence: 蛋白质序列字符串
        max_length: 最大序列长度，默认1500

    返回:
        feature: 蛋白质嵌入张量 (max_length, 768)
        size: 实际序列长度
    """
    model, tokenizer = load_tape_model()

    # 编码序列
    sequence = sequence.upper()
    token_ids = torch.tensor([tokenizer.encode(sequence)])

    # 如果有GPU，移动到GPU
    if USE_CUDA:
        token_ids = token_ids.to(DEVICE)

    # 获取embedding
    with torch.no_grad():
        sequence_output, _ = model(token_ids)

    # 移回CPU进行处理
    feature = sequence_output.squeeze().cpu().numpy()

    # 去除首尾的特殊token ([CLS] 和 [SEP]) - 与训练时一致
    feature = np.delete(feature, -1, axis=0)  # 删除最后一个 [SEP]
    feature = np.delete(feature, 0, axis=0)  # 删除第一个 [CLS]

    size = feature.shape[0]

    # Padding到max_length - 与训练时一致
    pad_length = max_length - size
    if pad_length > 0:
        padding = np.zeros((pad_length, 768), dtype='float32')
        feature = np.vstack([feature, padding])
    elif pad_length < 0:
        # 如果序列过长，截断
        feature = feature[:max_length]
        size = max_length

    return torch.from_numpy(feature.astype('float32')), size


# ==================== CPI模型相关函数 ====================
from torch_geometric.nn import SAGEConv, global_max_pool as gmp
import torch.nn as nn


class CPI_regression(torch.nn.Module):
    """
    CPI回归模型 - 与训练时model.py中的定义完全一致
    """

    def __init__(self, device, emb_size, max_length, n_output=1, hidden_size=128, num_features_mol=78, dropout=None):
        super(CPI_regression, self).__init__()
        print('CPI_regression model Loaded..')
        self.device = device
        self.skip = 1
        self.n_output = n_output
        self.max_length = max_length
        # compounds network
        self.mol_conv1 = SAGEConv(num_features_mol, num_features_mol * 2, 'mean')
        self.mol_conv2_f = SAGEConv(num_features_mol * 2, num_features_mol * 2, 'mean')
        self.mol_conv3_f = SAGEConv(num_features_mol * 2, num_features_mol * 4, 'mean')
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # proteins network
        self.prot_rnn = nn.LSTM(emb_size, hidden_size, 1)
        self.relu = nn.LeakyReLU()
        # combined layers
        self.prot_comp_mix = nn.Sequential(nn.Linear(129 * 129, 1024), nn.LeakyReLU(), nn.Dropout(dropout))
        self.fc = nn.Sequential(nn.Linear(1282, 512), nn.LeakyReLU(), nn.Dropout(dropout),
                                nn.Linear(512, self.n_output))

    def forward(self, data_mol, data_pro, data_pro_len):
        # protein network
        pro_seq_lengths, pro_idx_sort = torch.sort(data_pro_len, descending=True)[::-1][1], torch.argsort(-data_pro_len)
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        data_pro = data_pro.index_select(0, pro_idx_sort)
        xt = nn.utils.rnn.pack_padded_sequence(data_pro, pro_seq_lengths.cpu(), batch_first=True)
        xt, _ = self.prot_rnn(xt)
        xt = nn.utils.rnn.pad_packed_sequence(xt, batch_first=True, total_length=max(1500, self.max_length))[0]
        xt = xt.index_select(0, pro_idx_unsort)
        xt = xt.mean(1)

        # compound network
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv2_f(x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv3_f(x, mol_edge_index)
        x = self.relu(x)
        x = gmp(x, mol_batch)  # global max pooling
        x = self.mol_fc_g1(x)
        x = self.dropout(x)

        # kronecker product
        prot_out = torch.cat((xt, torch.FloatTensor(xt.shape[0], 1).fill_(1).to(self.device)), 1)
        comp_out = torch.cat((x, torch.FloatTensor(x.shape[0], 1).fill_(1).to(self.device)), 1)
        output = torch.bmm(prot_out.unsqueeze(2), comp_out.unsqueeze(1)).flatten(start_dim=1)
        output = self.dropout(output)
        output = self.prot_comp_mix(output)
        if self.skip:
            output = torch.cat((output, prot_out, comp_out), 1)
        output = self.fc(output)
        return output


def load_cpi_model(model_name: str = "tape_reg") -> CPI_regression:
    """
    加载CPI预测模型

    参数:
        model_name: 模型名称，可选 "tape_reg", "tape_cls", "esm2_reg"
    """
    global loaded_cpi_models

    if model_name in loaded_cpi_models:
        return loaded_cpi_models[model_name]

    # 模型配置
    model_configs = {
        "tape_reg": {
            "path": os.path.join(MODELS_DIR, "tape_BindingDB_reg.pt"),
            "emb_size": 768,
            "max_length": 1500,
        },
        "tape_cls": {
            "path": os.path.join(MODELS_DIR, "tape_BindingDB_cls.pt"),
            "emb_size": 768,
            "max_length": 1500,
        },
        "esm2_reg": {
            "path": os.path.join(MODELS_DIR, "esm2_BindingDB_reg.pt"),
            "emb_size": 1280,
            "max_length": 1500,
        },
    }

    config = model_configs.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = config["path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 创建模型实例
    model = CPI_regression(
        device=DEVICE,
        emb_size=config["emb_size"],
        max_length=config["max_length"],
        dropout=0.2,
    )

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    loaded_cpi_models[model_name] = model
    print(f"Loaded CPI model: {model_name} from {model_path}")

    return model


# ==================== 分子图构建函数 ====================

import networkx as nx


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """分子原子特征提取 - 与训练时一致"""
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def smile_to_graph(smile):
    """
    将SMILES转换为分子图
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")

    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    return c_size, features, edge_index


# ==================== 蛋白质序列加载 ====================

def load_protein_sequences():
    """加载蛋白质序列数据"""
    global protein_sequences_cache

    protein_file = os.path.join(BASE_DIR, "protein_sequences.json")
    if os.path.exists(protein_file):
        with open(protein_file, "r", encoding="utf-8") as f:
            protein_sequences_cache = json.load(f)
        print(f"Loaded {len(protein_sequences_cache)} protein sequences from {protein_file}")
    else:
        print(f"Warning: {protein_file} not found")
        protein_sequences_cache = {}

    return protein_sequences_cache


def load_id_mapping():
    """加载蛋白质ID映射数据"""
    id_mapping = {}
    id_mapping_file = os.path.join(MODELS_DIR, "idmapping.csv")

    if os.path.exists(id_mapping_file):
        with open(id_mapping_file, "r", encoding="utf-8") as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                entry = row.get("Entry", "")
                if entry:
                    id_mapping[entry] = {
                        "entry": entry,
                        "entry_name": row.get("Entry Name", ""),
                        "protein_names": row.get("Protein names", ""),
                        "gene_names": row.get("Gene Names", ""),
                        "organism": row.get("Organism", ""),
                        "length": row.get("Length", ""),
                        "gene_names_synonym": row.get("Gene Names (synonym)", ""),
                        "organism_id": row.get("Organism (ID)", "")
                    }
        print(f"Loaded {len(id_mapping)} protein ID mappings")
    else:
        print(f"Warning: {id_mapping_file} not found")

    return id_mapping


# ==================== 本地预测函数 ====================

def predict_binding_affinity_local(smiles: str, protein_sequence: str, model_name: str = "tape_reg") -> float:
    """
    使用本地CPI模型预测分子-蛋白质结合亲和力

    参数:
        smiles: 分子的SMILES字符串
        protein_sequence: 蛋白质序列
        model_name: 使用的模型名称

    返回:
        binding_affinity: 预测的结合亲和力分数
    """
    from torch_geometric import data as DATA
    from torch_geometric.data import Batch

    # 加载CPI模型
    model = load_cpi_model(model_name)

    # 准备分子数据
    c_size, features, edge_index = smile_to_graph(smiles)

    # 使用TAPE模型获取蛋白质嵌入（768维）- 这是关键！
    prot_feature, prot_size = get_protein_embedding_tape(protein_sequence, max_length=1500)

    # 构建分子图输入
    mol_data = DATA.Data(
        x=torch.Tensor(features),
        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
        y=torch.FloatTensor([0]),
    )
    mol_data.__setitem__("c_size", torch.LongTensor([c_size]))

    # 批处理
    mol_batch = Batch.from_data_list([mol_data]).to(DEVICE)
    prot_batch = prot_feature.unsqueeze(0).to(DEVICE)
    prot_len_batch = torch.tensor([prot_size]).to(DEVICE)

    # 预测
    with torch.no_grad():
        output = model(mol_batch, prot_batch, prot_len_batch)
        prediction = output.cpu().numpy().flatten()[0]

    return float(prediction)


def find_target_proteins_local(smiles: str, top_k: int = 10, threshold: float = 5.5) -> List[Dict[str, Any]]:
    """
    使用本地模型查找可能的靶点蛋白质

    参数:
        smiles: 分子的SMILES字符串
        top_k: 返回的靶点数量
        threshold: 结合亲和力阈值

    返回:
        靶点蛋白质列表
    """
    from torch_geometric import data as DATA
    from torch_geometric.data import Batch

    # 确保蛋白质序列已加载
    if not protein_sequences_cache:
        load_protein_sequences()

    if not protein_sequences_cache:
        return []

    results = []
    model = load_cpi_model("tape_reg")

    # 准备分子数据
    try:
        c_size, features, edge_index = smile_to_graph(smiles)
    except Exception as e:
        raise ValueError(f"Invalid SMILES: {e}")

    mol_data = DATA.Data(
        x=torch.Tensor(features),
        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
        y=torch.FloatTensor([0]),
    )
    mol_data.__setitem__("c_size", torch.LongTensor([c_size]))
    mol_batch = Batch.from_data_list([mol_data]).to(DEVICE)

    # 加载ID映射
    id_mapping = load_id_mapping()

    # 遍历所有蛋白质进行预测
    protein_list = list(protein_sequences_cache.items())[:100]  # 限制数量以提高速度

    for protein_id, sequence in protein_list:
        try:
            prot_feature, prot_size = get_protein_embedding_tape(sequence, max_length=1500)
            prot_batch = prot_feature.unsqueeze(0).to(DEVICE)
            prot_len_batch = torch.tensor([prot_size]).to(DEVICE)

            with torch.no_grad():
                output = model(mol_batch, prot_batch, prot_len_batch)
                score = output.cpu().numpy().flatten()[0]

            if score >= threshold:
                # 获取idmapping中的完整信息
                protein_info = id_mapping.get(protein_id, {})
                
                results.append({
                    "protein_id": protein_id,
                    # 从idmapping获取完整描述信息
                    "entry_name": protein_info.get("entry_name", ""),
                    "protein_name": protein_info.get("protein_names", "Unknown"),
                    "gene_name": protein_info.get("gene_names", "Unknown"),
                    "gene_name_synonym": protein_info.get("gene_names_synonym", ""),
                    "organism": protein_info.get("organism", "Unknown"),
                    "organism_id": protein_info.get("organism_id", ""),
                    "protein_length": protein_info.get("length", ""),
                    "binding_score": float(score),
                    "sequence_length": len(sequence),
                    "sequence": sequence,
                })
        except Exception as e:
            print(f"Error processing protein {protein_id}: {e}")
            continue

    # 按结合分数排序
    results.sort(key=lambda x: x["binding_score"], reverse=True)

    return results[:top_k]


def generate_molecule_image_base64(mol, size=(400, 300)):
    """
    生成分子结构图并返回base64编码的字符串

    参数:
        mol: RDKit分子对象
        size: 图片尺寸，默认(400, 300)

    返回:
        base64编码的图片字符串，格式为 "data:image/png;base64,..."
    """
    try:
        # 生成分子图片
        img = Draw.MolToImage(mol, size=size)

        # 将图片转换为字节流
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        # 转换为base64编码
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 返回完整的data URL格式
        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        print(f"Error occurred while generating molecular image: {str(e)}")
        # 返回空值，前端可以显示默认图片
        return None


def calculate_kcat_safe(smi: str, protein_sequence: str, task_id: str = None) -> float:
    """
    线程安全的kcat计算函数，直接调用 kcat_Process 函数
    """
    with kcat_lock:
        try:
            # 直接调用 kcat_Process 函数
            result = kcat_Process.kcat_Process(smi, protein_sequence)

            if result.get("success"):
                # 读取输出文件
                output_file = os.path.join(BASE_DIR, "DLKcat/DeeplearningApproach/Code/example/output.tsv")
                kcat_value = kcat_Process.kcat_get(output_file)
                return float(kcat_value) if kcat_value is not None else 0.0
            else:
                print(f"kcat_Process 失败: {result.get('error')}")
                return 0.0

        except Exception as e:
            print(f"Error in calculate_kcat_safe: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0


# ==================== 内存存储任务数据 ====================
tasks: Dict[str, TaskResult] = {}


@app.get("/task/{taskid}", response_model=TaskResult)
async def get_task_result(taskid: str):
    """
    获取任务结果

    根据任务ID查询任务状态和结果

    参数:
        taskid: 任务ID

    返回:
        任务详细信息，包括状态、输入和结果

    异常:
        404: 如果找不到指定任务
    """
    print(taskid)
    if taskid not in tasks:
        raise HTTPException(status_code=404, detail="找不到该任务")
    return tasks[taskid]


# 获取任务进度的API端点
@app.get("/task/{taskid}/progress")
async def get_task_progress(taskid: str):
    """
    获取任务执行进度

    参数:
        taskid: 任务ID

    返回:
        任务进度信息
    """
    if taskid not in tasks:
        raise HTTPException(status_code=404, detail="找不到该任务")

    task = tasks[taskid]
    return {
        "taskid": taskid,
        "status": task.status,
        "progress": task.progress,
        "progress_message": task.progress_message,
        "error": task.error
    }


@app.get("/health")
async def health_check():
    """
    健康检查API

    用于检查服务是否正常运行

    返回:
        包含状态和版本信息的对象
    """
    return {"status": "ok", "version": "1.0.0", "admet_ai": "enabled", "device": str(DEVICE)}


@app.post("/attribute-predict", response_model=TaskResponse)
async def api_attribute_predict(
        request: MoleculeRequest,
        background_tasks: BackgroundTasks
):
    """
    分子属性预测API

    接收SMILES字符串，创建后台任务进行属性计算，并返回任务ID

    参数:
        request: 包含SMILES的请求对象
        background_tasks: FastAPI后台任务管理器

    返回:
        包含任务ID的响应对象
    """
    # 创建唯一任务ID
    taskid = str(uuid.uuid4())
    smi = request.smi

    # 创建任务并设置初始状态
    tasks[taskid] = TaskResult(
        taskid=taskid,
        status=TaskStatus.PENDING,
        input={"smi": smi, "type": "molecule"},
        progress=0,
        progress_message="Waiting to start"
    )

    # 定义处理函数（直接作为后台任务）
    def run_molecule_calculations(taskid: str, smi: str):
        try:
            # 更新任务状态为处理中
            tasks[taskid].status = TaskStatus.PROCESSING
            tasks[taskid].progress = 10
            tasks[taskid].progress_message = "Starting molecular calculation"

            # 从SMILES创建RDKit分子对象
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smi}")

            tasks[taskid].progress = 30
            tasks[taskid].progress_message = "Generating molecular structure diagram"

            # 生成分子结构图并转换为base64
            structure_image_base64 = generate_molecule_image_base64(mol)

            tasks[taskid].progress = 50
            tasks[taskid].progress_message = "Using ADMET-AI for prediction"
            with app.state.admet_predict_lock:
                admet_predictions = admet_model.predict(smiles=smi)
            # 使用ADMET-AI进行预测

            tasks[taskid].progress = 100
            tasks[taskid].progress_message = "Calculation completed"

            # 更新任务结果和状态 - 包含图片数据
            tasks[taskid].results = {
                **admet_predictions,  # 包含所有ADMET预测结果
                "structure_image": structure_image_base64,  # 添加结构图片
                "smi": smi  # 包含原始SMILES
            }
            tasks[taskid].status = TaskStatus.COMPLETED

        except Exception as e:
            # 发生错误时，更新任务状态和错误信息
            tasks[taskid].status = TaskStatus.FAILED
            tasks[taskid].error = str(e)
            tasks[taskid].progress_message = f"Calculation failed: {str(e)}"

    # 添加处理任务到后台任务队列
    background_tasks.add_task(run_molecule_calculations, taskid, smi)

    return TaskResponse(taskid=taskid)


@app.post("/protein-predict", response_model=TaskResponse)
async def protein_predict(request: Protein1Input, background_tasks: BackgroundTasks):
    """
    蛋白靶点预测API - 使用本地TAPE模型进行预测
    """
    # 创建唯一任务ID
    taskid = str(uuid.uuid4())

    smi = request.smi
    # 创建任务并设置初始状态
    tasks[taskid] = TaskResult(
        taskid=taskid,
        status=TaskStatus.PENDING,
        input={
            "smi": smi,
            "type": "protein_predict"
        },
        progress=0,
        progress_message="Task created, waiting to start processing"
    )

    def process_protein_predict(taskid: str, smi: str):
        try:
            # 更新任务状态为处理中
            tasks[taskid].status = TaskStatus.PROCESSING
            tasks[taskid].progress = 5
            tasks[taskid].progress_message = "Start processing"

            # 验证SMILES
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smi}")

            tasks[taskid].progress = 10
            tasks[taskid].progress_message = "Processing SMILES data"

            # 处理SMILES数据
            cid = SMILES_Process.get_cid_from_smiles(smiles=smi)

            tasks[taskid].progress = 15
            tasks[taskid].progress_message = "Generating molecular structure image"
            structure_image_base64 = generate_molecule_image_base64(mol)

            tasks[taskid].progress = 20
            tasks[taskid].progress_message = "Loading TAPE model"

            # 预加载TAPE模型
            load_tape_model()

            tasks[taskid].progress = 30
            tasks[taskid].progress_message = "Loading protein sequences"

            # 加载蛋白质序列
            load_protein_sequences()

            tasks[taskid].progress = 40
            tasks[taskid].progress_message = "Predicting target proteins using local model"

            # 使用本地模型查找靶点蛋白质
            target_proteins = find_target_proteins_local(smi, top_k=20, threshold=5.0)

            tasks[taskid].progress = 60
            tasks[taskid].progress_message = "Processing results and calculating kcat values"

            proteins_with_kcat = []
            total_proteins = len(target_proteins)

            for idx, protein in enumerate(target_proteins):
                protein_sequence = protein.get("sequence", "")
                protein_score = protein.get("binding_score", 0.0)

                # 计算每个蛋白质的kcat
                protein_kcat = 0.0
                if protein_sequence and len(protein_sequence) >= 10:
                    try:
                        # 使用新的线程安全函数计算kcat
                        protein_kcat = calculate_kcat_safe(
                            smi,
                            protein_sequence,
                            task_id=f"{taskid}_{idx}"
                        )
                        print(f"蛋白质 {idx + 1}/{total_proteins} score: {protein_score:.6f}, kcat: {protein_kcat:.8f}")
                    except Exception as kcat_error:
                        print(f"计算蛋白质 {idx + 1} kcat值时出错: {str(kcat_error)}")
                        protein_kcat = 0.0

                # 添加到结果列表，包含独立的score和kcat
                proteins_with_kcat.append({
                    "protein_id": protein.get("protein_id", ""),
                    "entry_name": protein.get("entry_name", ""),
                    "protein_name": protein.get("protein_name", "Unknown"),
                    "gene_name": protein.get("gene_name", "Unknown"),
                    "gene_name_synonym": protein.get("gene_name_synonym", ""),
                    "organism": protein.get("organism", "Unknown"),
                    "organism_id": protein.get("organism_id", ""),
                    "protein_length": protein.get("protein_length", ""),
                    "sequence": protein_sequence,
                    "sequence_length": len(protein_sequence),
                    "score": round(protein_score, 6),
                    "kcat": round(protein_kcat, 8)
                })

                # 更新进度
                progress = 60 + int(35 * (idx + 1) / total_proteins)
                tasks[taskid].progress = progress
                tasks[taskid].progress_message = f"Calculating kcat for protein {idx + 1}/{total_proteins}"

            # 获取最高分数作为摘要显示
            top_score = proteins_with_kcat[0]["score"] if proteins_with_kcat else 0.0
            top_kcat = proteins_with_kcat[0]["kcat"] if proteins_with_kcat else 0.0

            # 更新任务结果和状态
            tasks[taskid].progress = 100
            tasks[taskid].progress_message = "Task completed"
            tasks[taskid].results = {
                "score": top_score,
                "kcat": top_kcat,
                "smi": smi,
                "structure_image": structure_image_base64,  # 添加分子结构图
                "protein_sequence": [p["sequence"] for p in proteins_with_kcat],
                "proteins": proteins_with_kcat,  # 包含每个蛋白质独立的score和kcat
                "target_proteins": proteins_with_kcat,
                "computation_info": {
                    "score_description": "蛋白质-分子结合亲和力预测分数",
                    "kcat_description": "酶催化效率常数 (kcat)",
                    "score_unit": "binding_affinity_score",
                    "kcat_unit": "s^-1",
                    "model_used": "CPI_regression with TAPE embedding (768-dim)",
                    "device": str(DEVICE)
                }
            }
            tasks[taskid].status = TaskStatus.COMPLETED
            print(f"任务 {taskid} 成功完成，处理了 {len(proteins_with_kcat)} 个蛋白质")

        except Exception as e:
            import traceback
            # 发生错误时，更新任务状态和错误信息
            tasks[taskid].status = TaskStatus.FAILED
            tasks[taskid].error = str(e)
            tasks[taskid].progress_message = f"Task failed: {str(e)}"
            print(f"Task {taskid} failed: {str(e)}")
            traceback.print_exc()

    # 添加处理任务到后台任务队列
    background_tasks.add_task(process_protein_predict, taskid, smi)

    return TaskResponse(taskid=taskid)


@app.post("/protein-validate", response_model=TaskResponse)
async def protein_validate(request: Protein2Input, background_tasks: BackgroundTasks):
    """
    蛋白质验证API - 使用本地TAPE模型进行验证
    """
    # 创建唯一任务ID
    taskid = str(uuid.uuid4())
    smi = request.smi
    protein_sequence = request.protein_sequence

    # 创建任务并设置初始状态
    tasks[taskid] = TaskResult(
        taskid=taskid,
        status=TaskStatus.PENDING,
        input={
            "smi": smi,
            "protein_sequence": protein_sequence,
            "type": "protein_validate"
        },
        progress=0,
        progress_message="Task created, waiting to start processing"
    )

    def process_protein_validation(taskid: str, smi: str, protein_sequence: str):
        try:
            # 更新任务状态为处理中
            tasks[taskid].status = TaskStatus.PROCESSING
            tasks[taskid].progress = 5
            tasks[taskid].progress_message = "Start processing"

            # 验证SMILES
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smi}")

            # 验证蛋白质序列
            if not protein_sequence or len(protein_sequence) < 10:
                raise ValueError("Protein sequence too short (minimum 10 amino acids)")

            tasks[taskid].progress = 10
            tasks[taskid].progress_message = "Generating molecular structure image"
            structure_image_base64 = generate_molecule_image_base64(mol)

            tasks[taskid].progress = 15
            tasks[taskid].progress_message = "Processing SMILES data"

            # 处理SMILES数据
            cid = SMILES_Process.get_cid_from_smiles(smiles=smi)

            tasks[taskid].progress = 25
            tasks[taskid].progress_message = "Processing protein data"

            # 处理蛋白质数据
            protein_id = Protein_Process.find_protein_id_by_sequence(sequence=protein_sequence)
            print(f"Protein ID: {protein_id}")

            tasks[taskid].progress = 35
            tasks[taskid].progress_message = "Loading TAPE model"

            # 预加载TAPE模型
            load_tape_model()

            tasks[taskid].progress = 50
            tasks[taskid].progress_message = "Predicting binding affinity using local model"

            # 使用本地模型预测结合亲和力
            score = predict_binding_affinity_local(smi, protein_sequence, model_name="tape_reg")
            print(f"获取到的score: {score}")

            tasks[taskid].progress = 75
            tasks[taskid].progress_message = "Calculating kcat value"

            # 计算kcat
            kcat_Process.kcat_Process(smi, protein_sequence)
            kcat_address = os.path.join(BASE_DIR, "DLKcat/DeeplearningApproach/Code/example/output.tsv")
            kcat = kcat_Process.kcat_get(kcat_address)
            kcat = float(kcat) if kcat is not None else 0.0
            print(f"获取到的kcat: {kcat}")

            # 更新任务结果和状态
            tasks[taskid].progress = 100
            tasks[taskid].progress_message = "Task completed"
            tasks[taskid].results = {
                "score": score,
                "kcat": kcat,
                "smi": smi,
                "structure_image": structure_image_base64,  # 添加分子结构图
                "protein_sequence": protein_sequence,
                "computation_info": {
                    "score_description": "蛋白质-分子结合亲和力预测分数",
                    "kcat_description": "酶催化效率常数 (kcat)",
                    "score_unit": "binding_affinity_score",
                    "kcat_unit": "s^-1",
                    "model_used": "CPI_regression with TAPE embedding (768-dim)",
                    "device": str(DEVICE)
                }
            }
            tasks[taskid].status = TaskStatus.COMPLETED
            print(f"任务 {taskid} 成功完成，score: {score}, kcat: {kcat}")

        except Exception as e:
            import traceback
            # 发生错误时，更新任务状态和错误信息
            tasks[taskid].status = TaskStatus.FAILED
            tasks[taskid].error = str(e)
            tasks[taskid].progress_message = f"Task failed: {str(e)}"
            print(f"Task {taskid} failed: {str(e)}")
            traceback.print_exc()

    # 添加处理任务到后台任务队列
    background_tasks.add_task(process_protein_validation, taskid, smi, protein_sequence)

    return TaskResponse(taskid=taskid)


# API端点：获取TCM数据列表
@app.get("/tcm-data", response_model=TCMListResponse)
def get_tcm_data(
        db: Session = Depends(get_db),
        page: int = Query(1, ge=1),
        limit: int = Query(15, ge=1, le=100),
        search: Optional[str] = None
):
    """
    获取TCM数据列表，支持分页和模糊搜索
    """
    # 计算偏移量
    offset = (page - 1) * limit

    # 构建查询
    query = db.query(models.TCMData)

    # 添加搜索条件 - 模糊搜索
    if search and search.strip():
        search_pattern = f"%{search.strip()}%"
        query = query.filter(
            or_(
                models.TCMData.cid.like(search_pattern),
                models.TCMData.english.like(search_pattern),
                models.TCMData.smiles.like(search_pattern)
            )
        )

    # 获取总数
    total = query.count()
    pages = ceil(total / limit) if total > 0 else 0

    # 获取当前页数据
    results = query.offset(offset).limit(limit).all()

    # 处理结果
    processed_results = []
    for item in results:
        processed_results.append({
            "smiles": item.smiles or "",
            "cid": item.cid or "",
            "english": item.english or ""
        })

    return {
        "items": processed_results,
        "total": total,
        "pages": pages,
        "page": page,
        "limit": limit
    }


# API端点：获取单个TCM详情
@app.get("/tcm-data/{tcm_smiles}", response_model=TCMDataItem)
def get_tcm_by_smiles(tcm_smiles: str, db: Session = Depends(get_db)):
    result = db.query(models.TCMData).filter(models.TCMData.smiles == tcm_smiles).first()
    if not result:
        raise HTTPException(status_code=404, detail="TCM data not found")
    return result


# 应用启动时预加载模型和数据
@app.on_event("startup")
def startup_event():
    """应用启动时预加载TAPE模型和蛋白质数据"""
    print("=" * 50)
    print("Starting Bio-Chem Prediction API...")
    print(f"Device: {DEVICE}")
    print(f"Models directory: {MODELS_DIR}")
    print("=" * 50)

    # 预加载蛋白质序列
    try:
        load_protein_sequences()
    except Exception as e:
        print(f"Warning: Failed to load protein sequences: {e}")

    # 预加载TAPE模型（可选，首次预测时也会自动加载）
    try:
        print("Pre-loading TAPE model...")
        load_tape_model()
        print("TAPE model pre-loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to pre-load TAPE model: {e}")
        print("TAPE model will be loaded on first prediction request")

    print("=" * 50)
    print("API started successfully!")
    print("=" * 50)


# ==================== 额外的辅助API端点 ====================

@app.get("/tasks")
async def get_all_tasks():
    """
    获取所有任务列表

    返回:
        所有任务的简要信息列表
    """
    task_list = []
    for taskid, task in tasks.items():
        input_info = task.input or {}
        task_list.append({
            "taskid": taskid,
            "status": task.status,
            "progress": task.progress,
            "progress_message": task.progress_message,
            "input_smi": input_info.get("smi"),
            "task_type": input_info.get("type"),
            "error": task.error
        })

    return {
        "total_tasks": len(task_list),
        "tasks": task_list
    }


@app.delete("/task/{taskid}")
async def delete_task(taskid: str):
    """
    删除指定任务

    参数:
        taskid: 任务ID

    返回:
        删除结果
    """
    if taskid not in tasks:
        raise HTTPException(status_code=404, detail="找不到该任务")

    del tasks[taskid]
    return {"message": f"任务 {taskid} 已删除"}


@app.post("/tasks/cleanup")
async def cleanup_completed_tasks():
    """
    清理已完成或失败的任务

    返回:
        清理结果统计
    """
    completed_count = 0
    failed_count = 0

    tasks_to_remove = []
    for taskid, task in tasks.items():
        if task.status == TaskStatus.COMPLETED:
            completed_count += 1
            tasks_to_remove.append(taskid)
        elif task.status == TaskStatus.FAILED:
            failed_count += 1
            tasks_to_remove.append(taskid)

    for taskid in tasks_to_remove:
        del tasks[taskid]

    return {
        "message": "任务清理完成",
        "completed_tasks_removed": completed_count,
        "failed_tasks_removed": failed_count,
        "total_removed": len(tasks_to_remove),
        "remaining_tasks": len(tasks)
    }


@app.get("/system/status")
async def get_system_status():
    """
    获取系统状态信息

    返回:
        系统运行状态和统计信息
    """
    # 统计各状态任务数量
    status_counts = {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0
    }

    for task in tasks.values():
        status_counts[task.status] += 1

    # 检查模型加载状态
    tape_loaded = tape_model_cache is not None
    cpi_models_loaded = list(loaded_cpi_models.keys())

    return {
        "system_health": "healthy",
        "total_tasks": len(tasks),
        "task_status_counts": status_counts,
        "device": str(DEVICE),
        "cuda_available": USE_CUDA,
        "tape_model_loaded": tape_loaded,
        "cpi_models_loaded": cpi_models_loaded,
        "protein_sequences_loaded": len(protein_sequences_cache),
        "admet_model_loaded": admet_model is not None,
        "timestamp": time.time()
    }


@app.get("/download/{taskid}/csv")
async def download_csv(taskid: str):
    """
    下载任务结果为CSV格式

    参数:
        taskid: 任务ID

    返回:
        CSV文件流
    """
    if taskid not in tasks:
        raise HTTPException(status_code=404, detail="找不到该任务")

    task = tasks[taskid]

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务未完成")

    if not task.results:
        raise HTTPException(status_code=400, detail="没有可下载的结果")

    # 创建CSV内容
    output = io.StringIO()
    writer = csv.writer(output)

    # 写入标题行
    writer.writerow(['Property', 'Value', 'Unit'])

    # 获取输入信息
    input_smi = task.input.get('smi', '') if task.input else ''
    writer.writerow(['SMILES', input_smi, ''])
    writer.writerow(['', '', ''])  # 空行分隔

    # 写入物理化学性质
    writer.writerow(['=== PHYSICOCHEMICAL PROPERTIES ===', '', ''])
    physicochemical_props = {
        'molecular_weight': ('Molecular Weight', 'g/mol'),
        'logP': ('cLogP', ''),
        'hydrogen_bond_acceptors': ('Hydrogen Bond Acceptors (HBA)', ''),
        'hydrogen_bond_donors': ('Hydrogen Bond Donors (HBD)', ''),
        'tpsa': ('Topological Polar Surface Area (TPSA)', 'Ų'),
        'stereo_centers': ('Stereo Centers', ''),
        'QED': ('Quantitative Estimation of Drug-likeness (QED)', ''),
        'Lipinski': ('Lipinski\'s Rule of Five', ''),
        'HydrationFreeEnergy_FreeSolv': ('Hydration Free Energy', 'kcal/mol')
    }

    for prop_key, (prop_name, unit) in physicochemical_props.items():
        value = task.results.get(prop_key, 'Unknown')
        if isinstance(value, (int, float)) and value != 'Unknown':
            value = f"{value:.2f}"
        writer.writerow([prop_name, value, unit])

    writer.writerow(['', '', ''])  # 空行分隔

    # 写入ADMET性质
    admet_sections = {
        '=== ABSORPTION ===': {
            'Solubility_AqSolDB': ('Aqueous Solubility', 'log mol/L'),
            'Lipophilicity_AstraZeneca': ('Lipophilicity', 'LogD'),
            'HIA_Hou': ('Human Intestinal Absorption (HIA)', '%'),
            'Caco2_Wang': ('Caco-2 Permeability', 'log Papp'),
            'PAMPA_NCATS': ('PAMPA Permeability', 'log Pe'),
            'Bioavailability_Ma': ('Bioavailability', '%')
        },
        '=== DISTRIBUTION ===': {
            'BBB_Martins': ('Blood-Brain Barrier Penetration', 'log BB'),
            'PPBR_AZ': ('Plasma Protein Binding Rate', '%'),
            'VDss_Lombardo': ('Volume of Distribution (VDss)', 'L/kg'),
            'Pgp_Broccatelli': ('P-glycoprotein Substrate', 'probability')
        },
        '=== METABOLISM ===': {
            'CYP1A2_Veith': ('CYP1A2 Inhibition', 'probability'),
            'CYP2C9_Veith': ('CYP2C9 Inhibition', 'probability'),
            'CYP2C19_Veith': ('CYP2C19 Inhibition', 'probability'),
            'CYP2D6_Veith': ('CYP2D6 Inhibition', 'probability'),
            'CYP3A4_Veith': ('CYP3A4 Inhibition', 'probability')
        },
        '=== EXCRETION ===': {
            'Clearance_Hepatocyte_AZ': ('Hepatocyte Clearance', 'μL/min/10^6 cells'),
            'Clearance_Microsome_AZ': ('Microsomal Clearance', 'μL/min/mg'),
            'Half_Life_Obach': ('Half-Life', 'hours')
        },
        '=== TOXICITY ===': {
            'AMES': ('AMES Toxicity', 'probability'),
            'DILI': ('Drug-Induced Liver Injury (DILI)', 'probability'),
            'hERG': ('Cardiac Toxicity (hERG)', 'probability'),
            'Carcinogens_Lagunin': ('Carcinogenicity', 'probability'),
            'Skin_Reaction': ('Skin Reaction', 'probability'),
            'ClinTox': ('Clinical Toxicity', 'probability'),
            'LD50_Zhu': ('Lethal Dose (LD50)', 'log mg/kg')
        }
    }

    for section_name, properties in admet_sections.items():
        writer.writerow([section_name, '', ''])
        for prop_key, (prop_name, unit) in properties.items():
            value = task.results.get(prop_key, 'Unknown')
            if isinstance(value, (int, float)) and value != 'Unknown':
                value = f"{value:.2f}"
            writer.writerow([prop_name, value, unit])
        writer.writerow(['', '', ''])  # 空行分隔

    # 添加时间戳
    from datetime import datetime
    writer.writerow(['Generated on', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ''])

    output.seek(0)

    # 创建响应
    response = StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),  # 使用BOM确保Excel正确显示中文
        media_type="application/csv",
        headers={"Content-Disposition": f"attachment; filename=molecular_properties_{taskid}.csv"}
    )

    return response


@app.get("/download/{taskid}/pdf")
async def download_pdf(taskid: str):
    """
    下载任务结果为PDF格式

    参数:
        taskid: 任务ID

    返回:
        PDF文件流
    """
    if taskid not in tasks:
        raise HTTPException(status_code=404, detail="找不到该任务")

    task = tasks[taskid]

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务未完成")

    if not task.results:
        raise HTTPException(status_code=400, detail="没有可下载的结果")

    # 创建PDF内容
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []

    # 获取样式
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # 居中
    )

    # 添加标题
    input_smi = task.input.get('smi', '') if task.input else ''
    story.append(Paragraph(f"Molecular Properties Report", title_style))
    story.append(Paragraph(f"SMILES: {input_smi}", styles['Normal']))
    story.append(Spacer(1, 20))

    # 创建数据表格
    def create_section_table(section_title, properties_dict):
        data = [[section_title, '', '']]
        data.append(['Property', 'Value', 'Unit'])

        for prop_key, (prop_name, unit) in properties_dict.items():
            value = task.results.get(prop_key, 'Unknown')
            if isinstance(value, (int, float)) and value != 'Unknown':
                value = f"{value:.2f}"
            data.append([prop_name, str(value), unit])

        table = Table(data, colWidths=[3.5 * inch, 1.5 * inch, 1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 1), (-1, 1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
            ('FONTNAME', (0, 2), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 1), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        return table

    # 物理化学性质
    physicochemical_props = {
        'molecular_weight': ('Molecular Weight', 'g/mol'),
        'logP': ('cLogP', ''),
        'hydrogen_bond_acceptors': ('Hydrogen Bond Acceptors (HBA)', ''),
        'hydrogen_bond_donors': ('Hydrogen Bond Donors (HBD)', ''),
        'tpsa': ('Topological Polar Surface Area (TPSA)', 'Ų'),
        'stereo_centers': ('Stereo Centers', ''),
        'QED': ('Quantitative Estimation of Drug-likeness (QED)', ''),
        'Lipinski': ('Lipinski\'s Rule of Five', ''),
        'HydrationFreeEnergy_FreeSolv': ('Hydration Free Energy', 'kcal/mol')
    }

    story.append(create_section_table('PHYSICOCHEMICAL PROPERTIES', physicochemical_props))
    story.append(Spacer(1, 20))

    # ADMET部分
    admet_sections = {
        'ABSORPTION': {
            'Solubility_AqSolDB': ('Aqueous Solubility', 'log mol/L'),
            'Lipophilicity_AstraZeneca': ('Lipophilicity', 'LogD'),
            'HIA_Hou': ('Human Intestinal Absorption (HIA)', '%'),
            'Caco2_Wang': ('Caco-2 Permeability', 'log Papp'),
            'PAMPA_NCATS': ('PAMPA Permeability', 'log Pe'),
            'Bioavailability_Ma': ('Bioavailability', '%')
        },
        'DISTRIBUTION': {
            'BBB_Martins': ('Blood-Brain Barrier Penetration', 'log BB'),
            'PPBR_AZ': ('Plasma Protein Binding Rate', '%'),
            'VDss_Lombardo': ('Volume of Distribution (VDss)', 'L/kg'),
            'Pgp_Broccatelli': ('P-glycoprotein Substrate', 'probability')
        },
        'METABOLISM': {
            'CYP1A2_Veith': ('CYP1A2 Inhibition', 'probability'),
            'CYP2C9_Veith': ('CYP2C9 Inhibition', 'probability'),
            'CYP2C19_Veith': ('CYP2C19 Inhibition', 'probability'),
            'CYP2D6_Veith': ('CYP2D6 Inhibition', 'probability'),
            'CYP3A4_Veith': ('CYP3A4 Inhibition', 'probability')
        },
        'EXCRETION': {
            'Clearance_Hepatocyte_AZ': ('Hepatocyte Clearance', 'μL/min/10^6 cells'),
            'Clearance_Microsome_AZ': ('Microsomal Clearance', 'μL/min/mg'),
            'Half_Life_Obach': ('Half-Life', 'hours')
        },
        'TOXICITY': {
            'AMES': ('AMES Toxicity', 'probability'),
            'DILI': ('Drug-Induced Liver Injury (DILI)', 'probability'),
            'hERG': ('Cardiac Toxicity (hERG)', 'probability'),
            'Carcinogens_Lagunin': ('Carcinogenicity', 'probability'),
            'Skin_Reaction': ('Skin Reaction', 'probability'),
            'ClinTox': ('Clinical Toxicity', 'probability'),
            'LD50_Zhu': ('Lethal Dose (LD50)', 'log mg/kg')
        }
    }

    for section_name, properties in admet_sections.items():
        story.append(create_section_table(section_name, properties))
        story.append(Spacer(1, 15))

    # 添加免责声明
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )

    from datetime import datetime
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Disclaimer: These predictions are free to use but should not be used for serious professional medical decisions.",
        disclaimer_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", disclaimer_style))

    # 构建PDF
    doc.build(story)
    buffer.seek(0)

    # 创建响应
    response = StreamingResponse(
        io.BytesIO(buffer.getvalue()),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=molecular_properties_{taskid}.pdf"}
    )

    return response


# quick test 输入
@app.post("/start-task")
async def start_task(item: Item, db: Session = Depends(get_db)):
    # 把数据传到模型
    id = generate_unique_task_id()

    task = process_task.delay(
        [item.smi, item.counters[0], item.counters[1], item.counters[2], item.counters[3], item.blocks, str(id)])

    # 将jobid、product、status存入数据库中的job
    job_data = {"jobid": id, "product": item.smi, "status": "pending"}
    job = schemas.JobCreate(**job_data)
    created_job = crud.create_job(db=db, job=job)

    # 给前端返回一个jobid
    return {"message": "Task started", "task_id": id}


# 查询指定 jobid 是否存在，如果存在就返回job的状态
@app.get("/job_status")
def job_data(jobid: str, db: Session = Depends(get_db)):
    # 调用 CRUD 操作函数查询数据库
    job = crud.get_job_by_id(db, jobid)
    # 如果 job 不存在，则返回 404 Not Found
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # 如果 job 存在，则返回 jobid、product 和 status
    return {
        "jobid": job.jobid,
        "product": job.product,
        "status": job.status
    }


# 查询指定 jobid 是否存在，如果存在就返回job的enzyme number
@app.get("/query_enzyme_info/")
def job_data(jobid: str, db: Session = Depends(get_db)):
    # 调用 CRUD 操作函数查询数据库
    job = crud.get_jobenzyme_by_jobid(db, jobid)
    # 如果 job 不存在，则返回 404 Not Found
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # 如果 job 存在，则返回 jobid、product 和 status
    res = [{
        "smi": j.smi,
        "ec_number": j.ec_number,
        "file_path": j.file_path
    } for j in job]
    return res


@app.post("/add_user_email/", response_model=schemas.UserEmailDisplay)
def add_user_email(user_email: schemas.UserEmailCreate, db: Session = Depends(get_db)):
    existing_email = crud.email_exists(db, user_email.email, user_email.jobid)
    if existing_email:
        return JSONResponse(content={"message": "Email already registered"})
    else:
        db_user_email = crud.create_user_email(db=db, user_email=user_email)
        # 返回成功信号
        return JSONResponse(content={"message": "success"})


@app.get("/check_and_send_email/{jobid}")
def check_job_and_send_email(jobid: str, db: Session = Depends(get_db)):
    """
    检查任务状态，如果完成则自动发送邮件
    可以用于前端轮询或定时任务
    """
    job = crud.get_job_by_id(db, jobid)
    
    if not job:
        return JSONResponse(
            status_code=404,
            content={"status": "not_found", "message": "Job not found"}
        )
    
    if job.status == "finished":
        # 检查是否已经发送过邮件（可选：添加一个 email_sent 字段到数据库）
        try:
            success_count = send_result_email(jobid, db)
            return JSONResponse(
                content={
                    "status": "finished",
                    "email_sent": True,
                    "emails_count": success_count
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "finished",
                    "email_sent": False,
                    "error": str(e)
                }
            )
    else:
        return JSONResponse(
            content={
                "status": job.status,
                "email_sent": False,
                "message": f"Job is still {job.status}"
            }
        )


@app.post("/send_email_with_attached/")
def send_email_with_attached(background_tasks: BackgroundTasks, request: EmailRequest, db: Session = Depends(get_db)):
    jobid = request.jobid
    email = request.email
    excel_file_path = None
    result = sorted(crud.inner_join_nodes_edges(db, jobid), key=lambda x: x.rxn_id)
    if not result:
        # background_tasks.add_task(back_send_email_with_timeout, jobid, email, db)
        return JSONResponse(content={}, status_code=200)  # 返回一个空的 JSON 对象
    # 将查询结果转换为字典列表
    data = []
    for item in result:
        data.append({
            'rxnid': item.rxn_id,
            'sub_smi': item.sub_smi,
            'prod_smi': item.prod_smi,
            'cost': item.cost,
            'pathway_list': item.pathway_list,
            'rxn_link': item.rxn_link,
            'sub_link': item.sub_link,
            'prod_link': item.prod_link
        })

    # 将字典列表转换为 DataFrame
    df = pd.DataFrame(data)

    # 生成临时 Excel 文件
    with NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
        excel_file_path = tmp_file.name
        df.to_excel(excel_file_path, index=False)

    # 添加后台任务，以便在请求结束后删除临时文件
    background_tasks.add_task(remove_file, excel_file_path)

    if send_email_file(email, jobid, excel_file_path):
        return JSONResponse(content={"message": "success"})
    else:
        return JSONResponse(content={"message": "failed"})


# 获取模型输出结果画图
@app.get("/query-nodes-and-edges/{jobid}")
def query_nodes_and_edges(jobid: str, db: Session = Depends(get_db)):
    # 查询节点
    queried_nodes = crud.get_nodes_by_jobid(db=db, jobid=jobid)

    # 删除节点中的 "jobid" 字段
    nodes_without_jobid = []
    for node in queried_nodes:
        node_dict = node.__dict__
        del node_dict['_sa_instance_state']  # 删除 SQLAlchemy 实例状态字段
        del node_dict['id']  # 删除 id 字段
        del node_dict['jobid']  # 删除 jobid 字段
        nodes_without_jobid.append(node_dict)

    # 查询边
    queried_edges = crud.get_edges_by_jobid(db=db, jobid=jobid)

    # 修改边的字段名，并删除部分字段
    edges_modified = []
    for edge in queried_edges:
        edge_dict = edge.__dict__
        del edge_dict['_sa_instance_state']  # 删除 SQLAlchemy 实例状态字段
        del edge_dict['id']  # 删除 id 字段
        del edge_dict['jobid']  # 删除 jobid 字段
        edge_dict['source'] = edge_dict.pop('sub_id')  # 将 sub_id 改为 source
        edge_dict['target'] = edge_dict.pop('prod_id')  # 将 prod_id 改为 target
        edge_dict['label'] = edge_dict.pop('cost')  # 将 cost 改为 label
        edges_modified.append(edge_dict)

    return {"nodes": nodes_without_jobid, "edges": edges_modified}


# 获取结果的excel文件
@app.get("/get_result_excel/{jobid}")
def get_result_excel(background_tasks: BackgroundTasks, jobid: str, db: Session = Depends(get_db)):
    result = sorted(crud.inner_join_nodes_edges(db, jobid), key=lambda x: x.rxn_id)
    if not result:
        return JSONResponse(content={}, status_code=200)  # 返回一个空的 JSON 对象

    # 将查询结果转换为字典列表
    data = []
    for item in result:
        data.append({
            'rxnid': item.rxn_id,
            'sub_smi': item.sub_smi,
            'prod_smi': item.prod_smi,
            'cost': item.cost,
            'pathway_list': item.pathway_list,
            'rxn_link': None, #item.rxn_link,
            'sub_link': item.sub_link,
            'prod_link': item.prod_link
        })

    # 将字典列表转换为 DataFrame
    df = pd.DataFrame(data)

    # 生成临时 Excel 文件
    with NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
        excel_file_path = tmp_file.name
        df.to_excel(excel_file_path, index=False)

    # 添加后台任务，以便在请求结束后删除临时文件
    background_tasks.add_task(remove_file, excel_file_path)

    # 返回 Excel 文件给前端
    return FileResponse(
        path=excel_file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="result.xlsx"
    )

