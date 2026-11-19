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
# crud.py
# CRUD操作（与数据库交互）
from sqlalchemy.orm import aliased, Session
from sqlalchemy import and_
import models
import schemas
from sqlalchemy.exc import SQLAlchemyError


# 查询Blocks表中的所有数据
def get_blocks(db: Session):
    try:
        blocks = db.query(models.Blocks).all()
        return blocks
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于Job表，增加一条数据
def create_job(db: Session, job: schemas.JobCreate):
    try:
        db_job = models.Job(**job.dict())
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return db_job
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于Job表，根据jobid，查询数据
def get_job_by_id(db: Session, jobid: str):
    try:
        job = db.query(models.Job).filter(models.Job.jobid == jobid).first()
        return job
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于Job表，根据jobid，查询数据
def get_jobenzyme_by_jobid(db: Session, jobid: str):
    try:
        job = db.query(models.JobEnzyme).filter(models.JobEnzyme.jobid == jobid).all()
        return job
    except SQLAlchemyError as e:
        db.rollback()
        raise e

# 对于Job表，根据jobid，修改status
def update_job_status(db: Session, jobid: str, status: str):
    try:
        db_job = db.query(models.Job).filter(models.Job.jobid == jobid).first()
        if db_job:
            db_job.status = status
            db.commit()
            return db_job
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 判断是否已经存在这个jobid和邮箱
def email_exists(db: Session, email: str, jobid: str):
    try:
        user_email = db.query(models.UserEmail).filter(models.UserEmail.email == email,
                                                       models.UserEmail.jobid == jobid).first()
        return user_email
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于UserEmail表，增加一条数据
def create_user_email(db: Session, user_email: schemas.UserEmailCreate):
    try:
        db_user_email = models.UserEmail(**user_email.dict())
        db.add(db_user_email)
        db.commit()
        db.refresh(db_user_email)
        return db_user_email
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于UserEmail表，根据jobid，查询email
def get_emails_by_jobid(db: Session, jobid: str):
    try:
        emails = db.query(models.UserEmail).filter(models.UserEmail.jobid == jobid).all()
        return emails
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于Nodes表，增加多条数据
def create_nodes(db: Session, nodes_create: schemas.NodesCreate):
    try:
        db_nodes = [models.Nodes(**node.dict()) for node in nodes_create.nodes]
        db.add_all(db_nodes)
        db.commit()
        for node in db_nodes:
            db.refresh(node)
        return db_nodes
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于Nodes表，根据jobid，查询数据信息
def get_nodes_by_jobid(db: Session, jobid: str):
    try:
        nodes = db.query(models.Nodes).filter(models.Nodes.jobid == jobid).all()
        return nodes
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于Edges表，增加多条数据
def create_edges(db: Session, edges_create: schemas.EdgesCreate):
    try:
        db_edges = [models.Edges(**edge.dict()) for edge in edges_create.edges]
        db.add_all(db_edges)
        db.commit()
        for edge in db_edges:
            db.refresh(edge)
        return db_edges
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于Edges表，根据jobid，查询数据
def get_edges_by_jobid(db: Session, jobid: str):
    try:
        edges = db.query(models.Edges).filter(models.Edges.jobid == jobid).all()
        return edges
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于Nodes表和Edges表，根据jobid，进行左外连接查询
def inner_join_nodes_edges(db: Session, jobid: str):
    try:
        # 使用别名创建两个Nodes模型的副本
        SubNodes = aliased(models.Nodes)
        ProdNodes = aliased(models.Nodes)

        # 执行内连接查询
        result = (
            db.query(
                models.Edges.rxnid.label('rxn_id'),
                models.Edges.cost.label('cost'),
                models.Edges.pathway_list.label('pathway_list'),
                SubNodes.smi.label('sub_smi'),
                SubNodes.node_link.label('sub_link'),
                ProdNodes.smi.label('prod_smi'),
                ProdNodes.node_link.label('prod_link'),
                models.Edges.rxn_link.label('rxn_link')
            )
            .join(SubNodes, and_(models.Edges.jobid == SubNodes.jobid, models.Edges.sub_id == SubNodes.nodeid))
            .join(ProdNodes, and_(models.Edges.jobid == ProdNodes.jobid, models.Edges.prod_id == ProdNodes.nodeid))
            .filter(models.Edges.jobid == jobid)
            .all()
        )
        return result
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 对于JobEnzyme表，增加多条数据
def create_jobenzyme(db: Session, jobenzyme_create: schemas.JobEnzymeCreate):
    job_id = jobenzyme_create.jobenzyme[0].jobid
    try:
        db_enzymes = db.query(models.JobEnzyme).filter(models.JobEnzyme.jobid == job_id).all()
        # 删除旧的记录
        for enz in db_enzymes:
            db.delete(enz)

        db_enz = [models.JobEnzyme(**enz.dict()) for enz in jobenzyme_create.jobenzyme]
        db.add_all(db_enz)
        db.commit()
        for enz in db_enz:
            db.refresh(enz)
        return db_enz
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# bkms_smi 根据 smi 查询 ID
def get_bkms_id_by_smi(db: Session, smi: str):
    try:
        bkms_smi = db.query(models.BkmsSmi).filter(models.BkmsSmi.smi == smi).first()
        return bkms_smi.id if bkms_smi else None
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# bkms_smi 根据 ID 查询 smi
def get_bkms_smi_by_id(db: Session, id: int):
    try:
        bkms_smi = db.query(models.BkmsSmi).filter(models.BkmsSmi.id == id).first()
        return bkms_smi.smi if bkms_smi else None
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 根据给定的分子 ID 查询 bkms_rxn 表中包含该 ID 的反应式
def get_bkms_rxns_by_smi_id(db: Session, smi_id: int):
    try:
        return db.query(models.BkmsRxn).filter(models.BkmsRxn.rxn.like(f'%,{smi_id},%')).all()
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# yeast_smi 根据 smi 查询 ID
def get_yeast_id_by_smi(db: Session, smi: str):
    try:
        yeast_smi = db.query(models.YeastSmi).filter(models.YeastSmi.smi == smi).first()
        return yeast_smi.id if yeast_smi else None
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# yeast_smi 根据 ID 查询 smi
def get_yeast_smi_by_id(db: Session, id: int):
    try:
        yeast_smi = db.query(models.YeastSmi).filter(models.YeastSmi.id == id).first()
        return yeast_smi.smi if yeast_smi else None
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 根据给定的分子 ID 查询 yeast_rxn 表中包含该 ID 的反应式
def get_yeast_rxns_by_smi_id(db: Session, smi_id: int):
    try:
        return db.query(models.YeastRxn).filter(models.YeastRxn.rxn.like(f'%,{smi_id},%')).all()
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# uspto50k_smi 根据 smi 查询 ID
def get_uspto50k_id_by_smi(db: Session, smi: str):
    try:
        uspto50k_smi = db.query(models.Uspto50kSmi).filter(models.Uspto50kSmi.smi == smi).first()
        return uspto50k_smi.id if uspto50k_smi else None
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# uspto50k_smi 根据 ID 查询 smi
def get_uspto50k_smi_by_id(db: Session, id: int):
    try:
        uspto50k_smi = db.query(models.Uspto50kSmi).filter(models.Uspto50kSmi.id == id).first()
        return uspto50k_smi.smi if uspto50k_smi else None
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# 根据给定的分子 ID 查询 uspto50k_rxn 表中包含该 ID 的反应式
def get_uspto50k_rxns_by_smi_id(db: Session, smi_id: int):
    try:
        return db.query(models.Uspto50kRxn).filter(models.Uspto50kRxn.rxn.like(f'%,{smi_id},%')).all()
    except SQLAlchemyError as e:
        db.rollback()
        raise e
