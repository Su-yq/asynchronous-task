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
# model_output.py
# 模型输入之后进行的操作

# 1.模型输出
# 2.数据处理、存入数据库
# 3.修改job的status
# 4.发送邮件(邮件标题、内容待编辑)
import os
import tempfile
import data_processing
import crud
from database1 import SessionLocal
from email_sender import send_email_file
import pandas as pd


def model_output(jobid, data_list):
    # 调用数据处理函数
    a = data_processing.data_processing_function(jobid, data_list)

    # 根据返回值判断存储是否成功
    if a:
        print("数据存储成功！")
    else:
        print("数据存储失败！")

    # 修改job的status
    # 创建数据库会话
    db = SessionLocal()

    try:
        # status = finished
        updated_job = crud.update_job_status(db=db, jobid=jobid, status="finished")

        if updated_job:
            print(f"Updated job: {updated_job.jobid}, New Status: {updated_job.status}")
        else:
            print(f"No job found with jobid: {jobid}")

    finally:
        # 关闭数据库会话
        db.close()


def send_result_email(jobid, db):
    """
    根据 jobid 发送结果邮件（带 Excel 附件）
    这个函数在主函数（FastAPI）中调用，避免 Celery worker 的网络限制问题
    
    Args:
        jobid: 任务ID
        db: 数据库会话对象
    
    Returns:
        发送成功的邮件数量
    """
    # 根据 jobid 查询用户电子邮件
    queried_emails = crud.get_emails_by_jobid(db=db, jobid=jobid)
    
    if not queried_emails:
        print(f"No emails found for job ID: {jobid}")
        return 0
    
    print(f"Queried emails for job ID {jobid}:")
    
    result = sorted(crud.inner_join_nodes_edges(db, jobid), key=lambda x: x.rxn_id)
    
    if result:
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
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            excel_file_path = tmp_file.name
            df.to_excel(excel_file_path, index=False)
        
        try:
            success_count = 0
            for email_obj in queried_emails:
                email = email_obj.email
                print(f"Sending email to: {email}, Job ID: {jobid}")
                
                result = send_email_file(email, jobid, excel_file_path)
                if result:
                    success_count += 1
            
            return success_count
        finally:
            if os.path.exists(excel_file_path):
                os.remove(excel_file_path)
                print(f"已删除临时文件: {excel_file_path}")
    else:
        print(f"No data found for job ID: {jobid}")
        return 0


if __name__ == '__main__':
    # pass
    # 调用函数
    # data_list = ['CC(=O)NCCCCC(N)C(=O)O>0.7309$Temp>NCCCCC(N)C(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1056$Free>NCCCC[C@H](N)C(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1021$Free>NCCCCC(N)C(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1990$Temp>CC(=O)NCCCCC(=O)C(=O)O|CC(=O)NCCCCC(=O)C(=O)O>0.2533$Temp>CC(=O)NCCCC[C@H](N)C(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1154$Free>NCCCCC(=O)C(=O)O|NCCCCC(=O)C(=O)O>0.4156$Temp>NCCCC[C@H](N)C(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1067$Free>CC(=O)NCCCCC(=O)C(=O)O|CC(=O)NCCCCC(=O)C(=O)O>0.2533$Temp>CC(=O)NCCCC[C@H](N)C(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1990$Temp>CC(=O)NCCCCC(=O)C(=O)O|CC(=O)NCCCCC(=O)C(=O)O>0.1170$Free>CC(=O)NCCCC[C@H](N)C(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1990$Temp>CC(=O)NCCCCC(=O)C(=O)O|CC(=O)NCCCCC(=O)C(=O)O>0.2614$Temp>CC(=O)NCCCCC(=O)O|CC(=O)NCCCCC(=O)O>0.3951$Temp>NCCCCC(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1154$Free>NCCCCC(=O)C(=O)O|NCCCCC(=O)C(=O)O>0.1421$Temp>NCCCC[C@@H](N)C(=O)O',
    #              'CC(=O)NCCCCC(N)C(=O)O>0.1154$Free>NCCCCC(=O)C(=O)O|NCCCCC(=O)C(=O)O>0.1226$Free>NCCCC[C@H](N)C(=O)O']

    data_list = ['C(CO)CO>0.7743_Temp>O=CCCO|O=CCCO>0.4552_Temp>OCCCO',
                 'C(CO)CO>0.7743_Temp>O=CCCO|O=CCCO>0.1239_Temp>NCCCO',
                 'C(CO)CO>0.7743_Temp>O=CCCO|O=CCCO>0.1075_Temp>O=C(O)CCO', 'C(CO)CO>0.0522_Temp>CCCO',
                 'C(CO)CO>0.0366_Free>O=C(O)CCO', 'C(CO)CO>0.0345_Temp>O=C(O)CCO', 'C(CO)CO>0.0312_Temp>OCC(O)CO',
                 'C(CO)CO>0.7743_Temp>O=CCCO|O=CCCO>0.0392_Free>O=C(O)CCO',
                 'C(CO)CO>0.7743_Temp>O=CCCO|O=CCCO>0.1176_Temp>O=C(O)C(=O)CCO|O=C(O)C(=O)CCO>0.3247_Temp>N[C@@H](CCO)C(=O)O',
                 'C(CO)CO>0.0276_Free>O=C/C=C/C=O']

    jobid = '5d680653-990b-4019-948b-1b37f0259133'

    model_output(jobid, data_list)
    # model_output('9d27975b-1764-46f4-afc7-36360c7039e9', ['O=C(O)CCCCC(=O)O>0$NULL>O=C(O)CCCCC(=O)O'])
