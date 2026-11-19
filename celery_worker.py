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
from celery import Celery
from retro_plan.retro_plan_fun import retro_plan
# from one_step_topk import run
from utils import Item
from model_output import model_output
import logging

logging.basicConfig(
    filename='/home/liuxiaoyi/bio_chem_crow_fastapi-new/api.log',  # 固定指向api.log
    level=logging.INFO,  # 日志级别：INFO及以上都会记录
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 创建日志实例
logger = logging.getLogger('celery_worker')


celery_app = Celery(
    "worker",
    backend="redis://localhost:6379/0",
    broker="redis://localhost:6379/0"
)

# celery -A celery_worker worker -l INFO

# Pathway_top_k
# Expansion_iters
# Expansion_top_k
# Expansion_time(s)
# Max_depth


@celery_app.task
def process_task(item: list):

    jobid = item[6]

    input_dict = {
        'target_mol': item[0],  # input molecule
        'route_topk': item[1],
        'expansion_iters': item[2],
        'expansion_topk': item[3],
        'max_depth': item[4],
        'building_blocks': item[5]
    }
    res = retro_plan(input_dict)
    logger.info(f"任务处理完成 | jobid: {jobid} | res结果: \n{res}")
    model_output(jobid, res)

    return res