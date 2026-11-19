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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# 数据库连接配置
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:Aa123456.@localhost:3306/bio_chem_crow"

# 创建数据库引擎
engine = create_engine(SQLALCHEMY_DATABASE_URL)
# 创建会话类
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 测试连接
try:
    # 尝试建立连接
    with engine.connect() as connection:
        print("连接成功！")
except OperationalError as e:
    print(f"连接失败: {e}")
