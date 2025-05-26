from setuptools import find_packages, setup
import os # os的导入在这里不是必需的，因为你用的是字符串拼接路径，但保留也无妨
from glob import glob

package_name = 'hexapod_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']), # 标准写法，如果你的包里有Python模块会找到它们
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]), # 标准的ament索引文件安装
        ('share/' + package_name, ['package.xml']), # 安装package.xml
        ('share/' + package_name + '/urdf', glob('urdf/*.urdf')), # 安装urdf目录下所有.urdf文件
        ('share/' + package_name + '/meshes', glob('meshes/*.stl')), # 安装meshes目录下所有.stl文件
    ],
    install_requires=['setuptools'], # 标准依赖
    zip_safe=True, # 标准设置
    maintainer='lee',
    maintainer_email='orangexy06@gmail.com',
    description='TODO: Package description', # 建议填写
    license='TODO: License declaration', # 建议填写
    tests_require=['pytest'], # 标准测试依赖
    entry_points={
        'console_scripts': [
            # 如果你有Python可执行节点，在这里添加
        ],
    },
)
