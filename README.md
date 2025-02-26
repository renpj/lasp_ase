# lasp_ase
ASE calculator for LASP (http://www.lasphub.com)

## Installation

### 环境配置
* Linux 环境。没有测试过 windows 环境，只要 lasp 二进制支持就可以。
* 安装 intel mpi 编译器 （lasp 运行需要），参考 intel oneapi 官网。
* 从 LASP 官网下载 lasp 可执行文件和相关的势函数 （需要许可）。
* 在 .bashrc 中配置 `ASE_LASP_COMMAND = mpirun -np 4 lasp` （根据自己的情况修改）。若是使用 pbs 脚本则可以在脚本中加入相应配置。

### 安装

使用 `pip install -U lasp_ase`

## run example
```bash
cd example
bash run_ase.sh
```
