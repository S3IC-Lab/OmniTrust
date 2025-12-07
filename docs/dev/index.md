# 新仓库使用说明

**请大家逐步把代码移到此分支上**，原先仓库因为很多组件耦合较乱、重复，将逐渐弃用。再整理完开源版本后，我将基于此仓库fork一个private版本供商业化使用。

## 代码组件更新

通过统计各组使用代码，新的公共组件已经放置在main分支上，请大家在此公共组件上进行开发！！

### API config更新
* 通过查看代码，我看到很多组的API调用混乱。
* 我在/config文件夹下建立了`api_config.yaml`文档，此文档供大家填入API信息，**但是注意此文件的修改不要commit到分支上**
* 调用方法：
```python
from config import GET_API, LOAD_API_CONFIG
cfg_default = LOAD_API_CONFIG()
print(cfg_default["api"]["active"])

# Load specific model
cfg_ds = LOAD_API_CONFIG("deepseek-chat")
print(cfg_ds["api"]["active"])

key, url = GET_API("deepseek-chat")
print("KEY:", key)
print("URL:", url)
```
* 在公共组件中我已经更改这部分调用代码，例如`models/llm.py`, `utils/api/openai_api`等文件
* **请大家在自己的调用方法中更改为此！！**

### `models`文件
* 多模态文件组`lvlm`移动至`models`下，请有需要的组更改调用路径
```python
from model import InternVL, LLaVA, LLaVANeXT, Qwen2VL
```
* 原`utils/llm.py`移动至`models`下，该模块集成huggingface、openai、ollama管理框架，若有需要可直接调用

### 路径说明
目前仍有很多文件里用的是direct的路径说明，请修改成匿名化！！

测试时若进行修改，请务必不要将个性化的model path等直接上传至仓库

## 数据集更新
我在hugging-face上直接新建了一个S3IC的仓库，数据集统一放到这个仓库中。[S3IC Hugging face](https://huggingface.co/S3IC)

[点击这个link](https://huggingface.co/organizations/S3IC/share/jbcNelnRmzSlhjTDrkcNlHwlkixCDLOAWT)可以加入我们的organization

新建数据集并上传，示例：
![img.png](assets/img.png)

### 你需要完善两个readme文档
1. hugging-face上的 写明出处、用法等
![img_1.png](assets/img_1.png)
2. 在`data/datasets/DATASET_NAME`路径中，添加`README.md`展示具体下载方法 
* 例如我放在`data/datasets/advbench`下的README文件

* **ATTENTION** 测试的时候下载到本地的文件，一定不要上传至Github上！！！

## 数据集代码更新
将原来`/data`下的代码统一整理到`/data_registry`文件夹下，大家注意更新

写完数据集类后在`data/__init__.py`文件中加入
```python
from .data_registry.HumanEval import HumanEvalDataset
```
这边可以参考`data/README.md`文件

## 文档更新
新的文档体系已经用Github Pages建立（就是你现在正在浏览的）

请大家在完善完方法之后，将方法说明增添到对应位置。

更新位置：`docs/各个组名`。我已经在`docs/各个组名`下建立`index.md`文档以及`assets`文件夹；`assets`文件夹用来给大家放相关图片。

**请组长更新`index.md`文档，完善本module具体介绍**

### 新建方法
在`docs/各个组名/`下新建对应方法即可。例如`docs/safety/cipherchat.md`（这个仍需相关同学再次修改，我写的很简约）

**新的方法使用说明请base本仓库的使用重新修改，后续仍会进行测试**

**为了避免混乱，请组长统一新建`methodA.md`,`methodB.md`文件**
* 在本module `index.md`中引用该方法；
* 组长请在`/根目录/mkdocs.yml`中将`methodA.md`,`methodB.md`加入对应位置，否则无法渲染！！
```yaml
# Navigation
nav:
  - Home: index.md
  - Develop: dev/index.md
  - Safety Module:
      - Overview: safety/index.md
      - CipherChat: safety/cipherchat.md
      - GCG: safety/gcg.md
      - AutoDAN: safety/autodan.md
```

### 预览以及渲染
本地预览：
```shell
pip install mkdocs-material
mkdocs serve
```

在push到main branch后会在Github Actions构建新的页面->完成主页面更新。

## push权限

* 目前只给各位组长开通了main分支的push以及merge权限
* 组员更新到自己的branch上->组长检查并merge

## requirements.txt文件
目前requirements.txt是可维持系统运行的最小单元
若某module需要新的requirements文件可自行新建并上传，并在各自module的index.md文档进行说明