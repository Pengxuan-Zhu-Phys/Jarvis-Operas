# Jarvis-Operas 维护者职责与函数接入手册（中文）

版本基线：JO 1.3.x
适用对象：JO 日常维护者（非架构负责人）

## 1. 文档目标

本手册只定义一类工作：
1. 新增 namespace。
2. 在 namespace 下新增 functions。
3. 为每个新增函数补齐测试并通过回归。

本手册不允许维护者改动 JO 核心架构。维护者的目标是“稳定扩展函数库”，不是“重写系统设计”。

## 2. 职责边界

### 2.1 维护者负责
1. 在 `jarvis_operas/namespaces/` 下维护函数实现与声明。
2. 保证每个新增函数满足 numpy/polars/sync/async 调用契约。
3. 保证每个新增函数有清晰 metadata（summary/params/examples 等）。
4. 为每个新增函数新增或更新测试，并在本地通过。
5. 保持命名、目录、代码风格一致。

### 2.2 维护者不负责
1. 不修改 `jarvis_operas/core/*` 的架构设计。
2. 不修改全局注册生命周期与 bootstrap 机制（`jarvis_operas/api.py` 主流程）。
3. 不修改后端适配核心策略（如 polars fallback 总体机制）。
4. 不改变持久化协议结构（`sources.json/overrides.json` 格式）。
5. 不引入新重依赖、不做大规模模块重构。

如果确实发现架构级问题，只提交 issue 描述，不直接改架构。

## 3. JO 当前可扩展稳定入口

维护者只应在以下位置做增量：
1. `jarvis_operas/namespaces/<namespace>/defs/*.py`
2. `jarvis_operas/namespaces/<namespace>/decls/*.py`
3. `jarvis_operas/namespaces/<namespace>/decls/__init__.py`
4. `jarvis_operas/namespaces/<namespace>/__init__.py`
5. `jarvis_operas/catalog/index.py`（仅添加 namespace 声明映射）
6. `tests/` 中对应测试文件

不要在其他位置“偷注册”函数。

## 4. 标准目录规范

新增 namespace 时，目录应为：

```text
jarvis_operas/namespaces/<namespace>/
  __init__.py
  defs/
    __init__.py
    <topic>.py
  decls/
    __init__.py
    <topic>.py
```

约定：
1. `defs` 只放实现逻辑。
2. `decls` 只放 `OperaFunction` 声明与 metadata。
3. `__init__.py` 只做导出聚合，不写业务逻辑。

## 5. 新增 namespace 操作流程

1. 创建 namespace 目录与 `defs/decls` 子目录。
2. 在 `defs` 写函数实现（numpy 必须可用）。
3. 在 `decls` 写 `OperaFunction` 声明。
4. 在 `decls/__init__.py` 导出该 namespace 的 `DECLARATIONS`。
5. 在 `namespace/__init__.py` 暴露统一 `DECLARATIONS`。
6. 在 `jarvis_operas/catalog/index.py` 注册新 namespace 到 `NAMESPACE_DECLARATIONS`。
7. 添加测试。
8. 跑测试并修复。

## 6. 新增 function 操作流程

### 6.1 实现层（defs）要求
1. 至少提供 `numpy_impl`。
2. numpy 实现必须支持标量和数组输入场景。
3. numpy 实现应尽量向量化，避免逐元素 Python 循环。
4. 可选提供 `polars_expr_impl`（优先推荐）。

### 6.2 声明层（decls）要求
每个函数必须声明为 `OperaFunction`，至少包含：
1. `namespace`
2. `name`
3. `arity`
4. `numpy_impl`
5. `return_dtype`（若无 polars native 且需要 fallback，则必须填写）
6. `metadata`

建议 metadata 字段：
1. `category`
2. `summary`
3. `params`
4. `return`
5. `backend_support`
6. `examples`
7. `since`
8. `tags`
9. `note`（可选）

### 6.3 关于 polars
1. 有 `polars_expr_impl` 时，优先走 native expr。
2. 无 `polars_expr_impl` 时，JO 会走 numpy fallback（`map_batches`）。
3. fallback 路径必须有 `return_dtype`，否则会报错。
4. 严禁使用 `map_elements` 作为 fallback 主路径。

### 6.4 关于 async
1. 普通同步 numpy 实现即可，`acall` 会通过 registry 线程卸载。
2. 不要在函数内部自行管理线程池。
3. 不要写阻塞事件循环的 async 假实现。

## 7. 代码模板

### 7.1 defs 模板

```python
# jarvis_operas/namespaces/demo/defs/basic.py
from __future__ import annotations

import numpy as np


def scale_numpy(x, factor=1.0):
    arr = np.asarray(x)
    return arr * factor


def scale_polars_expr(x, factor=1.0):
    return x * factor
```

### 7.2 decls 模板

```python
# jarvis_operas/namespaces/demo/decls/basic.py
from __future__ import annotations

from ....core.spec import OperaFunction
from ..defs.basic import scale_numpy, scale_polars_expr

DEMO_BASIC_DECLARATIONS: tuple[OperaFunction, ...] = (
    OperaFunction(
        namespace="demo",
        name="scale",
        arity=1,
        return_dtype=None,
        numpy_impl=scale_numpy,
        polars_expr_impl=scale_polars_expr,
        metadata={
            "category": "demo",
            "summary": "Scale input by factor.",
            "params": {"x": "Input value", "factor": "Scale factor"},
            "return": "Scaled value",
            "backend_support": {"numpy": "native", "polars": "native_expr"},
            "examples": ["jopera call demo.scale --kwargs '{\"x\":2,\"factor\":3}'"],
            "since": "1.3.x",
            "tags": ["demo", "scale"],
        },
    ),
)
```

### 7.3 namespace 聚合模板

```python
# jarvis_operas/namespaces/demo/decls/__init__.py
from .basic import DEMO_BASIC_DECLARATIONS

DECLARATIONS = DEMO_BASIC_DECLARATIONS
```

```python
# jarvis_operas/namespaces/demo/__init__.py
from .decls import DECLARATIONS

__all__ = ["DECLARATIONS"]
```

### 7.4 catalog 注册模板

```python
# jarvis_operas/catalog/index.py
from ..namespaces.demo import DECLARATIONS as DEMO_DECLARATIONS

NAMESPACE_DECLARATIONS = {
    # ...existing
    "demo": DEMO_DECLARATIONS,
}
```

## 8. 测试要求（每个新增函数必须覆盖）

每新增 1 个函数，至少覆盖以下测试项：
1. `call` 的 numpy 基本正确性。
2. 标量输入正确性。
3. 数组输入正确性。
4. `acall` 正常返回且结果与 `call` 一致。
5. 有 polars native 时：expr 路径正确。
6. 无 polars native 时：fallback 路径正确（且声明了 `return_dtype`）。
7. 错误输入时有清晰报错（例如 arity 不符）。
8. `info` 可读到 metadata 关键字段。

建议测试文件位置：
1. 通用后端行为：`tests/test_operas_backends.py`
2. namespace 相关行为：新增 `tests/test_<namespace>.py`

## 9. 本地验证命令（最小集合）

```bash
pytest -q tests/test_public_api_contract.py tests/test_operas_backends.py tests/test_async.py
pytest -q tests/test_loading.py tests/test_loading_operas.py tests/test_persistence.py
pytest -q tests/test_cli.py
```

涉及插值或曲线相关函数时追加：

```bash
pytest -q tests/test_curves.py
```

发布前完整验证：

```bash
pytest -q
```

## 10. 提交规范

1. 一个 namespace 的新增放一个 commit（或按功能拆成实现+测试两个 commit）。
2. commit message 清楚说明 namespace/function 名称。
3. 必须先有测试再合入。
4. 不提交与本任务无关的文件。

推荐 message 风格：
1. `feat(namespace): add demo.scale declaration and implementations`
2. `test(namespace): add demo.scale numpy/polars/async coverage`

## 11. 维护者交付清单（Checklist）

每次提交前自检：

- [ ] 函数放在正确 namespace 目录下（defs/decls 分离）。
- [ ] 使用 `OperaFunction` 声明，未绕过注册体系。
- [ ] metadata 完整，`examples` 可直接用于 CLI。
- [ ] numpy 路径已测。
- [ ] polars 路径已测（native 或 fallback）。
- [ ] async 路径已测（至少一条）。
- [ ] `pytest` 相关用例通过。
- [ ] 未改动 core/api 架构文件。

## 12. 升级与异常处理原则

1. 发现架构问题时，先记录问题与复现步骤，不直接在维护分支重构核心。
2. 如果新增函数需要新的通用机制，先提“架构需求单”，待架构负责人评审。
3. 维护者分支必须保持“可回滚、可定位、可测试”。

