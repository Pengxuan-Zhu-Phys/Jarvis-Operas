# 内置函数单位参考

日期：2026-10-13。机读合同版本：`1.0`。

本次为 **64 个现有内置函数**声明可确定的规范单位：17 个普通内置函数和 47 个物理插值函数。数值实现、常数值、插值点与命名保持原有语义。`metadata.unit_contract` 是单位权威来源；普通调用缺少来源单位时按合同的规范单位解释，公开单位执行 API 在本次调用审计里记录 `assumed_canonical`。

## 逐项清单

| 内置函数 | 输入规范单位 | 输出规范单位与路径 | 说明 |
|---|---|---|---|
| `helper.eggbox` | `x`、`y`：`rad` | 根结果 `""`：`1` | 支持显式 `deg → rad`；保留标量、数组广播和 pandas 返回容器。 |
| `helper.eggbox2d` | `x`、`y`：`rad` | `/z`：`1` | 原有 `{"z": ...}` 完整结果独立保留，投影单独返回。 |
| `stat.chi2_cov` | 未设置固定输入单位 | 根结果 `""`：`1` | 残差和协方差取决于观测量；本期不推导不同残差的单位及协方差的单位乘积。保留单样本 float、多样本数组和 DataFrame 输入的 Series 输出。 |
| `cmb.loglike` | 直接关键字 `model_cl`、`noise_cl`：`uK^2` | `/loglike`：`1` | 数值数组合同；原函数继续校验 111 点的一维内部网格。`noise_cl=None` 被允许并保持为 None。 |
| `cmb.tt_cosmopower` | 直接关键字 `omegabh2`、`omegach2`、`thetaMC`、`tau`、`ns`、`As`、`h`：`1` | `/model_cl`：`uK^2`，形状 `[111]` | 源数据标题明确标注 `uK^2`；模型转换与 Planck `D_l → C_l` 不改数值。 |
| `pdg.mZ`、`pdg.mW`、`pdg.mt`、`pdg.mh`、`pdg.me`、`pdg.mmu`、`pdg.mtau` | 无输入，常数 | 根结果 `""`：`GeV` | 7 个质量常数，可投影为 MeV、TeV 等同维单位。 |
| `pdg.alphaEM`、`pdg.alphaSMZ` | 无输入，常数 | 根结果 `""`：`1` | 2 个无量纲耦合常数。 |
| `pdg.GF` | 无输入，常数 | 根结果 `""`：`GeV^-2` | 可以投影成 MeV^-2；不会根据自然单位自动换算为 pb。 |
| `pdg.hbarc` | 无输入，常数 | 根结果 `""`：`MeV·fm` | 明确登记复合单位，不据此自动推导能量与长度的物理转换。 |
| `pdg.c` | 无输入，常数 | 根结果 `""`：`m/s` | 明确登记速度单位。 |
| 以下 27 个 `dmddxn.*` | `x`：`GeV` | 根结果 `""`：`cm^2` | 可接收 TeV、MeV 质量并投影到 pb、fb；全部单位写入对应 manifest。 |
| 以下 20 个 `dmddxe.*` | `x`：`MeV` | 根结果 `""`：`cm^2` | 注意电子散射数据的横轴与核子散射不同，使用 MeV。 |

`dmddxn` 的 27 个函数：

- `LZSI2024`
- `XENONnTSI2025`
- `XENON1TMigdalSI2019`
- `PandaXSI2024`
- `CRESSTSI2019`
- `DarkSide50SI2022`
- `EDELWEISSSI2016`
- `DEAP3600SI2019`
- `PICO60SI2019`
- `COSINE100SI2018`
- `NEWSGSI2017`
- `LZSDp2024`
- `PICASSOSDp2012`
- `PICO60SDp2019`
- `CRESSTSDp2022`
- `XENONnTSDp2025`
- `PandaXSDp2024`
- `COUPPSDp2012`
- `NEWSGSDp2024`
- `XENONnTSDp2025Combined`
- `LZSDn2024`
- `XENONnTSDn2025`
- `PandaXSDn2024`
- `CDMSSDn2010`
- `CRESSTSDn2022`
- `CDMSliteSDn2019`
- `XENON1TMigdalSDn2019`

`dmddxe` 的 20 个函数：

- `CDEXLM2022`
- `DarkSide50LM2022`
- `CDMSHVeVLM2023`
- `SENSEILM2024Combined`
- `DAMICMLM2025Combined`
- `EDELWEISSLM2020`
- `PandaXLM2021`
- `XENON10LM2017`
- `XENON1TSolarReflectionLM2022`
- `AllLimitsLM2024`
- `CDEXHM2022`
- `DarkSide50HM2022`
- `CDMSHVeVHM2023`
- `SENSEIHM2024Combined`
- `DAMICMHM2025Combined`
- `EDELWEISSHM2020`
- `PandaXHM2021`
- `XENON1THM2019`
- `XENON1TSolarReflectionHM2022`
- `AllLimitsHM2024`

## 兼容策略与明确边界

- `math.add` 与 `math.identity` 补充单位说明，保持泛型行为，没有添加固定单位合同。`identity` 仍返回原对象；`add` 保留 Python/后端的既有加法语义。要求显式转换时返回 `contract_missing`，不会假定任意对象或物理数值都无量纲。
- `interp1.interp1_xy_flat`、`interp1.interp1_xlog_yflat`、`interp1.interp1_xflat_ylog`、`interp1.interp1_xy_log` 是没有物理单位来源的通用示例曲线，继续保持未声明单位。
- CMB 的单位绑定适用于直接数值关键字。旧 `payload`、`observables` 映射及其别名继续按函数原有规范值调用；合同 v1 不提供映射内字段的单位绑定。显式或严格单位调用不能同时携带这些映射（包括空映射），避免它们覆盖已经规范化的参数；需要单位处理时，应把映射中的配置与数值展开为直接关键字。无法证明嵌套来源或有覆盖歧义时，在执行前报 `contract_missing`。
- 插值的冷加载与热缓存注册复用 manifest 的同一合同，不另外按函数名称推断单位。有单位合同的物理曲线现在在执行前拒绝 NaN/Inf 输入，报 `non_finite`；有限数值在区间外返回 NaN 的原有策略继续由插值函数决定。单位投影也要求有限数值，NaN 输出会报 `non_finite`。
- PDG 的旧 `metadata.unit` 仅为显示兼容别名，由根输出合同派生；常数声明时两个声明冲突会被拒绝。未提供单位的自定义常数仍没有单位合同。
- HTs、GMFit 和 Jarvis-HEP 的函数实现与 YAML schema 没有在此任务中修改。HTs 的结构化预测不能推断成质量标量，需后续明确声明具体数值输出字段。

## 单位来源

- CMB：本仓库 `jarvis_operas/models/cmb/planck/data/TT_power_spec.txt` 第一行明确给出功率的 `uK^2` 单位；`namespaces/cmb/defs/core.py` 以 `D_l * 2π / [ell(ell+1)]` 转成 `C_l`，并将模型乘以 `10^12 * 2.725^2`。
- DM 电子：原始数据来源仓库的 [DM-Electron.ipynb](https://github.com/cajohare/DirectDetectionPlots/blob/main/notebooks/DM-Electron.ipynb) 中轻/重媒介轴标签与合并曲线 header 给出 `MeV` 和 `cm^2`。声明使用现有数据的数值质量轴 convention，与需求中的能量/质量单位族一致。
- DM 核子：[PICO60 数据 header](https://github.com/cajohare/DirectDetectionPlots/blob/main/data/limits/Nucleon/SDp/PICO60.txt) 与 [LZ 2024 SDn 数据 header](https://github.com/cajohare/DirectDetectionPlots/blob/main/data/limits/Nucleon/SDn/LZ_2024.txt) 给出 `GeV` 和 `cm^2`，对应本仓库 manifest 中的来源与散射面板 convention。
- PDG：现有常数声明的单位与数值迁移进合同，没有改变或更新 PDG 数据年份。

公共 API、完整 schema、默认策略和 CLI 示例见[单位合同说明](unit_contract.md)。
