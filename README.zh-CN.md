**CTA 客流预测：保留的三个最佳模型**

已按 **2024 年验证集 WMAPE** 为每个模型类别保留一个最佳版本。测试集为 2025-01-01 至 2025-11-30，仅用于最终报告。该测试期在之前的分析中已被查看，因此这些结果属于历史基准，不是新的前瞻性验证。

| 保留模型 | 验证 WMAPE | 测试 WMAPE | 测试 MAE | 测试 RMSE |
|---|---:|---:|---:|---:|
| XGBoost | **7.77%** | **8.10%** | **175.26** | **498.52** |
| Residual LSTM | 8.60% | 9.00% | 194.64 | 523.16 |
| GCN–LSTM | 8.63% | 9.13% | 197.59 | 527.51 |

三个模型现在使用相同的 143 个站点定义，比较覆盖相同的 52,156 个验证目标和 47,762 个测试目标。原始旧 LSTM 的站点定义不同；在 51,762 个共同且目标一致的验证样本上，旧 LSTM WMAPE 为 9.47%，保留的 Residual LSTM 为 8.67%。因此选用新的无跨站点混合版本作为 LSTM，避免仅按不同样本集合的指标进行选择。

这里的 LSTM 是上一轮 identity-adjacency 实验胜出的残差时序模型：邻接矩阵为单位阵，每个站点仅处理自身信息；静态网络特征仍保留。GCN–LSTM 使用实际轨道图进行跨站点聚合。两者共享实现和参数结构，但模型文件、配置和结果分别保存。

**XGBoost 改进**

共比较五种预设方案：log 平方损失、log 季节残差、原始客流绝对误差（深度 6/8）、原始客流平方误差。所有方案按验证集原始客流 WMAPE 早停和选模，未用测试分数调参。

胜出方案：`reg:absoluteerror`、最大深度 8、学习率 0.04、869 棵树、上周客流作为 `base_margin`。55 个输入特征包含原有历史/邻居特征，以及固定参考的时间趋势、日历/节日指标、原始客流 lag 和周间变化。联邦节日特征只是可预先确定的日历信号，不等同于 CTA 的实际服务安排。

旧 XGBoost 的测试 WMAPE 为 9.16%，新模型为 8.10%，约降低 **11.6%**；测试 MAE 从 198.25 降至 175.26。不过旧模型的 RMSE 为 487.90，略好于新模型的 498.52。本次“最佳”的定义是预先选择的验证 WMAPE，不是每项指标都最优。旧权重按用户要求删除，历史指标保留在报告中。

推理时必须通过提供的代码设置 `base_margin`。仅加载 Booster、传入普通特征而遗漏上周客流初始预测，会得到错误结果。

**目录**

```text
models/
  xgboost/model.ubj        # 唯一保留的 XGBoost 权重
  lstm/model.pt            # 唯一保留的 LSTM 权重
  gcn_lstm/model.pt        # 唯一保留的 GCN–LSTM 权重
  best_models.json         # 模型登记及最终指标
  neural_migration_verification.json
data/cta_features.parquet  # 可从原始数据重建的特征缓存，不是模型
reports/
  best_model_metrics.csv
  best_models.png
  previous_metrics.csv    # 已删除旧模型的历史成绩
  cleanup_manifest.json   # 实际删除的文件、大小及 SHA-256
  xgboost_search/          # 五个候选的验证曲线和成绩；不保留落选权重
  gcn_lstm_experiment/     # 上一轮对照实验的历史报告
```

每个模型目录还保存配置/预处理元数据、验证和测试预测及指标。模型来源、原始训练代码/输入哈希和整理后的哈希均保留；改名和提取公共预处理后已重新验证预测一致性。原始 CSV、GTFS、KMZ 和无关项目文件不属于清理范围。

**复现与检查**

在项目目录运行，使用本次已有 Python 环境：XGBoost 3.2.0、PyTorch 2.10.0、NumPy 1.26.4、pandas 2.3.1、scikit-learn 1.5.1。

```bash
# 测试分开运行：当前环境中 torch 和 xgboost 的原生运行库在
# 同一测试进程退出时发生过冲突；分进程运行均正常退出。
python -m unittest -v test_cta_neural_forecast.py
python -m unittest -v test_cta_xgboost.py

# 重新加载保留模型并计算测试预测；不会重新选模
python cta_xgboost_train.py --evaluate-test --output models/xgboost
python cta_neural_forecast.py --evaluate-test --output models/lstm
python cta_neural_forecast.py --evaluate-test --output models/gcn_lstm

# 对共同目标重新计算指标和图表
python compare_models.py

# 如需重新训练，必须指定新的实验目录，避免覆盖保留的最佳模型
python cta_xgboost_train.py --output experiments/new_xgboost
python cta_neural_forecast.py --graph identity --output experiments/new_lstm
python cta_neural_forecast.py --graph physical --output experiments/new_gcn_lstm

# 单独重建特征并另存；已验证其与训练缓存数值一致
python cta_xgboost_features.py /tmp/cta_features_rebuilt.parquet
```

训练/评估命令会校验特征缓存或原始输入及相关源码哈希，避免误将改变后的输入用于旧 checkpoint。重建缓存用于新训练时，可运行 `cta_xgboost_train.py --rebuild-features --output experiments/new_xgboost`；重建由子进程完成，隔离原生运行库。

8 项行为测试覆盖输入不依赖目标/未来客流、训练期缩放、缺失标签掩码、季节基线、类别编码和推理的季节初始预测。公共特征提取代码重建出的 407,064 行、42 列基础特征与原缓存一致。比较脚本检查三个模型的目标键一致、实际值一致、预测非负且有限。

**仍需注意**

这是单随机种子、一个历史验证期上的比较。未证明统计显著性，也未部署。历史 GTFS/站点映射和目标日服务类型的可获得性仍需审计。不要把身份邻接 LSTM 的成绩当作“图结构带来提升”的证据。更强的结论需要新留出时间段、多随机种子和多个时间窗口验证。

英文说明见 [README.md](README.md)。原始客流 CSV 和 GTFS 数据不在仓库中，下载方式见英文说明的 Data 部分。

实现早停保存与自定义评估指标时参考了 [XGBoost 官方 API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) 和 [自定义指标文档](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)。
