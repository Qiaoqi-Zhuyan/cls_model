# 分类模型训练通用框架

## 整体文件结构

│  [main.py](http://main.py/)
│  model_val.py
│  [trainer.py](http://trainer.py/)
│
├─.idea
│  │  classification-model.iml
│  │  deployment.xml
│  │  misc.xml
│  │  modules.xml
│  │  vcs.xml
│  │  workspace.xml
│  │
│  └─inspectionProfiles
│          profiles_settings.xml
│          Project_Default.xml
│
├─config
│      model_config.py
│
├─datas
│      [build.py](http://build.py/)
│      imgs_augs.py
│      img_randaugs.py
│
├─labs
│  ├─logger
│  │      [logger.py](http://logger.py/)
│  │
│  └─weights
│          [weights.py](http://weights.py/)
│
├─model
│      ConvNext_official_impl.py
│      [models.py](http://models.py/)
│      poolformer_official_impl.py
│
└─tools
build_optimizer.py
get_mean_std.py
label_trans.py
lr_scheduler.py

---

TBD