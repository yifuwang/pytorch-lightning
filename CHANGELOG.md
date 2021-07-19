# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [1.4.0] - 2021-MM-DD

### Added

- Add `extract_batch_size` utility and corresponding tests to extract batch dimension from multiple batch types. ([#8357](https://github.com/PyTorchLightning/pytorch-lightning/pull/8357/))

- Add support for named parameter groups in `LearningRateMonitor` ([#7987](https://github.com/PyTorchLightning/pytorch-lightning/pull/7987))


- Add `dataclass` support for `pytorch_lightning.utilities.apply_to_collection` ([#7935](https://github.com/PyTorchLightning/pytorch-lightning/pull/7935))


- Added support to `LightningModule.to_torchscript` for saving to custom filesystems with fsspec ([#7617](https://github.com/PyTorchLightning/pytorch-lightning/pull/7617))


- Added `KubeflowEnvironment` for use with the `PyTorchJob` operator in Kubeflow


- Added LightningCLI support for config files on object stores ([#7521](https://github.com/PyTorchLightning/pytorch-lightning/pull/7521))


- Added `ModelPruning(prune_on_train_epoch_end=True|False)` to choose when to apply pruning ([#7704](https://github.com/PyTorchLightning/pytorch-lightning/pull/7704))


- Added support for checkpointing based on a provided time interval during training ([#7515](https://github.com/PyTorchLightning/pytorch-lightning/pull/7515))


- Progress tracking
  * Added dataclasses for progress tracking ([#6603](https://github.com/PyTorchLightning/pytorch-lightning/pull/6603), [#7574](https://github.com/PyTorchLightning/pytorch-lightning/pull/7574), [#8140](https://github.com/PyTorchLightning/pytorch-lightning/pull/8140), [#8362](https://github.com/PyTorchLightning/pytorch-lightning/pull/8362))
  * Add `{,load_}state_dict` to the progress tracking dataclasses ([#8140](https://github.com/PyTorchLightning/pytorch-lightning/pull/8140))
  * Connect the progress tracking dataclasses to the loops ([#8244](https://github.com/PyTorchLightning/pytorch-lightning/pull/8244), [#8362](https://github.com/PyTorchLightning/pytorch-lightning/pull/8362))
  * Do not reset the progress tracking dataclasses total counters ([#8475](https://github.com/PyTorchLightning/pytorch-lightning/pull/8475))


- Added support for passing a `LightningDataModule` positionally as the second argument to `trainer.{validate,test,predict}` ([#7431](https://github.com/PyTorchLightning/pytorch-lightning/pull/7431))


- Added argument `trainer.predict(ckpt_path)` ([#7430](https://github.com/PyTorchLightning/pytorch-lightning/pull/7430))


- Added `clip_grad_by_value` support for TPUs ([#7025](https://github.com/PyTorchLightning/pytorch-lightning/pull/7025))


- Added support for passing any class to `is_overridden` ([#7918](https://github.com/PyTorchLightning/pytorch-lightning/pull/7918))


- Added `sub_dir` parameter to `TensorBoardLogger` ([#6195](https://github.com/PyTorchLightning/pytorch-lightning/pull/6195))


- Added correct `dataloader_idx` to batch transfer hooks ([#6241](https://github.com/PyTorchLightning/pytorch-lightning/pull/6241))


- Added `include_none=bool` argument to `apply_to_collection` ([#7769](https://github.com/PyTorchLightning/pytorch-lightning/pull/7769))


- Added `apply_to_collections` to apply a function to two zipped collections ([#7769](https://github.com/PyTorchLightning/pytorch-lightning/pull/7769))


- Added `ddp_fully_sharded` support ([#7487](https://github.com/PyTorchLightning/pytorch-lightning/pull/7487))


- Added `should_rank_save_checkpoint` property to Training Plugins ([#7684](https://github.com/PyTorchLightning/pytorch-lightning/pull/7684))


- Added `log_grad_norm` hook to `LightningModule` to customize the logging of gradient norms ([#7873](https://github.com/PyTorchLightning/pytorch-lightning/pull/7873))


- Added `save_config_filename` init argument to `LightningCLI` to ease resolving name conflicts ([#7741](https://github.com/PyTorchLightning/pytorch-lightning/pull/7741))


- Added `save_config_overwrite` init argument to `LightningCLI` to ease overwriting existing config files ([#8059](https://github.com/PyTorchLightning/pytorch-lightning/pull/8059))


- Added reset dataloader hooks to Training Plugins and Accelerators ([#7861](https://github.com/PyTorchLightning/pytorch-lightning/pull/7861))


- Added trainer stage hooks for Training Plugins and Accelerators ([#7864](https://github.com/PyTorchLightning/pytorch-lightning/pull/7864))


- Added the `on_before_optimizer_step` hook ([#8048](https://github.com/PyTorchLightning/pytorch-lightning/pull/8048))


- Added IPU Accelerator ([#7867](https://github.com/PyTorchLightning/pytorch-lightning/pull/7867))


- Fault-tolerant training
    * Added `{,load_}state_dict` to `ResultCollection` ([#7948](https://github.com/PyTorchLightning/pytorch-lightning/pull/7948))
    * Added `{,load_}state_dict` to `Loops` ([#8197](https://github.com/PyTorchLightning/pytorch-lightning/pull/8197))
    * Set `Loop.restarting=False` at the end of the first iteration ([#8362](https://github.com/PyTorchLightning/pytorch-lightning/pull/8362))
    * Save the loops state with the checkpoint (opt-in) ([#8362](https://github.com/PyTorchLightning/pytorch-lightning/pull/8362))
    * Save a checkpoint to restore the state on exception (opt-in) ([#8362](https://github.com/PyTorchLightning/pytorch-lightning/pull/8362))
    * Added `state_dict` and `load_state_dict` utilities for `CombinedLoader` + utilities for dataloader ([#8364](https://github.com/PyTorchLightning/pytorch-lightning/pull/8364))


- Added `rank_zero_only` to `LightningModule.log` function ([#7966](https://github.com/PyTorchLightning/pytorch-lightning/pull/7966))


- Added `metric_attribute` to `LightningModule.log` function ([#7966](https://github.com/PyTorchLightning/pytorch-lightning/pull/7966))


- Added a warning if `Trainer(log_every_n_steps)` is a value too high for the training dataloader ([#7734](https://github.com/PyTorchLightning/pytorch-lightning/pull/7734))


- Added LightningCLI support for argument links applied on instantiation ([#7895](https://github.com/PyTorchLightning/pytorch-lightning/pull/7895))


- Added LightningCLI support for configurable callbacks that should always be present ([#7964](https://github.com/PyTorchLightning/pytorch-lightning/pull/7964))


- Added DeepSpeed Infinity Support, and updated to DeepSpeed 0.4.0 ([#7234](https://github.com/PyTorchLightning/pytorch-lightning/pull/7234))


- Added support for `torch.nn.UninitializedParameter` in `ModelSummary` ([#7642](https://github.com/PyTorchLightning/pytorch-lightning/pull/7642))


- Added support `LightningModule.save_hyperparameters` when `LightningModule` is a dataclass ([#7992](https://github.com/PyTorchLightning/pytorch-lightning/pull/7992))


- Added support for overriding `optimizer_zero_grad` and `optimizer_step` when using accumulate_grad_batches ([#7980](https://github.com/PyTorchLightning/pytorch-lightning/pull/7980))


- Added `logger` boolean flag to `save_hyperparameters` ([#7960](https://github.com/PyTorchLightning/pytorch-lightning/pull/7960))


- Add support for calling scripts using the module syntax (`python -m package.script`) ([#8073](https://github.com/PyTorchLightning/pytorch-lightning/pull/8073))


- Add support for optimizers and learning rate schedulers to `LightningCLI` ([#8093](https://github.com/PyTorchLightning/pytorch-lightning/pull/8093))


- Decoupled device parsing logic from Accelerator connector to Trainer ([#8180](https://github.com/PyTorchLightning/pytorch-lightning/pull/8180))


- Added XLA Profiler ([#8014](https://github.com/PyTorchLightning/pytorch-lightning/pull/8014))


- Added `PrecisionPlugin.{pre,post}_backward` ([#8328](https://github.com/PyTorchLightning/pytorch-lightning/pull/8328))


- Added `on_load_checkpoint` and `on_save_checkpoint` hooks to the `PrecisionPlugin` base class ([#7831](https://github.com/PyTorchLightning/pytorch-lightning/pull/7831))


- Added `max_depth` parameter in `ModelSummary` ([#8062](https://github.com/PyTorchLightning/pytorch-lightning/pull/8062))


- Added `XLAStatsMonitor` callback ([#8235](https://github.com/PyTorchLightning/pytorch-lightning/pull/8235))


- Added `restore` function and `restarting` attribute to base `Loop` ([#8247](https://github.com/PyTorchLightning/pytorch-lightning/pull/8247))


- Added `FastForwardSampler` and `CaptureIterableDataset` ([#8307](https://github.com/PyTorchLightning/pytorch-lightning/pull/8307))


- Added support for `save_hyperparameters` in `LightningDataModule` ([#3792](https://github.com/PyTorchLightning/pytorch-lightning/pull/3792))


- Added the `ModelCheckpoint(save_on_train_epoch_end)` to choose when to run the saving logic ([#8389](https://github.com/PyTorchLightning/pytorch-lightning/pull/8389))


- Added `LSFEnvironment` for distributed training with the LSF resource manager `jsrun` ([#5102](https://github.com/PyTorchLightning/pytorch-lightning/pull/5102))


- Added support for `accelerator='cpu'|'gpu'|'tpu'|'ipu'|'auto'` ([#7808](https://github.com/PyTorchLightning/pytorch-lightning/pull/7808))


- Added `tpu_spawn_debug` to plugin registry ([#7933](https://github.com/PyTorchLightning/pytorch-lightning/pull/7933))


- Enabled traditional/manual launching of DDP processes through `LOCAL_RANK` and `NODE_RANK` environment variable assignments ([#7480](https://github.com/PyTorchLightning/pytorch-lightning/pull/7480))


- Added experimental support for loop specialization ([#8226](https://github.com/PyTorchLightning/pytorch-lightning/pull/8226))


### Changed


- Changed the `Trainer`'s `checkpoint_callback` argument to allow only boolean values ([#7539](https://github.com/PyTorchLightning/pytorch-lightning/pull/7539))


- Log epoch metrics before the `on_evaluation_end` hook ([#7272](https://github.com/PyTorchLightning/pytorch-lightning/pull/7272))


- Explicitly disallow calling `self.log(on_epoch=False)` during epoch-only or single-call hooks ([#7874](https://github.com/PyTorchLightning/pytorch-lightning/pull/7874))


- Changed these `Trainer` methods to be protected: `call_setup_hook`, `call_configure_sharded_model`, `pre_dispatch`, `dispatch`, `post_dispatch`, `call_teardown_hook`, `run_train`, `run_sanity_check`, `run_evaluate`, `run_evaluation`, `run_predict`, `track_output_for_epoch_end`


- Changed `metrics_to_scalars` to work with any collection or value ([#7888](https://github.com/PyTorchLightning/pytorch-lightning/pull/7888))


- Changed `clip_grad_norm` to use `torch.nn.utils.clip_grad_norm_` ([#7025](https://github.com/PyTorchLightning/pytorch-lightning/pull/7025))


- Validation is now always run inside the training epoch scope ([#7357](https://github.com/PyTorchLightning/pytorch-lightning/pull/7357))


- `ModelCheckpoint` now runs at the end of the training epoch by default ([#8389](https://github.com/PyTorchLightning/pytorch-lightning/pull/8389))


- `EarlyStopping` now runs at the end of the training epoch by default ([#8286](https://github.com/PyTorchLightning/pytorch-lightning/pull/8286))


- Refactored Loops
    * Moved attributes `global_step`, `current_epoch`, `max/min_steps`, `max/min_epochs`, `batch_idx`, and `total_batch_idx` to TrainLoop ([#7437](https://github.com/PyTorchLightning/pytorch-lightning/pull/7437))
    * Refactored result handling in training loop ([#7506](https://github.com/PyTorchLightning/pytorch-lightning/pull/7506))
    * Moved attributes `hiddens` and `split_idx` to TrainLoop ([#7507](https://github.com/PyTorchLightning/pytorch-lightning/pull/7507))
    * Refactored the logic around manual and automatic optimization inside the optimizer loop ([#7526](https://github.com/PyTorchLightning/pytorch-lightning/pull/7526))
    * Simplified "should run validation" logic ([#7682](https://github.com/PyTorchLightning/pytorch-lightning/pull/7682))
    * Simplified logic for updating the learning rate for schedulers ([#7682](https://github.com/PyTorchLightning/pytorch-lightning/pull/7682))
    * Removed the `on_epoch` guard from the "should stop" validation check ([#7701](https://github.com/PyTorchLightning/pytorch-lightning/pull/7701))
    * Refactored internal loop interface; added new classes `FitLoop`, `TrainingEpochLoop`, `TrainingBatchLoop` ([#7871](https://github.com/PyTorchLightning/pytorch-lightning/pull/7871), [#8077](https://github.com/PyTorchLightning/pytorch-lightning/pull/8077))
    * Removed `pytorch_lightning/trainer/training_loop.py` ([#7985](https://github.com/PyTorchLightning/pytorch-lightning/pull/7985))
    * Refactored evaluation loop interface; added new classes `DataLoaderLoop`, `EvaluationLoop`, `EvaluationEpochLoop` ([#7990](https://github.com/PyTorchLightning/pytorch-lightning/pull/7990), [#8077](https://github.com/PyTorchLightning/pytorch-lightning/pull/8077))
    * Removed `pytorch_lightning/trainer/evaluation_loop.py` ([#8056](https://github.com/PyTorchLightning/pytorch-lightning/pull/8056))
    * Restricted public access to several internal functions ([#8024](https://github.com/PyTorchLightning/pytorch-lightning/pull/8024))
    * Refactored trainer `_run_*` functions and separate evaluation loops ([#8065](https://github.com/PyTorchLightning/pytorch-lightning/pull/8065))
    * Refactored prediction loop interface; added new classes `PredictionLoop`, `PredictionEpochLoop` ([#7700](https://github.com/PyTorchLightning/pytorch-lightning/pull/7700), [#8077](https://github.com/PyTorchLightning/pytorch-lightning/pull/8077))
    * Removed `pytorch_lightning/trainer/predict_loop.py` ([#8094](https://github.com/PyTorchLightning/pytorch-lightning/pull/8094))
    * Moved result teardown to the loops ([#8245](https://github.com/PyTorchLightning/pytorch-lightning/pull/8245))
    * Improve `Loop` API to better handle children `state_dict` and `progress` ([#8334](https://github.com/PyTorchLightning/pytorch-lightning/pull/8334))

- Refactored logging
    * Renamed and moved `core/step_result.py` to `trainer/connectors/logger_connector/result.py` ([#7736](https://github.com/PyTorchLightning/pytorch-lightning/pull/7736))
    * Dramatically simplify the `LoggerConnector` ([#7882](https://github.com/PyTorchLightning/pytorch-lightning/pull/7882))
    * `trainer.{logged,progress_bar,callback}_metrics` are now updated on-demand ([#7882](https://github.com/PyTorchLightning/pytorch-lightning/pull/7882))
    * Completely overhaul the `Result` object in favor of `ResultMetric` ([#7882](https://github.com/PyTorchLightning/pytorch-lightning/pull/7882))
    * Improve epoch-level reduction time and overall memory usage ([#7882](https://github.com/PyTorchLightning/pytorch-lightning/pull/7882))
    * Allow passing `self.log(batch_size=...)` ([#7891](https://github.com/PyTorchLightning/pytorch-lightning/pull/7891))
    * Each of the training loops now keeps its own results collection ([#7891](https://github.com/PyTorchLightning/pytorch-lightning/pull/7891))
    * Remove `EpochResultStore` and `HookResultStore` in favor of `ResultCollection` ([#7909](https://github.com/PyTorchLightning/pytorch-lightning/pull/7909))
    * Remove `MetricsHolder` ([#7909](https://github.com/PyTorchLightning/pytorch-lightning/pull/7909))


- Moved `ignore_scalar_return_in_dp` warning suppression to the DataParallelPlugin class ([#7421](https://github.com/PyTorchLightning/pytorch-lightning/pull/7421/))


- Changed the behaviour when logging evaluation step metrics to no longer append `/epoch_*` to the metric name ([#7351](https://github.com/PyTorchLightning/pytorch-lightning/pull/7351))


- Raise `ValueError` when a `None` value is `self.log`-ed ([#7771](https://github.com/PyTorchLightning/pytorch-lightning/pull/7771))


- Changed `resolve_training_type_plugins` to allow setting `num_nodes` and `sync_batchnorm` from `Trainer` setting ([#7026](https://github.com/PyTorchLightning/pytorch-lightning/pull/7026))


- Default `seed_everything(workers=True)` in the `LightningCLI` ([#7504](https://github.com/PyTorchLightning/pytorch-lightning/pull/7504))


- Changed `model.state_dict()` in `CheckpointConnector` to allow `training_type_plugin` to customize the model's `state_dict()` ([#7474](https://github.com/PyTorchLightning/pytorch-lightning/pull/7474))


- MLflowLogger now uses the env variable `MLFLOW_TRACKING_URI` as default tracking uri ([#7457](https://github.com/PyTorchLightning/pytorch-lightning/pull/7457))

- Changed `Trainer` arg and functionality from `reload_dataloaders_every_epoch` to `reload_dataloaders_every_n_epochs` ([#5043](https://github.com/PyTorchLightning/pytorch-lightning/pull/5043))

- Changed `WandbLogger(log_model={True/'all'})` to log models as artifacts ([#6231](https://github.com/PyTorchLightning/pytorch-lightning/pull/6231))


- MLFlowLogger now accepts `run_name` as an constructor argument ([#7622](https://github.com/PyTorchLightning/pytorch-lightning/issues/7622))


- Changed `teardown()` in `Accelerator` to allow `training_type_plugin` to customize `teardown` logic ([#7579](https://github.com/PyTorchLightning/pytorch-lightning/pull/7579))


- `Trainer.fit` now raises an error when using manual optimization with unsupported features such as `gradient_clip_val` or `accumulate_grad_batches` ([#7788](https://github.com/PyTorchLightning/pytorch-lightning/pull/7788))


- Accelerator hooks are called regardless if `LightningModule` overrides the same hooks ([#7826](https://github.com/PyTorchLightning/pytorch-lightning/pull/7826))


- Moved profilers to their own file ([#7822](https://github.com/PyTorchLightning/pytorch-lightning/pull/7822))


- The `on_after_backward` hook is now called on accumulating iterations. Use the `on_before_optimizer_step` hook to mimic the old behaviour ([#8328](https://github.com/PyTorchLightning/pytorch-lightning/pull/8328))


- The mixed precision loss is no longer unscaled before the `on_after_backward` hook. Use the `on_before_optimizer_step` hook to mimic the old behaviour  ([#8328](https://github.com/PyTorchLightning/pytorch-lightning/pull/8328))


- The `TrainingTypePlugin.{pre,post}_backward` hooks no longer take the `optimizer, opt_idx, should_accumulate` arguments ([#8328](https://github.com/PyTorchLightning/pytorch-lightning/pull/8328))


- The `PrecisionPlugin.backward` hooks no longer returns a value ([#8328](https://github.com/PyTorchLightning/pytorch-lightning/pull/8328))


- The `PrecisionPlugin.backward` hooks no longer takes a `should_accumulate` argument ([#8328](https://github.com/PyTorchLightning/pytorch-lightning/pull/8328))


- Added the `on_before_backward` hook ([#7865](https://github.com/PyTorchLightning/pytorch-lightning/pull/7865))


- `LightningCLI` now aborts with a clearer message if config already exists and disables save config during `fast_dev_run`([#7963](https://github.com/PyTorchLightning/pytorch-lightning/pull/7963))


- Save the `LightningCLI` config on `setup` and only on the main process ([#8017](https://github.com/PyTorchLightning/pytorch-lightning/pull/8017))


- Drop the `LightningCLI` `ArgumentParser` when pickling ([#8017](https://github.com/PyTorchLightning/pytorch-lightning/pull/8017))


- Skip `broadcast` if distributed not initialized for the spawn plugins ([#8017](https://github.com/PyTorchLightning/pytorch-lightning/pull/8017))


- `Trainer(resume_from_checkpoint=...)` now restores the model directly after `LightningModule.setup()`, which is before `LightningModule.configure_sharded_model()` ([#7652](https://github.com/PyTorchLightning/pytorch-lightning/pull/7652))


- Moved `torch.cuda.set_device()` to enable collective calls earlier in setup ([#8312](https://github.com/PyTorchLightning/pytorch-lightning/pull/8312))


- Use XLA utility API to move data to CPU (Single TPU core) ([#8078](https://github.com/PyTorchLightning/pytorch-lightning/pull/8078))


### Deprecated


- Deprecated `LightningModule.loaded_optimizer_states_dict` ([#8229](https://github.com/PyTorchLightning/pytorch-lightning/pull/8229))


- Standardized the dataloaders arguments of `trainer.{fit,valdiate,test,tune}` ([#7431](https://github.com/PyTorchLightning/pytorch-lightning/pull/7431))


- Deprecated `DataModule` properties: `has_prepared_data`, `has_setup_fit`, `has_setup_validate`, `has_setup_test`, `has_setup_predict`, `has_teardown_fit`, `has_teardown_validate`, `has_teardown_test`, `has_teardown_predict` ([#7657](https://github.com/PyTorchLightning/pytorch-lightning/pull/7657/))


- Deprecated `TrainerModelHooksMixin` in favor of `pytorch_lightning.utilities.signature_utils` ([#7422](https://github.com/PyTorchLightning/pytorch-lightning/pull/7422))


- Deprecated `num_nodes` and `sync_batchnorm` arguments in `DDPPlugin` and `DDPSpawnPlugin` ([#7026](https://github.com/PyTorchLightning/pytorch-lightning/pull/7026))


- Deprecated `self.log(sync_dist_op)` in favor of `self.log(reduce_fx)`. ([#7891](https://github.com/PyTorchLightning/pytorch-lightning/pull/7891))


- Deprecated `is_overridden(model=...)` in favor of `is_overridden(instance=...)` ([#7918](https://github.com/PyTorchLightning/pytorch-lightning/pull/7918))


- Deprecated automatically detaching returned extras with grads ([#7994](https://github.com/PyTorchLightning/pytorch-lightning/pull/7994))


- Deprecated default value of `monitor` argument in EarlyStopping callback to enforce `monitor` as a required argument ([#7907](https://github.com/PyTorchLightning/pytorch-lightning/pull/7907))


- Deprecated importing `rank_zero_{warn,deprecation}` directly from `pytorch_lightning.utilities.distributed` ([#8085](https://github.com/PyTorchLightning/pytorch-lightning/pull/8085))


- Deprecated the use of `CheckpointConnector.hpc_load()` in favor of `CheckpointConnector.restore()` ([#7652](https://github.com/PyTorchLightning/pytorch-lightning/pull/7652))


- Deprecated `ModelCheckpoint(every_n_val_epochs)` in favor of `ModelCheckpoint(every_n_epochs)` ([#8383](https://github.com/PyTorchLightning/pytorch-lightning/pull/8383))


- Deprecated `DDPPlugin.task_idx` in favor of `DDPPlugin.local_rank` ([#8203](https://github.com/PyTorchLightning/pytorch-lightning/pull/8203))


- Deprecated the `Trainer.train_loop` property in favor of `Trainer.fit_loop` ([#8025](https://github.com/PyTorchLightning/pytorch-lightning/pull/8025))


- Deprecated the `Trainer.disable_validation` property in favor of `not Trainer.enable_validation` ([#8291](https://github.com/PyTorchLightning/pytorch-lightning/pull/8291))


- Deprecated `mode` parameter in `ModelSummary` in favor of `max_depth` ([#8062](https://github.com/PyTorchLightning/pytorch-lightning/pull/8062))


- Deprecated `reload_dataloaders_every_epoch` argument of `Trainer` in favor of `reload_dataloaders_every_n_epochs` ([#5043](https://github.com/PyTorchLightning/pytorch-lightning/pull/5043))


### Removed


- Dropped official support/testing for PyTorch <1.6 ([#8288](https://github.com/PyTorchLightning/pytorch-lightning/pull/8288))


- Removed `ProfilerConnector` ([#7654](https://github.com/PyTorchLightning/pytorch-lightning/pull/7654))


- Prune deprecated classif. metrics from `pytorch_lightning.metrics.functional.classification` ([#7499](https://github.com/PyTorchLightning/pytorch-lightning/pull/7499))


- Removed deprecated data parallel classes `LightningDataParallel` and `LightningDistributedDataParallel` from `pytorch_lightning.overrides.data_parallel` ([#7510](https://github.com/PyTorchLightning/pytorch-lightning/pull/7510))


- Removed deprecated trainer attributes - `get_model` and `accelerator_backend` ([#7502](https://github.com/PyTorchLightning/pytorch-lightning/pull/7502))


- Removed support for automatically monitoring the `val_loss` key with `ModelCheckpoint`. Pass your `monitor` of choice to the `ModelCheckpoint` instance instead ([#8293](https://github.com/PyTorchLightning/pytorch-lightning/pull/8293))


- Removed support for `self.log(tbptt_reduce_fx)` and `self.log(tbptt_pad_token)`. Please, open a discussion explaining your use-case if you relied on these. ([#7644](https://github.com/PyTorchLightning/pytorch-lightning/pull/7644))


- Removed deprecated utils modules `model_utils`, `warning_utils`, `xla_device_utils` and partially `argparse_utils` ([#7503](https://github.com/PyTorchLightning/pytorch-lightning/pull/7503))


- Removed `RPCPlugin` and `RPCSequentialPlugin`. If you were successfully using these plugins, please open a GitHub discussion about your use case ([#8101](https://github.com/PyTorchLightning/pytorch-lightning/pull/8101))


- Removed deprecated trainer attributes - `on_cpu`, `on_tpu`, `use_tpu`, `on_gpu`, `use_dp`, `use_ddp`, `use_ddp2`, `use_horovod`, `use_single_gpu` ([#7501](https://github.com/PyTorchLightning/pytorch-lightning/pull/7501))


- Removed deprecated `optimizer` argument in `LightningModule.manual_backward()`; Toggling optimizers in manual optimization should be done using `LightningModule.{un}toggle_optimizer()` ([#8287](https://github.com/PyTorchLightning/pytorch-lightning/pull/8287))


- Removed DeepSpeed FP16 Exception as FP32 is now supported ([#8462](https://github.com/PyTorchLightning/pytorch-lightning/pull/8462))


### Fixed

- Fixed the `GPUStatsMonitor` callbacks to use the correct GPU IDs if `CUDA_VISIBLE_DEVICES` set ([#8260](https://github.com/PyTorchLightning/pytorch-lightning/pull/8260))

- Fixed `lr_scheduler` checkpointed state by calling `update_lr_schedulers` before saving checkpoints ([#7877](https://github.com/PyTorchLightning/pytorch-lightning/pull/7877))


- Fixed ambiguous warning when both overfit and train dataloader shuffling are enabled ([#7685](https://github.com/PyTorchLightning/pytorch-lightning/pull/7685))


- Fixed dev debugger memory growing due to tracking events even when disabled ([#7875](https://github.com/PyTorchLightning/pytorch-lightning/pull/7875))


- Fixed `None` loss keys getting added in `training_epoch_end` when using manual optimization and not returning a loss ([#7772](https://github.com/PyTorchLightning/pytorch-lightning/pull/7772))


- Fixed a bug where `precision=64` with `accelerator='ddp_spawn'` would throw a pickle error ([#6924](https://github.com/PyTorchLightning/pytorch-lightning/pull/6924))


- Do not override the existing `epoch` value in `logged_metrics` when already logged by the user ([#7982](https://github.com/PyTorchLightning/pytorch-lightning/issues/7982))


- Support manual optimization with DeepSpeed ([#7970](https://github.com/PyTorchLightning/pytorch-lightning/pull/7970))


- Fixed `dataloader_idx` argument value when predicting with only one `DataLoader` ([#7941](https://github.com/PyTorchLightning/pytorch-lightning/pull/7941))


- Pass the `stage` argument of `Callback.{setup,teardown}` as a keyword ([#7973](https://github.com/PyTorchLightning/pytorch-lightning/pull/7973))


- Fixed metrics generated during `validation sanity checking` are cleaned on end ([#8171](https://github.com/PyTorchLightning/pytorch-lightning/pull/8171))


- Fixed `log_gpu_memory` metrics not being added to `logging` when nothing else is logged ([#8174](https://github.com/PyTorchLightning/pytorch-lightning/pull/8174))


- Fixed a bug where calling `log` with a `Metric` instance would raise an error if it was a nested attribute of the model ([#8181](https://github.com/PyTorchLightning/pytorch-lightning/pull/8181))


- Fixed a bug where using `precision=64` would cause buffers with complex dtype to be cast to real ([#8208](https://github.com/PyTorchLightning/pytorch-lightning/pull/8208))


- Fixed `is_overridden` returning true for wrapped functions with no changes ([#8296](https://github.com/PyTorchLightning/pytorch-lightning/pull/8296))


- Fixed a bug where `truncated_bptt_steps` would throw an AttributeError when the target RNN has multiple hidden states ([#8145](https://github.com/PyTorchLightning/pytorch-lightning/pull/8145))


- Fixed `self.optimizers()` not returning a single optimizer if it had been wrapped ([#8326](https://github.com/PyTorchLightning/pytorch-lightning/pull/8326))


- Fixed the `on_after_backward` hook not getting called when using manual optimization and no plugins ([#8328](https://github.com/PyTorchLightning/pytorch-lightning/pull/8328))


- Fixed the `LightningModule.backward` hook only getting called with the `apex` plugin when using manual optimization ([#8328](https://github.com/PyTorchLightning/pytorch-lightning/pull/8328))


- Fixed moving batch to device before sending it to the `on_*_batch_start`/`on_*_batch_end` callbacks and model hooks ([#7378](https://github.com/PyTorchLightning/pytorch-lightning/pull/7378))


- Fixed passing a custom `DDPPlugin` when choosing `accelerator="ddp_cpu"` for the accelerator ([#6208](https://github.com/PyTorchLightning/pytorch-lightning/pull/6208))


- Fixed missing call to `LightningModule.untoggle_optimizer` in training loop when running gradient accumulation with multiple optimizers ([#8284](https://github.com/PyTorchLightning/pytorch-lightning/pull/8284))

- Fixed hash of LightningEnum to work with value instead of name([#8421](https://github.com/PyTorchLightning/pytorch-lightning/pull/8421)).

- Fixed `move_data_to_device` to return the batch if the object `to` function didn't return `self` ([#8433](https://github.com/PyTorchLightning/pytorch-lightning/pull/8433))


- Fixed progress bar updates for Pod Training ([#8258](https://github.com/PyTorchLightning/pytorch-lightning/pull/8258))


- Fixed clearing dataloader references before attaching new dataloaders in consecutive `Trainer.{fit,validate,test,predict}´ runs ([#8442](https://github.com/PyTorchLightning/pytorch-lightning/pull/8442))


## [1.3.8] - 2021-07-01

### Fixed

- Fixed a sync deadlock when checkpointing a `LightningModule` that uses a torchmetrics 0.4 `Metric` ([#8218](https://github.com/PyTorchLightning/pytorch-lightning/pull/8218))
- Fixed compatibility TorchMetrics v0.4 ([#8206](https://github.com/PyTorchLightning/pytorch-lightning/pull/8206))
- Added torchelastic check when sanitizing GPUs ([#8095](https://github.com/PyTorchLightning/pytorch-lightning/pull/8095))
- Fixed a DDP info message that was never shown ([#8111](https://github.com/PyTorchLightning/pytorch-lightning/pull/8111))
- Fixed metrics deprecation message at module import level ([#8163](https://github.com/PyTorchLightning/pytorch-lightning/pull/8163))
- Fixed a bug where an infinite recursion would be triggered when using the `BaseFinetuning` callback on a model that contains a `ModuleDict` ([#8170](https://github.com/PyTorchLightning/pytorch-lightning/pull/8170))
- Added a mechanism to detect `deadlock` for `DDP` when only 1 process trigger an `Exception`. The mechanism will `kill the processes` when it happens ([#8167](https://github.com/PyTorchLightning/pytorch-lightning/pull/8167))
- Fixed NCCL error when selecting non-consecutive device ids ([#8165](https://github.com/PyTorchLightning/pytorch-lightning/pull/8165))
- Fixed SWA to also work with `IterableDataset` ([#8172](https://github.com/PyTorchLightning/pytorch-lightning/pull/8172))


## [1.3.7] - 2021-06-22

### Fixed

- Fixed a bug where skipping an optimizer while using amp causes amp to trigger an assertion error ([#7975](https://github.com/PyTorchLightning/pytorch-lightning/pull/7975))
- Fixed deprecation messages not showing due to incorrect stacklevel ([#8002](https://github.com/PyTorchLightning/pytorch-lightning/pull/8002), [#8005](https://github.com/PyTorchLightning/pytorch-lightning/pull/8005))
- Fixed setting a `DistributedSampler` when using a distributed plugin in a custom accelerator ([#7814](https://github.com/PyTorchLightning/pytorch-lightning/pull/7814))
- Improved `PyTorchProfiler` chrome traces names ([#8009](https://github.com/PyTorchLightning/pytorch-lightning/pull/8009))
- Fixed moving the best score to device in `EarlyStopping` callback for TPU devices ([#7959](https://github.com/PyTorchLightning/pytorch-lightning/pull/7959))
- Fixes access to `callback_metrics` in ddp_spawn ([#7916](https://github.com/PyTorchLightning/pytorch-lightning/pull/7916))


## [1.3.6] - 2021-06-15

### Fixed

- Fixed logs overwriting issue for remote filesystems ([#7889](https://github.com/PyTorchLightning/pytorch-lightning/pull/7889))
- Fixed `DataModule.prepare_data` could only be called on the global rank 0 process ([#7945](https://github.com/PyTorchLightning/pytorch-lightning/pull/7945))
- Fixed setting `worker_init_fn` to seed dataloaders correctly when using DDP ([#7942](https://github.com/PyTorchLightning/pytorch-lightning/pull/7942))
- Fixed `BaseFinetuning` callback to properly handle parent modules w/ parameters ([#7931](https://github.com/PyTorchLightning/pytorch-lightning/pull/7931))


## [1.3.5] - 2021-06-08

### Added

- Added warning to Training Step output ([#7779](https://github.com/PyTorchLightning/pytorch-lightning/pull/7779))

### Fixed

- Fixed `LearningRateMonitor` and `BackboneFinetuning` ([#7835](https://github.com/PyTorchLightning/pytorch-lightning/pull/7835))
- Minor improvements to `apply_to_collection` and type signature of `log_dict` ([#7851](https://github.com/PyTorchLightning/pytorch-lightning/pull/7851))
- Fixed docker versions ([#7834](https://github.com/PyTorchLightning/pytorch-lightning/pull/7834))
- Fixed sharded training check for fp16 precision ([#7825](https://github.com/PyTorchLightning/pytorch-lightning/pull/7825))
- Fixed support for torch Module type hints in LightningCLI ([#7807](https://github.com/PyTorchLightning/pytorch-lightning/pull/7807))

### Changed

- Move `training_output` validation to after `train_step_end` ([#7868](https://github.com/PyTorchLightning/pytorch-lightning/pull/7868))


## [1.3.4] - 2021-06-01

### Fixed

- Fixed info message when max training time reached ([#7780](https://github.com/PyTorchLightning/pytorch-lightning/pull/7780))
- Fixed missing `__len__` method to `IndexBatchSamplerWrapper` ([#7681](https://github.com/PyTorchLightning/pytorch-lightning/pull/7681))


## [1.3.3] - 2021-05-27

### Changed

- Changed calling of `untoggle_optimizer(opt_idx)` out of the closure function ([#7563](https://github.com/PyTorchLightning/pytorch-lightning/pull/7563))

### Fixed

- Fixed `ProgressBar` pickling after calling `trainer.predict` ([#7608](https://github.com/PyTorchLightning/pytorch-lightning/pull/7608))
- Fixed broadcasting in multi-node, multi-gpu DDP using torch 1.7 ([#7592](https://github.com/PyTorchLightning/pytorch-lightning/pull/7592))
- Fixed dataloaders are not reset when tuning the model ([#7566](https://github.com/PyTorchLightning/pytorch-lightning/pull/7566))
- Fixed print errors in `ProgressBar` when `trainer.fit` is not called ([#7674](https://github.com/PyTorchLightning/pytorch-lightning/pull/7674))
- Fixed global step update when the epoch is skipped ([#7677](https://github.com/PyTorchLightning/pytorch-lightning/pull/7677))
- Fixed training loop total batch counter when accumulate grad batches was enabled ([#7692](https://github.com/PyTorchLightning/pytorch-lightning/pull/7692))


## [1.3.2] - 2021-05-18

### Changed

- `DataModule`s now avoid duplicate `{setup,teardown,prepare_data}` calls for the same stage ([#7238](https://github.com/PyTorchLightning/pytorch-lightning/pull/7238))

### Fixed

- Fixed parsing of multiple training dataloaders ([#7433](https://github.com/PyTorchLightning/pytorch-lightning/pull/7433))
- Fixed recursive passing of `wrong_type` keyword argument in `pytorch_lightning.utilities.apply_to_collection` ([#7433](https://github.com/PyTorchLightning/pytorch-lightning/pull/7433))
- Fixed setting correct `DistribType` for `ddp_cpu` (spawn) backend ([#7492](https://github.com/PyTorchLightning/pytorch-lightning/pull/7492))
- Fixed incorrect number of calls to LR scheduler when `check_val_every_n_epoch > 1` ([#7032](https://github.com/PyTorchLightning/pytorch-lightning/pull/7032))


## [1.3.1] - 2021-05-11

### Fixed

- Fixed DeepSpeed with IterableDatasets ([#7362](https://github.com/PyTorchLightning/pytorch-lightning/pull/7362))
- Fixed `Trainer.current_epoch` not getting restored after tuning ([#7434](https://github.com/PyTorchLightning/pytorch-lightning/pull/7434))
- Fixed local rank displayed in console log ([#7395](https://github.com/PyTorchLightning/pytorch-lightning/pull/7395))


## [1.3.0] - 2021-05-06

### Added

- Added support for the `EarlyStopping` callback to run at the end of the training epoch ([#6944](https://github.com/PyTorchLightning/pytorch-lightning/pull/6944))
- Added synchronization points before and after `setup` hooks are run ([#7202](https://github.com/PyTorchLightning/pytorch-lightning/pull/7202))
- Added a `teardown` hook to `ClusterEnvironment` ([#6942](https://github.com/PyTorchLightning/pytorch-lightning/pull/6942))
- Added utils for metrics to scalar conversions ([#7180](https://github.com/PyTorchLightning/pytorch-lightning/pull/7180))
- Added utils for NaN/Inf detection for gradients and parameters ([#6834](https://github.com/PyTorchLightning/pytorch-lightning/pull/6834))
- Added more explicit exception message when trying to execute `trainer.test()` or `trainer.validate()` with `fast_dev_run=True` ([#6667](https://github.com/PyTorchLightning/pytorch-lightning/pull/6667))
- Added `LightningCLI` class to provide simple reproducibility with minimum boilerplate training CLI (
    [#4492](https://github.com/PyTorchLightning/pytorch-lightning/pull/4492),
    [#6862](https://github.com/PyTorchLightning/pytorch-lightning/pull/6862),
    [#7156](https://github.com/PyTorchLightning/pytorch-lightning/pull/7156),
    [#7299](https://github.com/PyTorchLightning/pytorch-lightning/pull/7299))
- Added `gradient_clip_algorithm` argument to Trainer for gradient clipping by value ([#6123](https://github.com/PyTorchLightning/pytorch-lightning/pull/6123)).
- Added a way to print to terminal without breaking up the progress bar ([#5470](https://github.com/PyTorchLightning/pytorch-lightning/pull/5470))
- Added support to checkpoint after training steps in `ModelCheckpoint` callback ([#6146](https://github.com/PyTorchLightning/pytorch-lightning/pull/6146))
- Added `TrainerStatus.{INITIALIZING,RUNNING,FINISHED,INTERRUPTED}` ([#7173](https://github.com/PyTorchLightning/pytorch-lightning/pull/7173))
- Added `Trainer.validate()` method to perform one evaluation epoch over the validation set ([#4948](https://github.com/PyTorchLightning/pytorch-lightning/pull/4948))
- Added `LightningEnvironment` for Lightning-specific DDP ([#5915](https://github.com/PyTorchLightning/pytorch-lightning/pull/5915))
- Added `teardown()` hook to LightningDataModule ([#4673](https://github.com/PyTorchLightning/pytorch-lightning/pull/4673))
- Added `auto_insert_metric_name` parameter to `ModelCheckpoint` ([#6277](https://github.com/PyTorchLightning/pytorch-lightning/pull/6277))
- Added arg to `self.log` that enables users to give custom names when dealing with multiple dataloaders ([#6274](https://github.com/PyTorchLightning/pytorch-lightning/pull/6274))
- Added `teardown` method to `BaseProfiler` to enable subclasses defining post-profiling steps outside of `__del__` ([#6370](https://github.com/PyTorchLightning/pytorch-lightning/pull/6370))
- Added `setup` method to `BaseProfiler` to enable subclasses defining pre-profiling steps for every process ([#6633](https://github.com/PyTorchLightning/pytorch-lightning/pull/6633))
- Added no return warning to predict ([#6139](https://github.com/PyTorchLightning/pytorch-lightning/pull/6139))
- Added `Trainer.predict` config validation ([#6543](https://github.com/PyTorchLightning/pytorch-lightning/pull/6543))
- Added `AbstractProfiler` interface ([#6621](https://github.com/PyTorchLightning/pytorch-lightning/pull/6621))
- Added support for including module names for forward in the autograd trace of `PyTorchProfiler` ([#6349](https://github.com/PyTorchLightning/pytorch-lightning/pull/6349))
- Added support for the PyTorch 1.8.1 autograd profiler ([#6618](https://github.com/PyTorchLightning/pytorch-lightning/pull/6618))
- Added `outputs` parameter to callback's `on_validation_epoch_end` & `on_test_epoch_end` hooks ([#6120](https://github.com/PyTorchLightning/pytorch-lightning/pull/6120))
- Added `configure_sharded_model` hook ([#6679](https://github.com/PyTorchLightning/pytorch-lightning/pull/6679))
- Added support for `precision=64`, enabling training with double precision ([#6595](https://github.com/PyTorchLightning/pytorch-lightning/pull/6595))
- Added support for DDP communication hooks ([#6736](https://github.com/PyTorchLightning/pytorch-lightning/pull/6736))
- Added `artifact_location` argument to `MLFlowLogger` which will be passed to the `MlflowClient.create_experiment` call ([#6677](https://github.com/PyTorchLightning/pytorch-lightning/pull/6677))
- Added `model` parameter to precision plugins' `clip_gradients` signature (
    [#6764](https://github.com/PyTorchLightning/pytorch-lightning/pull/6764),
    [#7231](https://github.com/PyTorchLightning/pytorch-lightning/pull/7231))
- Added `is_last_batch` attribute to `Trainer` ([#6825](https://github.com/PyTorchLightning/pytorch-lightning/pull/6825))
- Added `LightningModule.lr_schedulers()` for manual optimization  ([#6567](https://github.com/PyTorchLightning/pytorch-lightning/pull/6567))
- Added `MpModelWrapper` in TPU Spawn ([#7045](https://github.com/PyTorchLightning/pytorch-lightning/pull/7045))
- Added `max_time` Trainer argument to limit training time ([#6823](https://github.com/PyTorchLightning/pytorch-lightning/pull/6823))
- Added `on_predict_{batch,epoch}_{start,end}` hooks ([#7141](https://github.com/PyTorchLightning/pytorch-lightning/pull/7141))
- Added new `EarlyStopping` parameters `stopping_threshold` and `divergence_threshold` ([#6868](https://github.com/PyTorchLightning/pytorch-lightning/pull/6868))
- Added `debug` flag to TPU Training Plugins (PT_XLA_DEBUG) ([#7219](https://github.com/PyTorchLightning/pytorch-lightning/pull/7219))
- Added new `UnrepeatedDistributedSampler` and `IndexBatchSamplerWrapper` for tracking distributed predictions ([#7215](https://github.com/PyTorchLightning/pytorch-lightning/pull/7215))
- Added `trainer.predict(return_predictions=None|False|True)` ([#7215](https://github.com/PyTorchLightning/pytorch-lightning/pull/7215))
- Added `BasePredictionWriter` callback to implement prediction saving ([#7127](https://github.com/PyTorchLightning/pytorch-lightning/pull/7127))
- Added `trainer.tune(scale_batch_size_kwargs, lr_find_kwargs)` arguments to configure the tuning algorithms ([#7258](https://github.com/PyTorchLightning/pytorch-lightning/pull/7258))
- Added `tpu_distributed` check for TPU Spawn barrier ([#7241](https://github.com/PyTorchLightning/pytorch-lightning/pull/7241))
- Added device updates to TPU Spawn for Pod training ([#7243](https://github.com/PyTorchLightning/pytorch-lightning/pull/7243))
- Added warning when missing `Callback` and using `resume_from_checkpoint` ([#7254](https://github.com/PyTorchLightning/pytorch-lightning/pull/7254))
- DeepSpeed single file saving ([#6900](https://github.com/PyTorchLightning/pytorch-lightning/pull/6900))
- Added Training type Plugins Registry (
    [#6982](https://github.com/PyTorchLightning/pytorch-lightning/pull/6982),
    [#7063](https://github.com/PyTorchLightning/pytorch-lightning/pull/7063),
    [#7214](https://github.com/PyTorchLightning/pytorch-lightning/pull/7214),
    [#7224](https://github.com/PyTorchLightning/pytorch-lightning/pull/7224)
)
- Add `ignore` param to `save_hyperparameters` ([#6056](https://github.com/PyTorchLightning/pytorch-lightning/pull/6056))

### Changed

- Changed `LightningModule.truncated_bptt_steps` to be property ([#7323](https://github.com/PyTorchLightning/pytorch-lightning/pull/7323))
- Changed `EarlyStopping` callback from by default running `EarlyStopping.on_validation_end` if only training is run. Set `check_on_train_epoch_end` to run the callback at the end of the train epoch instead of at the end of the validation epoch ([#7069](https://github.com/PyTorchLightning/pytorch-lightning/pull/7069))
- Renamed `pytorch_lightning.callbacks.swa` to `pytorch_lightning.callbacks.stochastic_weight_avg` ([#6259](https://github.com/PyTorchLightning/pytorch-lightning/pull/6259))
- Refactor `RunningStage` and `TrainerState` usage (
    [#4945](https://github.com/PyTorchLightning/pytorch-lightning/pull/4945),
    [#7173](https://github.com/PyTorchLightning/pytorch-lightning/pull/7173))
    * Added `RunningStage.SANITY_CHECKING`
    * Added `TrainerFn.{FITTING,VALIDATING,TESTING,PREDICTING,TUNING}`
    * Changed `trainer.evaluating` to return `True` if validating or testing
- Changed `setup()` and `teardown()` stage argument to take any of `{fit,validate,test,predict}` ([#6386](https://github.com/PyTorchLightning/pytorch-lightning/pull/6386))
- Changed profilers to save separate report files per state and rank ([#6621](https://github.com/PyTorchLightning/pytorch-lightning/pull/6621))
- The trainer no longer tries to save a checkpoint on exception or run callback's `on_train_end` functions ([#6864](https://github.com/PyTorchLightning/pytorch-lightning/pull/6864))
- Changed `PyTorchProfiler` to use `torch.autograd.profiler.record_function` to record functions ([#6349](https://github.com/PyTorchLightning/pytorch-lightning/pull/6349))
- Disabled `lr_scheduler.step()` in manual optimization  ([#6825](https://github.com/PyTorchLightning/pytorch-lightning/pull/6825))
- Changed warnings and recommendations for dataloaders in `ddp_spawn` ([#6762](https://github.com/PyTorchLightning/pytorch-lightning/pull/6762))
- `pl.seed_everything` will now also set the seed on the `DistributedSampler` ([#7024](https://github.com/PyTorchLightning/pytorch-lightning/pull/7024))
- Changed default setting for communication of multi-node training using `DDPShardedPlugin` ([#6937](https://github.com/PyTorchLightning/pytorch-lightning/pull/6937))
- `trainer.tune()` now returns the tuning result ([#7258](https://github.com/PyTorchLightning/pytorch-lightning/pull/7258))
- `LightningModule.from_datasets()` now accepts `IterableDataset` instances as training datasets. ([#7503](https://github.com/PyTorchLightning/pytorch-lightning/pull/7503))
- Changed `resume_from_checkpoint` warning to an error when the checkpoint file does not exist ([#7075](https://github.com/PyTorchLightning/pytorch-lightning/pull/7075))
- Automatically set `sync_batchnorm` for `training_type_plugin` ([#6536](https://github.com/PyTorchLightning/pytorch-lightning/pull/6536))
- Allowed training type plugin to delay optimizer creation ([#6331](https://github.com/PyTorchLightning/pytorch-lightning/pull/6331))
- Removed ModelSummary validation from train loop on_trainer_init ([#6610](https://github.com/PyTorchLightning/pytorch-lightning/pull/6610))
- Moved `save_function` to accelerator ([#6689](https://github.com/PyTorchLightning/pytorch-lightning/pull/6689))
- Updated DeepSpeed ZeRO ([#6546](https://github.com/PyTorchLightning/pytorch-lightning/pull/6546),
    [#6752](https://github.com/PyTorchLightning/pytorch-lightning/pull/6752),
    [#6142](https://github.com/PyTorchLightning/pytorch-lightning/pull/6142),
    [#6321](https://github.com/PyTorchLightning/pytorch-lightning/pull/6321))
- Improved verbose logging for `EarlyStopping` callback ([#6811](https://github.com/PyTorchLightning/pytorch-lightning/pull/6811))
- Run ddp_spawn dataloader checks on Windows ([#6930](https://github.com/PyTorchLightning/pytorch-lightning/pull/6930))
- Updated mlflow with using `resolve_tags` ([#6746](https://github.com/PyTorchLightning/pytorch-lightning/pull/6746))
- Moved `save_hyperparameters` to its own function ([#7119](https://github.com/PyTorchLightning/pytorch-lightning/pull/7119))
- Replaced `_DataModuleWrapper` with `__new__` ([#7289](https://github.com/PyTorchLightning/pytorch-lightning/pull/7289))
- Reset `current_fx` properties on lightning module in teardown ([#7247](https://github.com/PyTorchLightning/pytorch-lightning/pull/7247))
- Auto-set `DataLoader.worker_init_fn` with `seed_everything` ([#6960](https://github.com/PyTorchLightning/pytorch-lightning/pull/6960))
- Remove `model.trainer` call inside of dataloading mixin ([#7317](https://github.com/PyTorchLightning/pytorch-lightning/pull/7317))
- Split profilers module ([#6261](https://github.com/PyTorchLightning/pytorch-lightning/pull/6261))
- Ensure accelerator is valid if running interactively ([#5970](https://github.com/PyTorchLightning/pytorch-lightning/pull/5970))
- Disabled batch transfer in DP mode ([#6098](https://github.com/PyTorchLightning/pytorch-lightning/pull/6098))

### Deprecated

- Deprecated `outputs` in both `LightningModule.on_train_epoch_end` and `Callback.on_train_epoch_end` hooks ([#7339](https://github.com/PyTorchLightning/pytorch-lightning/pull/7339))
- Deprecated `Trainer.truncated_bptt_steps` in favor of `LightningModule.truncated_bptt_steps` ([#7323](https://github.com/PyTorchLightning/pytorch-lightning/pull/7323))
- Deprecated `outputs` in both `LightningModule.on_train_epoch_end` and `Callback.on_train_epoch_end` hooks ([#7339](https://github.com/PyTorchLightning/pytorch-lightning/pull/7339))
- Deprecated `LightningModule.grad_norm` in favor of `pytorch_lightning.utilities.grads.grad_norm` ([#7292](https://github.com/PyTorchLightning/pytorch-lightning/pull/7292))
- Deprecated the `save_function` property from the `ModelCheckpoint` callback ([#7201](https://github.com/PyTorchLightning/pytorch-lightning/pull/7201))
- Deprecated `LightningModule.write_predictions` and `LightningModule.write_predictions_dict` ([#7066](https://github.com/PyTorchLightning/pytorch-lightning/pull/7066))
- Deprecated `TrainerLoggingMixin` in favor of a separate utilities module for metric handling ([#7180](https://github.com/PyTorchLightning/pytorch-lightning/pull/7180))
- Deprecated `TrainerTrainingTricksMixin` in favor of a separate utilities module for NaN/Inf detection for gradients and parameters ([#6834](https://github.com/PyTorchLightning/pytorch-lightning/pull/6834))
- `period` has been deprecated in favor of `every_n_val_epochs` in the `ModelCheckpoint` callback ([#6146](https://github.com/PyTorchLightning/pytorch-lightning/pull/6146))
- Deprecated `trainer.running_sanity_check` in favor of `trainer.sanity_checking` ([#4945](https://github.com/PyTorchLightning/pytorch-lightning/pull/4945))
- Deprecated `Profiler(output_filename)` in favor of `dirpath` and `filename` ([#6621](https://github.com/PyTorchLightning/pytorch-lightning/pull/6621))
- Deprecated `PytorchProfiler(profiled_functions)` in favor of `record_functions` ([#6349](https://github.com/PyTorchLightning/pytorch-lightning/pull/6349))
- Deprecated `@auto_move_data` in favor of `trainer.predict` ([#6993](https://github.com/PyTorchLightning/pytorch-lightning/pull/6993))
- Deprecated `Callback.on_load_checkpoint(checkpoint)` in favor of `Callback.on_load_checkpoint(trainer, pl_module, checkpoint)` ([#7253](https://github.com/PyTorchLightning/pytorch-lightning/pull/7253))
- Deprecated metrics in favor of `torchmetrics` (
    [#6505](https://github.com/PyTorchLightning/pytorch-lightning/pull/6505),
    [#6530](https://github.com/PyTorchLightning/pytorch-lightning/pull/6530),
    [#6540](https://github.com/PyTorchLightning/pytorch-lightning/pull/6540),
    [#6547](https://github.com/PyTorchLightning/pytorch-lightning/pull/6547),
    [#6515](https://github.com/PyTorchLightning/pytorch-lightning/pull/6515),
    [#6572](https://github.com/PyTorchLightning/pytorch-lightning/pull/6572),
    [#6573](https://github.com/PyTorchLightning/pytorch-lightning/pull/6573),
    [#6584](https://github.com/PyTorchLightning/pytorch-lightning/pull/6584),
    [#6636](https://github.com/PyTorchLightning/pytorch-lightning/pull/6636),
    [#6637](https://github.com/PyTorchLightning/pytorch-lightning/pull/6637),
    [#6649](https://github.com/PyTorchLightning/pytorch-lightning/pull/6649),
    [#6659](https://github.com/PyTorchLightning/pytorch-lightning/pull/6659),
    [#7131](https://github.com/PyTorchLightning/pytorch-lightning/pull/7131),
)
- Deprecated the `LightningModule.datamodule` getter and setter methods; access them through `Trainer.datamodule` instead ([#7168](https://github.com/PyTorchLightning/pytorch-lightning/pull/7168))
- Deprecated the use of `Trainer(gpus="i")` (string) for selecting the i-th GPU; from v1.5 this will set the number of GPUs instead of the index ([#6388](https://github.com/PyTorchLightning/pytorch-lightning/pull/6388))

### Removed

- Removed the `exp_save_path` property from the `LightningModule` ([#7266](https://github.com/PyTorchLightning/pytorch-lightning/pull/7266))
- Removed training loop explicitly calling `EarlyStopping.on_validation_end` if no validation is run ([#7069](https://github.com/PyTorchLightning/pytorch-lightning/pull/7069))
- Removed `automatic_optimization` as a property from the training loop in favor of `LightningModule.automatic_optimization` ([#7130](https://github.com/PyTorchLightning/pytorch-lightning/pull/7130))
- Removed evaluation loop legacy returns for `*_epoch_end` hooks ([#6973](https://github.com/PyTorchLightning/pytorch-lightning/pull/6973))
- Removed support for passing a bool value to `profiler` argument of Trainer ([#6164](https://github.com/PyTorchLightning/pytorch-lightning/pull/6164))
- Removed no return warning from val/test step ([#6139](https://github.com/PyTorchLightning/pytorch-lightning/pull/6139))
- Removed passing a `ModelCheckpoint` instance to `Trainer(checkpoint_callback)` ([#6166](https://github.com/PyTorchLightning/pytorch-lightning/pull/6166))
- Removed deprecated Trainer argument `enable_pl_optimizer` and `automatic_optimization` ([#6163](https://github.com/PyTorchLightning/pytorch-lightning/pull/6163))
- Removed deprecated metrics ([#6161](https://github.com/PyTorchLightning/pytorch-lightning/pull/6161))
    * from `pytorch_lightning.metrics.functional.classification` removed `to_onehot`, `to_categorical`, `get_num_classes`, `roc`, `multiclass_roc`, `average_precision`, `precision_recall_curve`, `multiclass_precision_recall_curve`
    * from `pytorch_lightning.metrics.functional.reduction` removed `reduce`, `class_reduce`
- Removed deprecated `ModelCheckpoint` arguments `prefix`, `mode="auto"` ([#6162](https://github.com/PyTorchLightning/pytorch-lightning/pull/6162))
- Removed `mode='auto'` from `EarlyStopping` ([#6167](https://github.com/PyTorchLightning/pytorch-lightning/pull/6167))
- Removed `epoch` and `step` arguments from `ModelCheckpoint.format_checkpoint_name()`, these are now included in the `metrics` argument ([#7344](https://github.com/PyTorchLightning/pytorch-lightning/pull/7344))
- Removed legacy references for magic keys in the `Result` object ([#6016](https://github.com/PyTorchLightning/pytorch-lightning/pull/6016))
- Removed deprecated `LightningModule` `hparams` setter ([#6207](https://github.com/PyTorchLightning/pytorch-lightning/pull/6207))
- Removed legacy code to log or include metrics in the progress bar by returning them in a dict with the `"log"/"progress_bar"` magic keys. Use `self.log` instead ([#6734](https://github.com/PyTorchLightning/pytorch-lightning/pull/6734))
- Removed `trainer.fit()` return value of `1`. It has no return now ([#7237](https://github.com/PyTorchLightning/pytorch-lightning/pull/7237))
- Removed `logger_connector` legacy code ([#6733](https://github.com/PyTorchLightning/pytorch-lightning/pull/6733))
- Removed unused mixin attributes ([#6487](https://github.com/PyTorchLightning/pytorch-lightning/pull/6487))

### Fixed

- Fixed NaN errors in progress bars when training with iterable datasets with no length defined ([#7306](https://github.com/PyTorchLightning/pytorch-lightning/pull/7306))
- Fixed attaching train and validation dataloaders when `reload_dataloaders_every_epoch=True` and `num_sanity_val_steps=0` ([#7207](https://github.com/PyTorchLightning/pytorch-lightning/pull/7207))
- Added a barrier in the accelerator `teardown` to synchronize processes before execution finishes ([#6814](https://github.com/PyTorchLightning/pytorch-lightning/pull/6814))
- Fixed multi-node DDP sub-process launch by using `local_rank` instead of `global_rank` for main process assertion ([#7061](https://github.com/PyTorchLightning/pytorch-lightning/pull/7061))
- Fixed incorrect removal of `WORLD_SIZE` environment variable in DDP training when launching with torch distributed/torchelastic ([#6942](https://github.com/PyTorchLightning/pytorch-lightning/pull/6942))
- Made the `Plugin.reduce` method more consistent across all Plugins to reflect a mean-reduction by default ([#6011](https://github.com/PyTorchLightning/pytorch-lightning/pull/6011))
- Move lightning module to correct device type when using LightningDistributedWrapper ([#6070](https://github.com/PyTorchLightning/pytorch-lightning/pull/6070))
- Do not print top-k verbose log with `ModelCheckpoint(monitor=None)` ([#6109](https://github.com/PyTorchLightning/pytorch-lightning/pull/6109))
- Fixed `ModelCheckpoint(save_top_k=0, save_last=True)` not saving the `last` checkpoint ([#6136](https://github.com/PyTorchLightning/pytorch-lightning/pull/6136))
- Fixed `.teardown(stage='fit')` and `.on_fit_{start,end}()` getting called during `trainer.test` ([#6386](https://github.com/PyTorchLightning/pytorch-lightning/pull/6386))
- Fixed LightningModule `all_gather` on cpu tensors ([#6416](https://github.com/PyTorchLightning/pytorch-lightning/pull/6416))
- Fixed torch distributed not available in setup hook for DDP ([#6506](https://github.com/PyTorchLightning/pytorch-lightning/pull/6506))
- Fixed `trainer.tuner.{lr_find,scale_batch_size}` not setting the `Trainer` state properly ([#7258](https://github.com/PyTorchLightning/pytorch-lightning/pull/7258))
- Fixed bug where the learning rate schedulers did not follow the optimizer frequencies ([#4868](https://github.com/PyTorchLightning/pytorch-lightning/pull/4868))
- Fixed pickle error checker to now check for `pickle.PickleError` to catch all pickle errors ([#6917](https://github.com/PyTorchLightning/pytorch-lightning/pull/6917))
- Fixed a bug where the outputs object passed to `LightningModule.training_epoch_end` was different from the object passed to the `on_train_end_epoch` hook ([#6969](https://github.com/PyTorchLightning/pytorch-lightning/pull/6969))
- Fixed a bug where the outputs passed to `train_batch_end` would be lists even when using a single optimizer and no truncated backprop through time steps ([#6969](https://github.com/PyTorchLightning/pytorch-lightning/pull/6969))
- Fixed bug for trainer error handling which would cause hang for distributed training ([#6864](https://github.com/PyTorchLightning/pytorch-lightning/pull/6864))
- Fixed `self.device` not returning the correct device in replicas of data-parallel ([#6414](https://github.com/PyTorchLightning/pytorch-lightning/pull/6414))
- Fixed `lr_find` trying beyond `num_training` steps and suggesting a too high learning rate ([#7076](https://github.com/PyTorchLightning/pytorch-lightning/pull/7076))
- Fixed logger creating incorrect version folder in DDP with repeated `Trainer.fit` calls ([#7077](https://github.com/PyTorchLightning/pytorch-lightning/pull/7077))
- Fixed metric objects passed directly to `self.log` not being reset correctly ([#7055](https://github.com/PyTorchLightning/pytorch-lightning/pull/7055))
- Fixed `CombinedLoader` in distributed settings for validation / testing ([#7102](https://github.com/PyTorchLightning/pytorch-lightning/pull/7102))
- Fixed the save_dir in `WandbLogger` when the run was initiated externally ([#7106](https://github.com/PyTorchLightning/pytorch-lightning/pull/7106))
- Fixed `num_sanity_val_steps` affecting reproducibility of training data shuffling ([#7014](https://github.com/PyTorchLightning/pytorch-lightning/pull/7014))
- Fixed resetting device after `fitting/evaluating/predicting` ([#7188](https://github.com/PyTorchLightning/pytorch-lightning/pull/7188))
- Fixed bug where `trainer.tuner.scale_batch_size(max_trials=0)` would not return the correct batch size result ([#7262](https://github.com/PyTorchLightning/pytorch-lightning/pull/7262))
- Fixed metrics not being properly logged with `precision=16` and `manual_optimization` ([#7228](https://github.com/PyTorchLightning/pytorch-lightning/pull/7228))
- Fixed `BaseFinetuning` properly reloading `optimizer_states` when using `resume_from_checkpoint` ([#6891](https://github.com/PyTorchLightning/pytorch-lightning/pull/6891))
- Fixed `parameters_to_ignore` not properly set to DDPWrapper ([#7239](https://github.com/PyTorchLightning/pytorch-lightning/pull/7239))
- Fixed parsing of `fast_dev_run=True` with the built-in `ArgumentParser` ([#7240](https://github.com/PyTorchLightning/pytorch-lightning/pull/7240))
- Fixed handling an `IterableDataset` that fails to produce a batch at the beginning of an epoch ([#7294](https://github.com/PyTorchLightning/pytorch-lightning/pull/7294))
- Fixed `LightningModule.save_hyperparameters()` when attempting to save an empty container ([#7268](https://github.com/PyTorchLightning/pytorch-lightning/pull/7268))
- Fixed `apex` not properly instantiated when running with `ddp` ([#7274](https://github.com/PyTorchLightning/pytorch-lightning/pull/7274))
- Fixed optimizer `state` not moved to `GPU` ([#7277](https://github.com/PyTorchLightning/pytorch-lightning/pull/7277))
- Fixed custom init args for `WandbLogger` ([#6989](https://github.com/PyTorchLightning/pytorch-lightning/pull/6989))
- Fixed a bug where an error would be raised if the train dataloader sometimes produced None for a batch ([#7342](https://github.com/PyTorchLightning/pytorch-lightning/pull/7342))
- Fixed examples (
    [#6600](https://github.com/PyTorchLightning/pytorch-lightning/pull/6600),
    [#6638](https://github.com/PyTorchLightning/pytorch-lightning/pull/6638),
    [#7096](https://github.com/PyTorchLightning/pytorch-lightning/pull/7096),
    [#7246](https://github.com/PyTorchLightning/pytorch-lightning/pull/7246),
    [#6357](https://github.com/PyTorchLightning/pytorch-lightning/pull/6357),
    [#6476](https://github.com/PyTorchLightning/pytorch-lightning/pull/6476),
    [#6294](https://github.com/PyTorchLightning/pytorch-lightning/pull/6294),
    [#6373](https://github.com/PyTorchLightning/pytorch-lightning/pull/6373),
    [#6088](https://github.com/PyTorchLightning/pytorch-lightning/pull/6088),
    [#7398](https://github.com/PyTorchLightning/pytorch-lightning/pull/7398)
)
- Resolved schedule step bug for PyTorch Profiler ([#6674](https://github.com/PyTorchLightning/pytorch-lightning/pull/6674),
    [#6681](https://github.com/PyTorchLightning/pytorch-lightning/pull/6681))
- Updated logic for checking TPUs availability ([#6767](https://github.com/PyTorchLightning/pytorch-lightning/pull/6767))
- Resolve TPU miss rendezvous ([#6781](https://github.com/PyTorchLightning/pytorch-lightning/pull/6781))
- Fixed auto-scaling mode when calling tune method on trainer ([#7321](https://github.com/PyTorchLightning/pytorch-lightning/pull/7321))
- Fixed finetuning complex models correctly unfreezes ([#6880](https://github.com/PyTorchLightning/pytorch-lightning/pull/6880))
- Ensure we set the eval/train flag correctly on accelerator model ([#6877](https://github.com/PyTorchLightning/pytorch-lightning/pull/6877))
- Set better defaults for `rank_zero_only.rank` when training is launched with SLURM and torchelastic ([#6802](https://github.com/PyTorchLightning/pytorch-lightning/pull/6802))
- Fixed matching the number of outputs of backward with forward for AllGatherGrad ([#6625](https://github.com/PyTorchLightning/pytorch-lightning/pull/6625))
- Fixed the `gradient_clip_algorithm` has no effect ([#6928](https://github.com/PyTorchLightning/pytorch-lightning/pull/6928))
- Fixed CUDA OOM detection and handling ([#6934](https://github.com/PyTorchLightning/pytorch-lightning/pull/6934))
- Fixed `unfreeze_and_add_param_group` expects `modules` rather than `module` ([#6822](https://github.com/PyTorchLightning/pytorch-lightning/pull/6822))
- Fixed DPP + SyncBN when move on device ([#6838](https://github.com/PyTorchLightning/pytorch-lightning/pull/6838))
- Fixed missing arguments in `lr_find` call ([#6784](https://github.com/PyTorchLightning/pytorch-lightning/pull/6784))
- Fixed `set_default_tensor_type` to `torch.DoubleTensor` with precision=64 ([#7108](https://github.com/PyTorchLightning/pytorch-lightning/pull/7108))
- Fixed `NeptuneLogger.log_text(step=None)` ([#7194](https://github.com/PyTorchLightning/pytorch-lightning/pull/7194))
- Fixed importing torchtext batch ([#6365](https://github.com/PyTorchLightning/pytorch-lightning/pull/6365),
    [#6323](https://github.com/PyTorchLightning/pytorch-lightning/pull/6323),
    [#6211](https://github.com/PyTorchLightning/pytorch-lightning/pull/6211))


## [1.2.9] - 2021-04-20

### Fixed

- Fixed the order to call for world ranks & the `root_device` property in `TPUSpawnPlugin` ([#7074](https://github.com/PyTorchLightning/pytorch-lightning/pull/7074))
- Fixed multi-gpu join for Horovod ([#6954](https://github.com/PyTorchLightning/pytorch-lightning/pull/6954))
- Fixed parsing for pre-release package versions ([#6999](https://github.com/PyTorchLightning/pytorch-lightning/pull/6999))


## [1.2.8] - 2021-04-14

### Added

- Added TPUSpawn + IterableDataset error message ([#6875](https://github.com/PyTorchLightning/pytorch-lightning/pull/6875))

### Fixed

- Fixed process rank not being available right away after `Trainer` instantiation ([#6941](https://github.com/PyTorchLightning/pytorch-lightning/pull/6941))
- Fixed `sync_dist` for tpus ([#6950](https://github.com/PyTorchLightning/pytorch-lightning/pull/6950))
- Fixed `AttributeError` for `require_backward_grad_sync` when running manual optimization with sharded plugin ([#6915](https://github.com/PyTorchLightning/pytorch-lightning/pull/6915))
- Fixed `--gpus` default for parser returned by `Trainer.add_argparse_args` ([#6898](https://github.com/PyTorchLightning/pytorch-lightning/pull/6898))
- Fixed TPU Spawn all gather ([#6896](https://github.com/PyTorchLightning/pytorch-lightning/pull/6896))
- Fixed `EarlyStopping` logic when `min_epochs` or `min_steps` requirement is not met ([#6705](https://github.com/PyTorchLightning/pytorch-lightning/pull/6705))
- Fixed csv extension check ([#6436](https://github.com/PyTorchLightning/pytorch-lightning/pull/6436))
- Fixed checkpoint issue when using Horovod distributed backend ([#6958](https://github.com/PyTorchLightning/pytorch-lightning/pull/6958))
- Fixed tensorboard exception raising ([#6901](https://github.com/PyTorchLightning/pytorch-lightning/pull/6901))
- Fixed setting the eval/train flag correctly on accelerator model ([#6983](https://github.com/PyTorchLightning/pytorch-lightning/pull/6983))
- Fixed DDP_SPAWN compatibility with bug_report_model.py ([#6892](https://github.com/PyTorchLightning/pytorch-lightning/pull/6892))
- Fixed bug where `BaseFinetuning.flatten_modules()` was duplicating leaf node parameters ([#6879](https://github.com/PyTorchLightning/pytorch-lightning/pull/6879))
- Set better defaults for `rank_zero_only.rank` when training is launched with SLURM and torchelastic:
    * Support SLURM and torchelastic global rank environment variables ([#5715](https://github.com/PyTorchLightning/pytorch-lightning/pull/5715))
    * Remove hardcoding of local rank in accelerator connector ([#6878](https://github.com/PyTorchLightning/pytorch-lightning/pull/6878))


## [1.2.7] - 2021-04-06

### Fixed

- Fixed resolve a bug with omegaconf and xm.save ([#6741](https://github.com/PyTorchLightning/pytorch-lightning/pull/6741))
- Fixed an issue with IterableDataset when __len__ is not defined ([#6828](https://github.com/PyTorchLightning/pytorch-lightning/pull/6828))
- Sanitize None params during pruning ([#6836](https://github.com/PyTorchLightning/pytorch-lightning/pull/6836))
- Enforce an epoch scheduler interval when using SWA ([#6588](https://github.com/PyTorchLightning/pytorch-lightning/pull/6588))
- Fixed TPU Colab hang issue, post training ([#6816](https://github.com/PyTorchLightning/pytorch-lightning/pull/6816))
- Fixed a bug where `TensorBoardLogger` would give a warning and not log correctly to a symbolic link `save_dir` ([#6730](https://github.com/PyTorchLightning/pytorch-lightning/pull/6730))
- Fixed bug where `predict` could not be used when `progress_bar_refresh_rate=0` ([#6884](https://github.com/PyTorchLightning/pytorch-lightning/pull/6884))


## [1.2.6] - 2021-03-30

### Changed

- Changed the behavior of `on_epoch_start` to run at the beginning of validation & test epoch ([#6498](https://github.com/PyTorchLightning/pytorch-lightning/pull/6498))

### Removed

- Removed legacy code to include `step` dictionary returns in `callback_metrics`. Use `self.log_dict` instead. ([#6682](https://github.com/PyTorchLightning/pytorch-lightning/pull/6682))

### Fixed

- Fixed `DummyLogger.log_hyperparams` raising a `TypeError` when running with `fast_dev_run=True` ([#6398](https://github.com/PyTorchLightning/pytorch-lightning/pull/6398))
- Fixed error on TPUs when there was no `ModelCheckpoint` ([#6654](https://github.com/PyTorchLightning/pytorch-lightning/pull/6654))
- Fixed `trainer.test` freeze on TPUs ([#6654](https://github.com/PyTorchLightning/pytorch-lightning/pull/6654))
- Fixed a bug where gradients were disabled after calling `Trainer.predict` ([#6657](https://github.com/PyTorchLightning/pytorch-lightning/pull/6657))
- Fixed bug where no TPUs were detected in a TPU pod env ([#6719](https://github.com/PyTorchLightning/pytorch-lightning/pull/6719))


## [1.2.5] - 2021-03-23

### Changed

- Update Gradient Clipping for the TPU Accelerator ([#6576](https://github.com/PyTorchLightning/pytorch-lightning/pull/6576))
- Refactored setup for typing friendly ([#6590](https://github.com/PyTorchLightning/pytorch-lightning/pull/6590))

### Fixed

- Fixed a bug where `all_gather` would not work correctly with `tpu_cores=8` ([#6587](https://github.com/PyTorchLightning/pytorch-lightning/pull/6587))
- Fixed comparing required versions ([#6434](https://github.com/PyTorchLightning/pytorch-lightning/pull/6434))
- Fixed duplicate logs appearing in console when using the python logging module ([#6275](https://github.com/PyTorchLightning/pytorch-lightning/pull/6275))
- Added Autocast in validation, test and predict modes for Native AMP ([#6565](https://github.com/PyTorchLightning/pytorch-lightning/pull/6565))


## [1.2.4] - 2021-03-16

### Changed

- Changed the default of `find_unused_parameters` back to `True` in DDP and DDP Spawn ([#6438](https://github.com/PyTorchLightning/pytorch-lightning/pull/6438))

### Fixed

- Expose DeepSpeed loss parameters to allow users to fix loss instability ([#6115](https://github.com/PyTorchLightning/pytorch-lightning/pull/6115))
- Fixed DP reduction with collection ([#6324](https://github.com/PyTorchLightning/pytorch-lightning/pull/6324))
- Fixed an issue where the tuner would not tune the learning rate if also tuning the batch size ([#4688](https://github.com/PyTorchLightning/pytorch-lightning/pull/4688))
- Fixed broadcast to use PyTorch `broadcast_object_list` and add `reduce_decision` ([#6410](https://github.com/PyTorchLightning/pytorch-lightning/pull/6410))
- Fixed logger creating directory structure too early in DDP ([#6380](https://github.com/PyTorchLightning/pytorch-lightning/pull/6380))
- Fixed DeepSpeed additional memory use on rank 0 when default device not set early enough ([#6460](https://github.com/PyTorchLightning/pytorch-lightning/pull/6460))
- Fixed an issue with `Tuner.scale_batch_size` not finding the batch size attribute in the datamodule ([#5968](https://github.com/PyTorchLightning/pytorch-lightning/pull/5968))
- Fixed an exception in the layer summary when the model contains torch.jit scripted submodules ([#6511](https://github.com/PyTorchLightning/pytorch-lightning/pull/6511))
- Fixed when Train loop config was run during `Trainer.predict` ([#6541](https://github.com/PyTorchLightning/pytorch-lightning/pull/6541))


## [1.2.3] - 2021-03-09

### Fixed

- Fixed `ModelPruning(make_pruning_permanent=True)` pruning buffers getting removed when saved during training ([#6073](https://github.com/PyTorchLightning/pytorch-lightning/pull/6073))
- Fixed when `_stable_1d_sort` to work when `n >= N` ([#6177](https://github.com/PyTorchLightning/pytorch-lightning/pull/6177))
- Fixed `AttributeError` when `logger=None` on TPU ([#6221](https://github.com/PyTorchLightning/pytorch-lightning/pull/6221))
- Fixed PyTorch Profiler with `emit_nvtx` ([#6260](https://github.com/PyTorchLightning/pytorch-lightning/pull/6260))
- Fixed `trainer.test` from `best_path` hangs after calling `trainer.fit`  ([#6272](https://github.com/PyTorchLightning/pytorch-lightning/pull/6272))
- Fixed `SingleTPU` calling `all_gather` ([#6296](https://github.com/PyTorchLightning/pytorch-lightning/pull/6296))
- Ensure we check DeepSpeed/Sharded in multi-node DDP ([#6297](https://github.com/PyTorchLightning/pytorch-lightning/pull/6297)
- Check `LightningOptimizer` doesn't delete optimizer hooks ([#6305](https://github.com/PyTorchLightning/pytorch-lightning/pull/6305)
- Resolve memory leak for evaluation ([#6326](https://github.com/PyTorchLightning/pytorch-lightning/pull/6326)
- Ensure that clip gradients is only called if the value is greater than 0 ([#6330](https://github.com/PyTorchLightning/pytorch-lightning/pull/6330)
- Fixed `Trainer` not resetting `lightning_optimizers` when calling `Trainer.fit()` multiple times ([#6372](https://github.com/PyTorchLightning/pytorch-lightning/pull/6372))


## [1.2.2] - 2021-03-02

### Added

- Added `checkpoint` parameter to callback's `on_save_checkpoint` hook ([#6072](https://github.com/PyTorchLightning/pytorch-lightning/pull/6072))

### Changed

- Changed the order of `backward`, `step`, `zero_grad` to `zero_grad`, `backward`, `step` ([#6147](https://github.com/PyTorchLightning/pytorch-lightning/pull/6147))
- Changed default for DeepSpeed CPU Offload to False, due to prohibitively slow speeds at smaller scale ([#6262](https://github.com/PyTorchLightning/pytorch-lightning/pull/6262))

### Fixed

- Fixed epoch level schedulers not being called when `val_check_interval < 1.0` ([#6075](https://github.com/PyTorchLightning/pytorch-lightning/pull/6075))
- Fixed multiple early stopping callbacks ([#6197](https://github.com/PyTorchLightning/pytorch-lightning/pull/6197))
- Fixed incorrect usage of `detach()`, `cpu()`, `to()` ([#6216](https://github.com/PyTorchLightning/pytorch-lightning/pull/6216))
- Fixed LBFGS optimizer support which didn't converge in automatic optimization ([#6147](https://github.com/PyTorchLightning/pytorch-lightning/pull/6147))
- Prevent `WandbLogger` from dropping values ([#5931](https://github.com/PyTorchLightning/pytorch-lightning/pull/5931))
- Fixed error thrown when using valid distributed mode in multi node ([#6297](https://github.com/PyTorchLightning/pytorch-lightning/pull/6297)


## [1.2.1] - 2021-02-23

### Fixed

- Fixed incorrect yield logic for the amp autocast context manager ([#6080](https://github.com/PyTorchLightning/pytorch-lightning/pull/6080))
- Fixed priority of plugin/accelerator when setting distributed mode ([#6089](https://github.com/PyTorchLightning/pytorch-lightning/pull/6089))
- Fixed error message for AMP + CPU incompatibility ([#6107](https://github.com/PyTorchLightning/pytorch-lightning/pull/6107))
- Disabled batch transfer in DP mode ([#6093](https://github.com/PyTorchLightning/pytorch-lightning/pull/6093))


## [1.2.0] - 2021-02-18

### Added

- Added `DataType`, `AverageMethod` and `MDMCAverageMethod` enum in metrics ([#5657](https://github.com/PyTorchLightning/pytorch-lightning/pull/5689))
- Added support for summarized model total params size in megabytes ([#5590](https://github.com/PyTorchLightning/pytorch-lightning/pull/5590))
- Added support for multiple train loaders ([#1959](https://github.com/PyTorchLightning/pytorch-lightning/pull/1959))
- Added `Accuracy` metric now generalizes to Top-k accuracy for (multi-dimensional) multi-class inputs using the `top_k` parameter ([#4838](https://github.com/PyTorchLightning/pytorch-lightning/pull/4838))
- Added `Accuracy` metric now enables the computation of subset accuracy for multi-label or multi-dimensional multi-class inputs with the `subset_accuracy` parameter ([#4838](https://github.com/PyTorchLightning/pytorch-lightning/pull/4838))
- Added `HammingDistance` metric to compute the hamming distance (loss) ([#4838](https://github.com/PyTorchLightning/pytorch-lightning/pull/4838))
- Added `max_fpr` parameter to `auroc` metric for computing partial auroc metric ([#3790](https://github.com/PyTorchLightning/pytorch-lightning/pull/3790))
- Added `StatScores` metric to compute the number of true positives, false positives, true negatives and false negatives ([#4839](https://github.com/PyTorchLightning/pytorch-lightning/pull/4839))
- Added `R2Score` metric ([#5241](https://github.com/PyTorchLightning/pytorch-lightning/pull/5241))
- Added `LambdaCallback` ([#5347](https://github.com/PyTorchLightning/pytorch-lightning/pull/5347))
- Added `BackboneLambdaFinetuningCallback` ([#5377](https://github.com/PyTorchLightning/pytorch-lightning/pull/5377))
- Accelerator `all_gather` supports collection ([#5221](https://github.com/PyTorchLightning/pytorch-lightning/pull/5221))
- Added `image_gradients` functional metric to compute the image gradients of a given input image. ([#5056](https://github.com/PyTorchLightning/pytorch-lightning/pull/5056))
- Added `MetricCollection` ([#4318](https://github.com/PyTorchLightning/pytorch-lightning/pull/4318))
- Added `.clone()` method to metrics ([#4318](https://github.com/PyTorchLightning/pytorch-lightning/pull/4318))
- Added `IoU` class interface ([#4704](https://github.com/PyTorchLightning/pytorch-lightning/pull/4704))
- Support to tie weights after moving model to TPU via `on_post_move_to_device` hook
- Added missing val/test hooks in `LightningModule` ([#5467](https://github.com/PyTorchLightning/pytorch-lightning/pull/5467))
- The `Recall` and `Precision` metrics (and their functional counterparts `recall` and `precision`) can now be generalized to Recall@K and Precision@K with the use of `top_k` parameter ([#4842](https://github.com/PyTorchLightning/pytorch-lightning/pull/4842))
- Added `ModelPruning` Callback ([#5618](https://github.com/PyTorchLightning/pytorch-lightning/pull/5618),
    [#5825](https://github.com/PyTorchLightning/pytorch-lightning/pull/5825),
    [#6045](https://github.com/PyTorchLightning/pytorch-lightning/pull/6045))
- Added `PyTorchProfiler` ([#5560](https://github.com/PyTorchLightning/pytorch-lightning/pull/5560))
- Added compositional metrics ([#5464](https://github.com/PyTorchLightning/pytorch-lightning/pull/5464))
- Added Trainer method `predict(...)` for high performence predictions ([#5579](https://github.com/PyTorchLightning/pytorch-lightning/pull/5579))
- Added `on_before_batch_transfer` and `on_after_batch_transfer` data hooks ([#3671](https://github.com/PyTorchLightning/pytorch-lightning/pull/3671))
- Added AUC/AUROC class interface ([#5479](https://github.com/PyTorchLightning/pytorch-lightning/pull/5479))
- Added `PredictLoop` object ([#5752](https://github.com/PyTorchLightning/pytorch-lightning/pull/5752))
- Added `QuantizationAwareTraining` callback ([#5706](https://github.com/PyTorchLightning/pytorch-lightning/pull/5706),
    [#6040](https://github.com/PyTorchLightning/pytorch-lightning/pull/6040))
- Added `LightningModule.configure_callbacks` to enable the definition of model-specific callbacks ([#5621](https://github.com/PyTorchLightning/pytorch-lightning/pull/5621))
- Added `dim` to `PSNR` metric for mean-squared-error reduction ([#5957](https://github.com/PyTorchLightning/pytorch-lightning/pull/5957))
- Added promxial policy optimization template to pl_examples ([#5394](https://github.com/PyTorchLightning/pytorch-lightning/pull/5394))
- Added `log_graph` to `CometLogger` ([#5295](https://github.com/PyTorchLightning/pytorch-lightning/pull/5295))
- Added possibility for nested loaders ([#5404](https://github.com/PyTorchLightning/pytorch-lightning/pull/5404))
- Added `sync_step` to Wandb logger ([#5351](https://github.com/PyTorchLightning/pytorch-lightning/pull/5351))
- Added `StochasticWeightAveraging` callback ([#5640](https://github.com/PyTorchLightning/pytorch-lightning/pull/5640))
- Added `LightningDataModule.from_datasets(...)` ([#5133](https://github.com/PyTorchLightning/pytorch-lightning/pull/5133))
- Added `PL_TORCH_DISTRIBUTED_BACKEND` env variable to select backend ([#5981](https://github.com/PyTorchLightning/pytorch-lightning/pull/5981))
- Added `Trainer` flag to activate Stochastic Weight Averaging (SWA) `Trainer(stochastic_weight_avg=True)` ([#6038](https://github.com/PyTorchLightning/pytorch-lightning/pull/6038))
- Added DeepSpeed integration ([#5954](https://github.com/PyTorchLightning/pytorch-lightning/pull/5954),
    [#6042](https://github.com/PyTorchLightning/pytorch-lightning/pull/6042))

### Changed

- Changed `stat_scores` metric now calculates stat scores over all classes and gains new parameters, in line with the new `StatScores` metric ([#4839](https://github.com/PyTorchLightning/pytorch-lightning/pull/4839))
- Changed `computer_vision_fine_tunning` example to use `BackboneLambdaFinetuningCallback` ([#5377](https://github.com/PyTorchLightning/pytorch-lightning/pull/5377))
- Changed `automatic casting` for LoggerConnector `metrics` ([#5218](https://github.com/PyTorchLightning/pytorch-lightning/pull/5218))
- Changed `iou` [func] to allow float input ([#4704](https://github.com/PyTorchLightning/pytorch-lightning/pull/4704))
- Metric `compute()` method will no longer automatically call `reset()` ([#5409](https://github.com/PyTorchLightning/pytorch-lightning/pull/5409))
- Set PyTorch 1.4 as min requirements, also for testing and examples `torchvision>=0.5` and `torchtext>=0.5` ([#5418](https://github.com/PyTorchLightning/pytorch-lightning/pull/5418))
- Changed `callbacks` argument in `Trainer` to allow `Callback` input ([#5446](https://github.com/PyTorchLightning/pytorch-lightning/pull/5446))
- Changed the default of `find_unused_parameters` to `False` in DDP ([#5185](https://github.com/PyTorchLightning/pytorch-lightning/pull/5185))
- Changed `ModelCheckpoint` version suffixes to start at 1 ([#5008](https://github.com/PyTorchLightning/pytorch-lightning/pull/5008))
- Progress bar metrics tensors are now converted to float ([#5692](https://github.com/PyTorchLightning/pytorch-lightning/pull/5692))
- Changed the default value for the `progress_bar_refresh_rate` Trainer argument in Google COLAB notebooks to 20 ([#5516](https://github.com/PyTorchLightning/pytorch-lightning/pull/5516))
- Extended support for purely iteration-based training ([#5726](https://github.com/PyTorchLightning/pytorch-lightning/pull/5726))
- Made `LightningModule.global_rank`, `LightningModule.local_rank` and `LightningModule.logger` read-only properties ([#5730](https://github.com/PyTorchLightning/pytorch-lightning/pull/5730))
- Forced `ModelCheckpoint` callbacks to run after all others to guarantee all states are saved to the checkpoint ([#5731](https://github.com/PyTorchLightning/pytorch-lightning/pull/5731))
- Refactored Accelerators and Plugins:
    * Added base classes for plugins ([#5715](https://github.com/PyTorchLightning/pytorch-lightning/pull/5715))
    * Added parallel plugins for DP, DDP, DDPSpawn, DDP2 and Horovod ([#5714](https://github.com/PyTorchLightning/pytorch-lightning/pull/5714))
    * Precision Plugins ([#5718](https://github.com/PyTorchLightning/pytorch-lightning/pull/5718))
    * Added new Accelerators for CPU, GPU and TPU ([#5719](https://github.com/PyTorchLightning/pytorch-lightning/pull/5719))
    * Added RPC and Sharded plugins ([#5732](https://github.com/PyTorchLightning/pytorch-lightning/pull/5732))
    * Added missing `LightningModule`-wrapper logic to new plugins and accelerator ([#5734](https://github.com/PyTorchLightning/pytorch-lightning/pull/5734))
    * Moved device-specific teardown logic from training loop to accelerator ([#5973](https://github.com/PyTorchLightning/pytorch-lightning/pull/5973))
    * Moved accelerator_connector.py to the connectors subfolder ([#6033](https://github.com/PyTorchLightning/pytorch-lightning/pull/6033))
    * Trainer only references accelerator ([#6039](https://github.com/PyTorchLightning/pytorch-lightning/pull/6039))
    * Made parallel devices optional across all plugins ([#6051](https://github.com/PyTorchLightning/pytorch-lightning/pull/6051))
    * Cleaning ([#5948](https://github.com/PyTorchLightning/pytorch-lightning/pull/5948),
        [#5949](https://github.com/PyTorchLightning/pytorch-lightning/pull/5949),
        [#5950](https://github.com/PyTorchLightning/pytorch-lightning/pull/5950))
- Enabled `self.log` in callbacks ([#5094](https://github.com/PyTorchLightning/pytorch-lightning/pull/5094))
- Renamed xxx_AVAILABLE as protected ([#5082](https://github.com/PyTorchLightning/pytorch-lightning/pull/5082))
- Unified module names in Utils ([#5199](https://github.com/PyTorchLightning/pytorch-lightning/pull/5199))
- Separated utils: imports & enums ([#5256](https://github.com/PyTorchLightning/pytorch-lightning/pull/5256)
    [#5874](https://github.com/PyTorchLightning/pytorch-lightning/pull/5874))
- Refactor: clean trainer device & distributed getters ([#5300](https://github.com/PyTorchLightning/pytorch-lightning/pull/5300))
- Simplified training phase as LightningEnum ([#5419](https://github.com/PyTorchLightning/pytorch-lightning/pull/5419))
- Updated metrics to use LightningEnum ([#5689](https://github.com/PyTorchLightning/pytorch-lightning/pull/5689))
- Changed the seq of `on_train_batch_end`, `on_batch_end` & `on_train_epoch_end`, `on_epoch_end hooks` ([#5688](https://github.com/PyTorchLightning/pytorch-lightning/pull/5688))
- Refactored `setup_training` and remove `test_mode` ([#5388](https://github.com/PyTorchLightning/pytorch-lightning/pull/5388))
- Disabled training with zero `num_training_batches` when insufficient `limit_train_batches` ([#5703](https://github.com/PyTorchLightning/pytorch-lightning/pull/5703))
- Refactored `EpochResultStore` ([#5522](https://github.com/PyTorchLightning/pytorch-lightning/pull/5522))
- Update `lr_finder` to check for attribute if not running `fast_dev_run` ([#5990](https://github.com/PyTorchLightning/pytorch-lightning/pull/5990))
- LightningOptimizer manual optimizer is more flexible and expose `toggle_model` ([#5771](https://github.com/PyTorchLightning/pytorch-lightning/pull/5771))
- `MlflowLogger` limit parameter value length to 250 char ([#5893](https://github.com/PyTorchLightning/pytorch-lightning/pull/5893))
- Re-introduced fix for Hydra directory sync with multiple process ([#5993](https://github.com/PyTorchLightning/pytorch-lightning/pull/5993))

### Deprecated

- Function `stat_scores_multiple_classes` is deprecated in favor of `stat_scores` ([#4839](https://github.com/PyTorchLightning/pytorch-lightning/pull/4839))
- Moved accelerators and plugins to its `legacy` pkg ([#5645](https://github.com/PyTorchLightning/pytorch-lightning/pull/5645))
- Deprecated `LightningDistributedDataParallel` in favor of new wrapper module `LightningDistributedModule` ([#5185](https://github.com/PyTorchLightning/pytorch-lightning/pull/5185))
- Deprecated `LightningDataParallel` in favor of new wrapper module `LightningParallelModule` ([#5670](https://github.com/PyTorchLightning/pytorch-lightning/pull/5670))
- Renamed utils modules ([#5199](https://github.com/PyTorchLightning/pytorch-lightning/pull/5199))
    * `argparse_utils` >> `argparse`
    * `model_utils` >> `model_helpers`
    * `warning_utils` >> `warnings`
    * `xla_device_utils` >> `xla_device`
- Deprecated using `'val_loss'` to set the `ModelCheckpoint` monitor ([#6012](https://github.com/PyTorchLightning/pytorch-lightning/pull/6012))
- Deprecated `.get_model()` with explicit `.lightning_module` property ([#6035](https://github.com/PyTorchLightning/pytorch-lightning/pull/6035))
- Deprecated Trainer attribute `accelerator_backend` in favor of `accelerator` ([#6034](https://github.com/PyTorchLightning/pytorch-lightning/pull/6034))

### Removed

- Removed deprecated checkpoint argument `filepath` ([#5321](https://github.com/PyTorchLightning/pytorch-lightning/pull/5321))
- Removed deprecated `Fbeta`, `f1_score` and `fbeta_score` metrics ([#5322](https://github.com/PyTorchLightning/pytorch-lightning/pull/5322))
- Removed deprecated `TrainResult` ([#5323](https://github.com/PyTorchLightning/pytorch-lightning/pull/5323))
- Removed deprecated `EvalResult` ([#5633](https://github.com/PyTorchLightning/pytorch-lightning/pull/5633))
- Removed `LoggerStages` ([#5673](https://github.com/PyTorchLightning/pytorch-lightning/pull/5673))

### Fixed

- Fixed distributed setting and `ddp_cpu` only with `num_processes>1` ([#5297](https://github.com/PyTorchLightning/pytorch-lightning/pull/5297))
- Fixed `num_workers` for Windows example ([#5375](https://github.com/PyTorchLightning/pytorch-lightning/pull/5375))
- Fixed loading yaml ([#5619](https://github.com/PyTorchLightning/pytorch-lightning/pull/5619))
- Fixed support custom DataLoader with DDP if they can be re-instantiated ([#5745](https://github.com/PyTorchLightning/pytorch-lightning/pull/5745))
- Fixed repeated `.fit()` calls ignore max_steps iteration bound ([#5936](https://github.com/PyTorchLightning/pytorch-lightning/pull/5936))
- Fixed throwing `MisconfigurationError` on unknown mode ([#5255](https://github.com/PyTorchLightning/pytorch-lightning/pull/5255))
- Resolve bug with Finetuning ([#5744](https://github.com/PyTorchLightning/pytorch-lightning/pull/5744))
- Fixed `ModelCheckpoint` race condition in file existence check ([#5155](https://github.com/PyTorchLightning/pytorch-lightning/pull/5155))
- Fixed some compatibility with PyTorch 1.8 ([#5864](https://github.com/PyTorchLightning/pytorch-lightning/pull/5864))
- Fixed forward cache ([#5895](https://github.com/PyTorchLightning/pytorch-lightning/pull/5895))
- Fixed recursive detach of tensors to CPU ([#6007](https://github.com/PyTorchLightning/pytorch-lightning/pull/6007))
- Fixed passing wrong strings for scheduler interval doesn't throw an error ([#5923](https://github.com/PyTorchLightning/pytorch-lightning/pull/5923))
- Fixed wrong `requires_grad` state after `return None` with multiple optimizers ([#5738](https://github.com/PyTorchLightning/pytorch-lightning/pull/5638))
- Fixed add `on_epoch_end` hook at the end of `validation`, `test` epoch ([#5986](https://github.com/PyTorchLightning/pytorch-lightning/pull/5986))
- Fixed missing `process_dataloader` call for `TPUSpawn` when in distributed mode ([#6015](https://github.com/PyTorchLightning/pytorch-lightning/pull/6015))
- Fixed progress bar flickering by appending 0 to floats/strings ([#6009](https://github.com/PyTorchLightning/pytorch-lightning/pull/6009))
- Fixed synchronization issues with TPU training ([#6027](https://github.com/PyTorchLightning/pytorch-lightning/pull/6027))
- Fixed `hparams.yaml` saved twice when using `TensorBoardLogger` ([#5953](https://github.com/PyTorchLightning/pytorch-lightning/pull/5953))
- Fixed basic examples ([#5912](https://github.com/PyTorchLightning/pytorch-lightning/pull/5912),
    [#5985](https://github.com/PyTorchLightning/pytorch-lightning/pull/5985))
- Fixed `fairscale` compatible with PT 1.8 ([#5996](https://github.com/PyTorchLightning/pytorch-lightning/pull/5996))
- Ensured `process_dataloader` is called when `tpu_cores > 1` to use Parallel DataLoader ([#6015](https://github.com/PyTorchLightning/pytorch-lightning/pull/6015))
- Attempted SLURM auto resume call when non-shell call fails ([#6002](https://github.com/PyTorchLightning/pytorch-lightning/pull/6002))
- Fixed wrapping optimizers upon assignment ([#6006](https://github.com/PyTorchLightning/pytorch-lightning/pull/6006))
- Fixed allowing hashing of metrics with lists in their state ([#5939](https://github.com/PyTorchLightning/pytorch-lightning/pull/5939))


## [1.1.8] - 2021-02-08

### Fixed

- Separate epoch validation from step validation ([#5208](https://github.com/PyTorchLightning/pytorch-lightning/pull/5208))
- Fixed `toggle_optimizers` not handling all optimizer parameters ([#5775](https://github.com/PyTorchLightning/pytorch-lightning/pull/5775))


## [1.1.7] - 2021-02-03

### Fixed

- Fixed `TensorBoardLogger` not closing `SummaryWriter` on `finalize` ([#5696](https://github.com/PyTorchLightning/pytorch-lightning/pull/5696))
- Fixed filtering of pytorch  "unsqueeze" warning when using DP ([#5622](https://github.com/PyTorchLightning/pytorch-lightning/pull/5622))
- Fixed `num_classes` argument in F1 metric ([#5663](https://github.com/PyTorchLightning/pytorch-lightning/pull/5663))
- Fixed `log_dir` property ([#5537](https://github.com/PyTorchLightning/pytorch-lightning/pull/5537))
- Fixed a race condition in `ModelCheckpoint` when checking if a checkpoint file exists ([#5144](https://github.com/PyTorchLightning/pytorch-lightning/pull/5144))
- Remove unnecessary intermediate layers in Dockerfiles ([#5697](https://github.com/PyTorchLightning/pytorch-lightning/pull/5697))
- Fixed auto learning rate ordering ([#5638](https://github.com/PyTorchLightning/pytorch-lightning/pull/5638))


## [1.1.6] - 2021-01-26

### Changed

- Increased TPU check timeout from 20s to 100s ([#5598](https://github.com/PyTorchLightning/pytorch-lightning/pull/5598))
- Ignored `step` param in Neptune logger's log_metric method ([#5510](https://github.com/PyTorchLightning/pytorch-lightning/pull/5510))
- Pass batch outputs to `on_train_batch_end` instead of `epoch_end` outputs ([#4369](https://github.com/PyTorchLightning/pytorch-lightning/pull/4369))

### Fixed

- Fixed `toggle_optimizer` to reset `requires_grad` state  ([#5574](https://github.com/PyTorchLightning/pytorch-lightning/pull/5574))
- Fixed FileNotFoundError for best checkpoint when using DDP with Hydra ([#5629](https://github.com/PyTorchLightning/pytorch-lightning/pull/5629))
- Fixed an error when logging a progress bar metric with a reserved name ([#5620](https://github.com/PyTorchLightning/pytorch-lightning/pull/5620))
- Fixed `Metric`'s `state_dict` not included when child modules ([#5614](https://github.com/PyTorchLightning/pytorch-lightning/pull/5614))
- Fixed Neptune logger creating multiple experiments when GPUs > 1 ([#3256](https://github.com/PyTorchLightning/pytorch-lightning/pull/3256))
- Fixed duplicate logs appearing in console when using the python logging module ([#5509](https://github.com/PyTorchLightning/pytorch-lightning/pull/5509))
- Fixed tensor printing in `trainer.test()` ([#5138](https://github.com/PyTorchLightning/pytorch-lightning/pull/5138))
- Fixed not using dataloader when `hparams` present ([#4559](https://github.com/PyTorchLightning/pytorch-lightning/pull/4559))


## [1.1.5] - 2021-01-19

### Fixed

- Fixed a visual bug in the progress bar display initialization ([#4579](https://github.com/PyTorchLightning/pytorch-lightning/pull/4579))
- Fixed logging `on_train_batch_end` in a callback with multiple optimizers ([#5521](https://github.com/PyTorchLightning/pytorch-lightning/pull/5521))
- Fixed `reinit_scheduler_properties` with correct optimizer ([#5519](https://github.com/PyTorchLightning/pytorch-lightning/pull/5519))
- Fixed `val_check_interval` with `fast_dev_run` ([#5540](https://github.com/PyTorchLightning/pytorch-lightning/pull/5540))


## [1.1.4] - 2021-01-12

### Added

- Add automatic optimization property setter to lightning module ([#5169](https://github.com/PyTorchLightning/pytorch-lightning/pull/5169))

### Changed

- Changed deprecated `enable_pl_optimizer=True` ([#5244](https://github.com/PyTorchLightning/pytorch-lightning/pull/5244))

### Fixed

- Fixed `transfer_batch_to_device` for DDP with `len(devices_ids) == 1` ([#5195](https://github.com/PyTorchLightning/pytorch-lightning/pull/5195))
- Logging only on `not should_accumulate()` during training ([#5417](https://github.com/PyTorchLightning/pytorch-lightning/pull/5417))
- Resolve interpolation bug with Hydra ([#5406](https://github.com/PyTorchLightning/pytorch-lightning/pull/5406))
- Check environ before selecting a seed to prevent warning message ([#4743](https://github.com/PyTorchLightning/pytorch-lightning/pull/4743))
- Fixed signature mismatch in `model_to_device` of `DDPCPUHPCAccelerator` ([#5505](https://github.com/PyTorchLightning/pytorch-lightning/pull/5505))

## [1.1.3] - 2021-01-05

### Added

- Added a check for optimizer attached to `lr_scheduler` ([#5338](https://github.com/PyTorchLightning/pytorch-lightning/pull/5338))
- Added support for passing non-existing filepaths to `resume_from_checkpoint` ([#4402](https://github.com/PyTorchLightning/pytorch-lightning/pull/4402))

### Changed

- Skip restore from `resume_from_checkpoint` while `testing` ([#5161](https://github.com/PyTorchLightning/pytorch-lightning/pull/5161))
- Allowed `log_momentum` for adaptive optimizers in `LearningRateMonitor` ([#5333](https://github.com/PyTorchLightning/pytorch-lightning/pull/5333))
- Disabled checkpointing, earlystopping and logging with `fast_dev_run` ([#5277](https://github.com/PyTorchLightning/pytorch-lightning/pull/5277))
- Distributed group defaults to `WORLD` if `None` ([#5125](https://github.com/PyTorchLightning/pytorch-lightning/pull/5125))

### Fixed

- Fixed `trainer.test` returning non-test metrics ([#5214](https://github.com/PyTorchLightning/pytorch-lightning/pull/5214))
- Fixed metric state reset ([#5273](https://github.com/PyTorchLightning/pytorch-lightning/pull/5273))
- Fixed `--num-nodes` on `DDPSequentialPlugin` ([#5327](https://github.com/PyTorchLightning/pytorch-lightning/pull/5327))
- Fixed invalid value for `weights_summary` ([#5296](https://github.com/PyTorchLightning/pytorch-lightning/pull/5296))
- Fixed `Trainer.test` not using the latest `best_model_path` ([#5161](https://github.com/PyTorchLightning/pytorch-lightning/pull/5161))
- Fixed existence check for hparams not using underlying filesystem ([#5250](https://github.com/PyTorchLightning/pytorch-lightning/pull/5250))
- Fixed `LightningOptimizer` AMP bug ([#5191](https://github.com/PyTorchLightning/pytorch-lightning/pull/5191))
- Fixed casted key to string in `_flatten_dict` ([#5354](https://github.com/PyTorchLightning/pytorch-lightning/pull/5354))


## [1.1.2] - 2020-12-23

### Added

- Support number for logging with `sync_dist=True` ([#5080](https://github.com/PyTorchLightning/pytorch-lightning/pull/5080))
- Added offset logging step when resuming for Wandb logger ([#5050](https://github.com/PyTorchLightning/pytorch-lightning/pull/5050))

### Removed

- `enable_pl_optimizer=False` by default to temporarily fix AMP issues ([#5163](https://github.com/PyTorchLightning/pytorch-lightning/pull/5163))

### Fixed

- Metric reduction with Logging ([#5150](https://github.com/PyTorchLightning/pytorch-lightning/pull/5150))
- Remove nan loss in manual optimization ([#5121](https://github.com/PyTorchLightning/pytorch-lightning/pull/5121))
- Un-balanced logging properly supported ([#5119](https://github.com/PyTorchLightning/pytorch-lightning/pull/5119))
- Fix hanging in DDP HPC accelerators ([#5157](https://github.com/PyTorchLightning/pytorch-lightning/pull/5157))
- Fix reset `TensorRunningAccum` ([#5106](https://github.com/PyTorchLightning/pytorch-lightning/pull/5106))
- Updated `DALIClassificationLoader` to not use deprecated arguments ([#4925](https://github.com/PyTorchLightning/pytorch-lightning/pull/4925))
- Corrected call to `torch.no_grad` ([#5124](https://github.com/PyTorchLightning/pytorch-lightning/pull/5124))


## [1.1.1] - 2020-12-15

### Added

- Add a notebook example to reach a quick baseline of ~94% accuracy on CIFAR10 using Resnet in Lightning ([#4818](https://github.com/PyTorchLightning/pytorch-lightning/pull/4818))

### Changed

- Simplify accelerator steps ([#5015](https://github.com/PyTorchLightning/pytorch-lightning/pull/5015))
- Refactor load in checkpoint connector ([#4593](https://github.com/PyTorchLightning/pytorch-lightning/pull/4593))
- Fixed the saved filename in `ModelCheckpoint` when it already exists ([#4861](https://github.com/PyTorchLightning/pytorch-lightning/pull/4861))

### Removed

- Drop duplicate metrics ([#5014](https://github.com/PyTorchLightning/pytorch-lightning/pull/5014))
- Remove beta arg from F1 class and functional ([#5076](https://github.com/PyTorchLightning/pytorch-lightning/pull/5076))

### Fixed

- Fixed trainer by default `None` in `DDPAccelerator` ([#4915](https://github.com/PyTorchLightning/pytorch-lightning/pull/4915))
- Fixed `LightningOptimizer` to expose optimizer attributes ([#5095](https://github.com/PyTorchLightning/pytorch-lightning/pull/5095))
- Do not warn when the `name` key is used in the `lr_scheduler` dict ([#5057](https://github.com/PyTorchLightning/pytorch-lightning/pull/5057))
- Check if optimizer supports closure ([#4981](https://github.com/PyTorchLightning/pytorch-lightning/pull/4981))
- Add deprecated metric utility functions back to functional (
    [#5067](https://github.com/PyTorchLightning/pytorch-lightning/pull/5067),
    [#5068](https://github.com/PyTorchLightning/pytorch-lightning/pull/5068))
- Allow any input in `to_onnx` and `to_torchscript` ([#4378](https://github.com/PyTorchLightning/pytorch-lightning/pull/4378))
- Fixed `DDPHPCAccelerator` hangs in DDP construction by calling `init_device` ([#5157](https://github.com/PyTorchLightning/pytorch-lightning/pull/5157))


## [1.1.0] - 2020-12-09

### Added

- Added "monitor" key to saved `ModelCheckpoints` ([#4383](https://github.com/PyTorchLightning/pytorch-lightning/pull/4383))
- Added `ConfusionMatrix` class interface ([#4348](https://github.com/PyTorchLightning/pytorch-lightning/pull/4348))
- Added multiclass AUROC metric ([#4236](https://github.com/PyTorchLightning/pytorch-lightning/pull/4236))
- Added global step indexing to the checkpoint name for a better sub-epoch checkpointing experience ([#3807](https://github.com/PyTorchLightning/pytorch-lightning/pull/3807))
- Added optimizer hooks in callbacks ([#4379](https://github.com/PyTorchLightning/pytorch-lightning/pull/4379))
- Added option to log momentum ([#4384](https://github.com/PyTorchLightning/pytorch-lightning/pull/4384))
- Added `current_score` to `ModelCheckpoint.on_save_checkpoint` ([#4721](https://github.com/PyTorchLightning/pytorch-lightning/pull/4721))
- Added logging using `self.log` in train and evaluation for epoch end hooks (
    [#4552](https://github.com/PyTorchLightning/pytorch-lightning/pull/4552),
    [#4495](https://github.com/PyTorchLightning/pytorch-lightning/pull/4495),
    [#4439](https://github.com/PyTorchLightning/pytorch-lightning/pull/4439),
    [#4684](https://github.com/PyTorchLightning/pytorch-lightning/pull/4684),
    [#4913](https://github.com/PyTorchLightning/pytorch-lightning/pull/4913))
- Added ability for DDP plugin to modify optimizer state saving ([#4675](https://github.com/PyTorchLightning/pytorch-lightning/pull/4675))
- Added `prefix` argument in loggers ([#4557](https://github.com/PyTorchLightning/pytorch-lightning/pull/4557))
- Added printing of total num of params, trainable and non-trainable params in ModelSummary ([#4521](https://github.com/PyTorchLightning/pytorch-lightning/pull/4521))
- Added `PrecisionRecallCurve, ROC, AveragePrecision` class metric ([#4549](https://github.com/PyTorchLightning/pytorch-lightning/pull/4549))
- Added custom `Apex` and `NativeAMP` as `Precision plugins` ([#4355](https://github.com/PyTorchLightning/pytorch-lightning/pull/4355))
- Added `DALI MNIST` example ([#3721](https://github.com/PyTorchLightning/pytorch-lightning/pull/3721))
- Added `sharded plugin` for DDP for multi-gpu training memory optimizations (
    [#4639](https://github.com/PyTorchLightning/pytorch-lightning/pull/4639),
    [#4686](https://github.com/PyTorchLightning/pytorch-lightning/pull/4686),
    [#4737](https://github.com/PyTorchLightning/pytorch-lightning/pull/4737),
    [#4773](https://github.com/PyTorchLightning/pytorch-lightning/pull/4773))
- Added `experiment_id` to the NeptuneLogger ([#3462](https://github.com/PyTorchLightning/pytorch-lightning/pull/3462))
- Added `Pytorch Geometric` integration example with Lightning ([#4568](https://github.com/PyTorchLightning/pytorch-lightning/pull/4568))
- Added `all_gather` method to `LightningModule` which allows gradient based tensor synchronizations for use-cases such as negative sampling. ([#5012](https://github.com/PyTorchLightning/pytorch-lightning/pull/5012))
- Enabled `self.log` in most functions ([#4969](https://github.com/PyTorchLightning/pytorch-lightning/pull/4969))
- Added changeable extension variable for `ModelCheckpoint` ([#4977](https://github.com/PyTorchLightning/pytorch-lightning/pull/4977))


### Changed

- Tuner algorithms will be skipped if `fast_dev_run=True` ([#3903](https://github.com/PyTorchLightning/pytorch-lightning/pull/3903))
- `WandbLogger` does not force wandb `reinit` arg to True anymore and creates a run only when needed ([#4648](https://github.com/PyTorchLightning/pytorch-lightning/pull/4648))
- Changed `automatic_optimization` to be a model attribute ([#4602](https://github.com/PyTorchLightning/pytorch-lightning/pull/4602))
- Changed `Simple Profiler` report to order by percentage time spent + num calls ([#4880](https://github.com/PyTorchLightning/pytorch-lightning/pull/4880))
- Simplify optimization Logic ([#4984](https://github.com/PyTorchLightning/pytorch-lightning/pull/4984))
- Classification metrics overhaul ([#4837](https://github.com/PyTorchLightning/pytorch-lightning/pull/4837))
- Updated `fast_dev_run` to accept integer representing num_batches ([#4629](https://github.com/PyTorchLightning/pytorch-lightning/pull/4629))
- Refactored optimizer ([#4658](https://github.com/PyTorchLightning/pytorch-lightning/pull/4658))


### Deprecated

- Deprecated `prefix` argument in `ModelCheckpoint` ([#4765](https://github.com/PyTorchLightning/pytorch-lightning/pull/4765))
- Deprecated the old way of assigning hyper-parameters through `self.hparams = ...` ([#4813](https://github.com/PyTorchLightning/pytorch-lightning/pull/4813))
- Deprecated `mode='auto'` from `ModelCheckpoint` and `EarlyStopping` ([#4695](https://github.com/PyTorchLightning/pytorch-lightning/pull/4695))

### Removed

- Removed `reorder` parameter of the `auc` metric ([#5004](https://github.com/PyTorchLightning/pytorch-lightning/pull/5004))
- Removed `multiclass_roc` and `multiclass_precision_recall_curve`, use `roc` and `precision_recall_curve` instead ([#4549](https://github.com/PyTorchLightning/pytorch-lightning/pull/4549))

### Fixed

- Added feature to move tensors to CPU before saving ([#4309](https://github.com/PyTorchLightning/pytorch-lightning/pull/4309))
- Fixed `LoggerConnector` to have logged metrics on root device in DP ([#4138](https://github.com/PyTorchLightning/pytorch-lightning/pull/4138))
- Auto convert tensors to contiguous format when `gather_all` ([#4907](https://github.com/PyTorchLightning/pytorch-lightning/pull/4907))
- Fixed `PYTHONPATH` for ddp test model ([#4528](https://github.com/PyTorchLightning/pytorch-lightning/pull/4528))
- Fixed allowing logger to support indexing ([#4595](https://github.com/PyTorchLightning/pytorch-lightning/pull/4595))
- Fixed DDP and manual_optimization ([#4976](https://github.com/PyTorchLightning/pytorch-lightning/pull/4976))


## [1.0.8] - 2020-11-24

### Added

- Added casting to python types for numpy scalars when logging `hparams` ([#4647](https://github.com/PyTorchLightning/pytorch-lightning/pull/4647))
- Added warning when progress bar refresh rate is less than 20 on Google Colab to prevent crashing ([#4654](https://github.com/PyTorchLightning/pytorch-lightning/pull/4654))
- Added `F1` class metric ([#4656](https://github.com/PyTorchLightning/pytorch-lightning/pull/4656))

### Changed

- Consistently use `step=trainer.global_step` in `LearningRateMonitor` independently of `logging_interval` ([#4376](https://github.com/PyTorchLightning/pytorch-lightning/pull/4376))
- Metric states are no longer as default added to `state_dict` ([#4685](https://github.com/PyTorchLightning/pytorch-lightning/pull/4685))
- Renamed class metric `Fbeta` >> `FBeta` ([#4656](https://github.com/PyTorchLightning/pytorch-lightning/pull/4656))
- Model summary: add 1 decimal place ([#4745](https://github.com/PyTorchLightning/pytorch-lightning/pull/4745))
- Do not override `PYTHONWARNINGS` ([#4700](https://github.com/PyTorchLightning/pytorch-lightning/pull/4700))
- Changed `init_ddp_connection` moved from `DDP` to `DDPPlugin` ([#4407](https://github.com/PyTorchLightning/pytorch-lightning/pull/4407))


### Fixed

- Fixed checkpoint `hparams` dict casting when `omegaconf` is available ([#4770](https://github.com/PyTorchLightning/pytorch-lightning/pull/4770))
- Fixed incomplete progress bars when total batches not divisible by refresh rate ([#4577](https://github.com/PyTorchLightning/pytorch-lightning/pull/4577))
- Updated SSIM metric ([#4566](https://github.com/PyTorchLightning/pytorch-lightning/pull/4566))
- Fixed batch_arg_name - add `batch_arg_name` to all calls to `_adjust_batch_size`bug ([#4812](https://github.com/PyTorchLightning/pytorch-lightning/pull/4812))
- Fixed `torchtext` data to GPU ([#4785](https://github.com/PyTorchLightning/pytorch-lightning/pull/4785))
- Fixed a crash bug in MLFlow logger ([#4716](https://github.com/PyTorchLightning/pytorch-lightning/pull/4716))

## [1.0.7] - 2020-11-17

### Added

- Added lambda closure to `manual_optimizer_step` ([#4618](https://github.com/PyTorchLightning/pytorch-lightning/pull/4618))

### Changed

- Change Metrics `persistent` default mode to `False` ([#4685](https://github.com/PyTorchLightning/pytorch-lightning/pull/4685))
- LoggerConnector log_metrics will use `total_batch_idx` instead of `global_step` when logging on `training step` ([#4738](https://github.com/PyTorchLightning/pytorch-lightning/pull/4738))


### Fixed

- Prevent crash if `sync_dist=True` on CPU ([#4626](https://github.com/PyTorchLightning/pytorch-lightning/pull/4626))
- Fixed average pbar Metrics ([#4534](https://github.com/PyTorchLightning/pytorch-lightning/pull/4534))
- Fixed `setup` callback hook to correctly pass the LightningModule through ([#4608](https://github.com/PyTorchLightning/pytorch-lightning/pull/4608))
- Allowing decorate model init with saving `hparams` inside ([#4662](https://github.com/PyTorchLightning/pytorch-lightning/pull/4662))
- Fixed `split_idx` set by `LoggerConnector` in `on_trainer_init` to `Trainer`  ([#4697](https://github.com/PyTorchLightning/pytorch-lightning/pull/4697))


## [1.0.6] - 2020-11-11

### Added

- Added metrics aggregation in Horovod and fixed early stopping ([#3775](https://github.com/PyTorchLightning/pytorch-lightning/pull/3775))
- Added `manual_optimizer_step` which work with `AMP Native` and `accumulated_grad_batches` ([#4485](https://github.com/PyTorchLightning/pytorch-lightning/pull/4485))
- Added `persistent(mode)` method to metrics, to enable and disable metric states being added to `state_dict` ([#4482](https://github.com/PyTorchLightning/pytorch-lightning/pull/4482))
- Added congratulations at the end of our notebooks ([#4555](https://github.com/PyTorchLightning/pytorch-lightning/pull/4555))
- Added parameters `move_metrics_to_cpu` in Trainer to disable gpu leak ([#4592](https://github.com/PyTorchLightning/pytorch-lightning/pull/4592))


### Changed

- Changed `fsspec` to tuner ([#4458](https://github.com/PyTorchLightning/pytorch-lightning/pull/4458))
- Unify SLURM/TorchElastic under backend plugin ([#4578](https://github.com/PyTorchLightning/pytorch-lightning/pull/4578),
        [#4580](https://github.com/PyTorchLightning/pytorch-lightning/pull/4580),
        [#4581](https://github.com/PyTorchLightning/pytorch-lightning/pull/4581),
        [#4582](https://github.com/PyTorchLightning/pytorch-lightning/pull/4582),
        [#4583](https://github.com/PyTorchLightning/pytorch-lightning/pull/4583))

### Fixed

- Fixed feature-lack in `hpc_load` ([#4526](https://github.com/PyTorchLightning/pytorch-lightning/pull/4526))
- Fixed metrics states being overridden in DDP mode ([#4482](https://github.com/PyTorchLightning/pytorch-lightning/pull/4482))
- Fixed `lightning_getattr`, `lightning_hasattr` not finding the correct attributes in datamodule ([#4347](https://github.com/PyTorchLightning/pytorch-lightning/pull/4347))
- Fixed automatic optimization AMP by `manual_optimization_step` ([#4485](https://github.com/PyTorchLightning/pytorch-lightning/pull/4485))
- Replace `MisconfigurationException` with warning in `ModelCheckpoint` Callback ([#4560](https://github.com/PyTorchLightning/pytorch-lightning/pull/4560))
- Fixed logged keys in mlflow logger ([#4412](https://github.com/PyTorchLightning/pytorch-lightning/pull/4412))
- Fixed `is_picklable` by catching `AttributeError` ([#4508](https://github.com/PyTorchLightning/pytorch-lightning/pull/4508))
- Fixed multi test dataloaders dict `AttributeError` error ([#4480](https://github.com/PyTorchLightning/pytorch-lightning/pull/4480))
- Fixed show progress bar only for `progress_rank 0` on `DDP_SLURM` ([#4437](https://github.com/PyTorchLightning/pytorch-lightning/pull/4437))

## [1.0.5] - 2020-11-03

### Added

- Added PyTorch 1.7 Stable support ([#3821](https://github.com/PyTorchLightning/pytorch-lightning/pull/3821))
- Added timeout for `tpu_device_exists` to ensure process does not hang indefinitely ([#4340](https://github.com/PyTorchLightning/pytorch-lightning/pull/4340))

### Changed

- W&B log in sync with `Trainer` step ([#4405](https://github.com/PyTorchLightning/pytorch-lightning/pull/4405))
- Hook `on_after_backward` is called only when `optimizer_step` is being called ([#4439](https://github.com/PyTorchLightning/pytorch-lightning/pull/4439))
- Moved `track_and_norm_grad` into `training loop` and called only when `optimizer_step` is being called ([#4439](https://github.com/PyTorchLightning/pytorch-lightning/pull/4439))
- Changed type checker with explicit cast of `ref_model` object ([#4457](https://github.com/PyTorchLightning/pytorch-lightning/pull/4457))
- Changed `distributed_backend` -> `accelerator` ([#4429](https://github.com/PyTorchLightning/pytorch-lightning/pull/4429))

### Deprecated

- Deprecated passing `ModelCheckpoint` instance to `checkpoint_callback` Trainer argument ([#4336](https://github.com/PyTorchLightning/pytorch-lightning/pull/4336))

### Fixed

- Disable saving checkpoints if not trained ([#4372](https://github.com/PyTorchLightning/pytorch-lightning/pull/4372))
- Fixed error using `auto_select_gpus=True` with `gpus=-1` ([#4209](https://github.com/PyTorchLightning/pytorch-lightning/pull/4209))
- Disabled training when `limit_train_batches=0` ([#4371](https://github.com/PyTorchLightning/pytorch-lightning/pull/4371))
- Fixed that metrics do not store computational graph for all seen data ([#4313](https://github.com/PyTorchLightning/pytorch-lightning/pull/4313))
- Fixed AMP unscale for `on_after_backward` ([#4439](https://github.com/PyTorchLightning/pytorch-lightning/pull/4439))
- Fixed TorchScript export when module includes Metrics ([#4428](https://github.com/PyTorchLightning/pytorch-lightning/pull/4428))
- Fixed TorchScript trace method's data to device and docstring ([#4360](https://github.com/PyTorchLightning/pytorch-lightning/pull/4360))
- Fixed CSV logger warning ([#4419](https://github.com/PyTorchLightning/pytorch-lightning/pull/4419))
- Fixed skip DDP parameter sync ([#4301](https://github.com/PyTorchLightning/pytorch-lightning/pull/4301))
- Fixed `WandbLogger` _sanitize_callable function ([#4422](https://github.com/PyTorchLightning/pytorch-lightning/pull/4422))
- Fixed `AMP Native` `_unscale` gradient ([#4441](https://github.com/PyTorchLightning/pytorch-lightning/pull/4441))


## [1.0.4] - 2020-10-27

### Added

- Added `dirpath` and `filename` parameter in `ModelCheckpoint` ([#4213](https://github.com/PyTorchLightning/pytorch-lightning/pull/4213))
- Added plugins docs and DDPPlugin to customize ddp across all accelerators ([#4258](https://github.com/PyTorchLightning/pytorch-lightning/pull/4285))
- Added `strict` option to the scheduler dictionary ([#3586](https://github.com/PyTorchLightning/pytorch-lightning/pull/3586))
- Added `fsspec` support for profilers ([#4162](https://github.com/PyTorchLightning/pytorch-lightning/pull/4162))
- Added autogenerated helptext to `Trainer.add_argparse_args` ([#4344](https://github.com/PyTorchLightning/pytorch-lightning/pull/4344))
- Added support for string values in `Trainer`'s `profiler` parameter ([#3656](https://github.com/PyTorchLightning/pytorch-lightning/pull/3656))
- Added `optimizer_closure` to `optimizer.step` when supported ([#4190](https://github.com/PyTorchLightning/pytorch-lightning/pull/4190))
- Added unification of regression metrics ([#4166](https://github.com/PyTorchLightning/pytorch-lightning/pull/4166))
- Added checkpoint load from Bytes ([#4314](https://github.com/PyTorchLightning/pytorch-lightning/pull/4314))

### Changed

- Improved error messages for invalid `configure_optimizers` returns ([#3587](https://github.com/PyTorchLightning/pytorch-lightning/pull/3587))
- Allow changing the logged step value in `validation_step` ([#4130](https://github.com/PyTorchLightning/pytorch-lightning/pull/4130))
- Allow setting `replace_sampler_ddp=True` with a distributed sampler already added ([#4273](https://github.com/PyTorchLightning/pytorch-lightning/pull/4273))
- Fixed santized parameters for `WandbLogger.log_hyperparams` ([#4320](https://github.com/PyTorchLightning/pytorch-lightning/pull/4320))

### Deprecated

- Deprecated `filepath` in `ModelCheckpoint` ([#4213](https://github.com/PyTorchLightning/pytorch-lightning/pull/4213))
- Deprecated `reorder` parameter of the `auc` metric ([#4237](https://github.com/PyTorchLightning/pytorch-lightning/pull/4237))
- Deprecated bool values in `Trainer`'s `profiler` parameter ([#3656](https://github.com/PyTorchLightning/pytorch-lightning/pull/3656))

### Fixed

- Fixed setting device ids in DDP ([#4297](https://github.com/PyTorchLightning/pytorch-lightning/pull/4297))
- Fixed synchronization of best model path in `ddp_accelerator` ([#4323](https://github.com/PyTorchLightning/pytorch-lightning/pull/4323))
- Fixed `WandbLogger` not uploading checkpoint artifacts at the end of training ([#4341](https://github.com/PyTorchLightning/pytorch-lightning/pull/4341))
- Fixed `FBeta` computation ([#4183](https://github.com/PyTorchLightning/pytorch-lightning/pull/4183))
- Fixed `accumulation across batches` has completed `before breaking training loop` ([#4278](https://github.com/PyTorchLightning/pytorch-lightning/pull/4278))
- Fixed `ModelCheckpoint` don't increase current_epoch and global_step when not training ([#4291](https://github.com/PyTorchLightning/pytorch-lightning/pull/4291))
- Fixed `COMET_EXPERIMENT_KEY` environment variable usage in comet logger ([#4230](https://github.com/PyTorchLightning/pytorch-lightning/pull/4230))

## [1.0.3] - 2020-10-20

### Added

- Added persistent flag to `Metric.add_state` ([#4195](https://github.com/PyTorchLightning/pytorch-lightning/pull/4195))

### Changed

- Used `checkpoint_connector.hpc_save` in SLURM ([#4217](https://github.com/PyTorchLightning/pytorch-lightning/pull/4217))
- Moved base req. to root ([#4219](https://github.com/PyTorchLightning/pytorch-lightning/pull/4219))

### Fixed

- Fixed `hparams` assign in init ([#4189](https://github.com/PyTorchLightning/pytorch-lightning/pull/4189))
- Fixed overwrite check for model hooks ([#4010](https://github.com/PyTorchLightning/pytorch-lightning/pull/4010))


## [1.0.2] - 2020-10-15

### Added

- Added trace functionality to the function `to_torchscript` ([#4142](https://github.com/PyTorchLightning/pytorch-lightning/pull/4142))

### Changed

- Called `on_load_checkpoint` before loading `state_dict` ([#4057](https://github.com/PyTorchLightning/pytorch-lightning/pull/4057))

### Removed

- Removed duplicate metric vs step log for train loop ([#4173](https://github.com/PyTorchLightning/pytorch-lightning/pull/4173))

### Fixed

- Fixed the `self.log` problem in `validation_step()` ([#4169](https://github.com/PyTorchLightning/pytorch-lightning/pull/4169))
- Fixed `hparams` saving - save the state when `save_hyperparameters()` is called [in `__init__`] ([#4163](https://github.com/PyTorchLightning/pytorch-lightning/pull/4163))
- Fixed runtime failure while exporting `hparams` to yaml ([#4158](https://github.com/PyTorchLightning/pytorch-lightning/pull/4158))


## [1.0.1] - 2020-10-14

### Added

- Added getstate/setstate method for torch.save serialization ([#4127](https://github.com/PyTorchLightning/pytorch-lightning/pull/4127))


## [1.0.0] - 2020-10-13

### Added

- Added Explained Variance Metric + metric fix ([#4013](https://github.com/PyTorchLightning/pytorch-lightning/pull/4013))
- Added Metric <-> Lightning Module integration tests ([#4008](https://github.com/PyTorchLightning/pytorch-lightning/pull/4008))
- Added parsing OS env vars in `Trainer` ([#4022](https://github.com/PyTorchLightning/pytorch-lightning/pull/4022))
- Added classification metrics ([#4043](https://github.com/PyTorchLightning/pytorch-lightning/pull/4043))
- Updated explained variance metric ([#4024](https://github.com/PyTorchLightning/pytorch-lightning/pull/4024))
- Enabled plugins ([#4041](https://github.com/PyTorchLightning/pytorch-lightning/pull/4041))
- Enabled custom clusters ([#4048](https://github.com/PyTorchLightning/pytorch-lightning/pull/4048))
- Enabled passing in custom accelerators ([#4050](https://github.com/PyTorchLightning/pytorch-lightning/pull/4050))
- Added `LightningModule.toggle_optimizer` ([#4058](https://github.com/PyTorchLightning/pytorch-lightning/pull/4058))
- Added `LightningModule.manual_backward` ([#4063](https://github.com/PyTorchLightning/pytorch-lightning/pull/4063))
- Added `output` argument to `*_batch_end` hooks ([#3965](https://github.com/PyTorchLightning/pytorch-lightning/pull/3965),
    [#3966](https://github.com/PyTorchLightning/pytorch-lightning/pull/3966))
- Added `output` argument to `*_epoch_end` hooks ([#3967](https://github.com/PyTorchLightning/pytorch-lightning/pull/3967))

### Changed

- Integrated metrics API with self.log ([#3961](https://github.com/PyTorchLightning/pytorch-lightning/pull/3961))
- Decoupled Apex ([#4052](https://github.com/PyTorchLightning/pytorch-lightning/pull/4052),
        [#4054](https://github.com/PyTorchLightning/pytorch-lightning/pull/4054),
        [#4055](https://github.com/PyTorchLightning/pytorch-lightning/pull/4055),
        [#4056](https://github.com/PyTorchLightning/pytorch-lightning/pull/4056),
        [#4058](https://github.com/PyTorchLightning/pytorch-lightning/pull/4058),
        [#4060](https://github.com/PyTorchLightning/pytorch-lightning/pull/4060),
        [#4061](https://github.com/PyTorchLightning/pytorch-lightning/pull/4061),
        [#4062](https://github.com/PyTorchLightning/pytorch-lightning/pull/4062),
        [#4063](https://github.com/PyTorchLightning/pytorch-lightning/pull/4063),
        [#4064](https://github.com/PyTorchLightning/pytorch-lightning/pull/4064),
        [#4065](https://github.com/PyTorchLightning/pytorch-lightning/pull/4065))
- Renamed all backends to `Accelerator` ([#4066](https://github.com/PyTorchLightning/pytorch-lightning/pull/4066))
- Enabled manual returns ([#4089](https://github.com/PyTorchLightning/pytorch-lightning/pull/4089))

### Removed

- Removed support for EvalResult and TrainResult ([#3968](https://github.com/PyTorchLightning/pytorch-lightning/pull/3968))
- Removed deprecated trainer flags: `overfit_pct`, `log_save_interval`, `row_log_interval` ([#3969](https://github.com/PyTorchLightning/pytorch-lightning/pull/3969))
- Removed deprecated early_stop_callback ([#3982](https://github.com/PyTorchLightning/pytorch-lightning/pull/3982))
- Removed deprecated model hooks ([#3980](https://github.com/PyTorchLightning/pytorch-lightning/pull/3980))
- Removed deprecated callbacks ([#3979](https://github.com/PyTorchLightning/pytorch-lightning/pull/3979))
- Removed `trainer` argument in `LightningModule.backward` [#4056](https://github.com/PyTorchLightning/pytorch-lightning/pull/4056))

### Fixed

- Fixed `current_epoch` property update to reflect true epoch number inside `LightningDataModule`, when `reload_dataloaders_every_epoch=True`. ([#3974](https://github.com/PyTorchLightning/pytorch-lightning/pull/3974))
- Fixed to print scaler value in progress bar ([#4053](https://github.com/PyTorchLightning/pytorch-lightning/pull/4053))
- Fixed mismatch between docstring and code regarding when `on_load_checkpoint` hook is called ([#3996](https://github.com/PyTorchLightning/pytorch-lightning/pull/3996))


## [0.10.0] - 2020-10-07

### Added

- Added new Metrics API. ([#3868](https://github.com/PyTorchLightning/pytorch-lightning/pull/3868), [#3921](https://github.com/PyTorchLightning/pytorch-lightning/pull/3921))
- Enable PyTorch 1.7 compatibility ([#3541](https://github.com/PyTorchLightning/pytorch-lightning/pull/3541))
- Added `LightningModule.to_torchscript` to support exporting as `ScriptModule` ([#3258](https://github.com/PyTorchLightning/pytorch-lightning/pull/3258))
- Added warning when dropping unpicklable `hparams` ([#2874](https://github.com/PyTorchLightning/pytorch-lightning/pull/2874))
- Added EMB similarity ([#3349](https://github.com/PyTorchLightning/pytorch-lightning/pull/3349))
- Added `ModelCheckpoint.to_yaml` method ([#3048](https://github.com/PyTorchLightning/pytorch-lightning/pull/3048))
- Allow `ModelCheckpoint` monitor to be `None`, meaning it will always save ([#3630](https://github.com/PyTorchLightning/pytorch-lightning/pull/3630))
- Disabled optimizers setup during testing ([#3059](https://github.com/PyTorchLightning/pytorch-lightning/pull/3059))
- Added support for datamodules to save and load checkpoints when training ([#3563](https://github.com/PyTorchLightning/pytorch-lightning/pull/3563))
- Added support for datamodule in learning rate finder ([#3425](https://github.com/PyTorchLightning/pytorch-lightning/pull/3425))
- Added gradient clip test for native AMP ([#3754](https://github.com/PyTorchLightning/pytorch-lightning/pull/3754))
- Added dist lib to enable syncing anything across devices ([#3762](https://github.com/PyTorchLightning/pytorch-lightning/pull/3762))
- Added `broadcast` to `TPUBackend` ([#3814](https://github.com/PyTorchLightning/pytorch-lightning/pull/3814))
- Added `XLADeviceUtils` class to check XLA device type ([#3274](https://github.com/PyTorchLightning/pytorch-lightning/pull/3274))

### Changed

- Refactored accelerator backends:
   * moved TPU `xxx_step` to backend ([#3118](https://github.com/PyTorchLightning/pytorch-lightning/pull/3118))
   * refactored DDP backend `forward` ([#3119](https://github.com/PyTorchLightning/pytorch-lightning/pull/3119))
   * refactored GPU backend `__step` ([#3120](https://github.com/PyTorchLightning/pytorch-lightning/pull/3120))
   * refactored Horovod backend ([#3121](https://github.com/PyTorchLightning/pytorch-lightning/pull/3121),
        [#3122](https://github.com/PyTorchLightning/pytorch-lightning/pull/3122))
   * remove obscure forward call in eval + CPU backend `___step` ([#3123](https://github.com/PyTorchLightning/pytorch-lightning/pull/3123))
   * reduced all simplified forward ([#3126](https://github.com/PyTorchLightning/pytorch-lightning/pull/3126))
   * added hook base method ([#3127](https://github.com/PyTorchLightning/pytorch-lightning/pull/3127))
   * refactor eval loop to use hooks - use `test_mode` for if so we can split later ([#3129](https://github.com/PyTorchLightning/pytorch-lightning/pull/3129))
   * moved `___step_end` hooks ([#3130](https://github.com/PyTorchLightning/pytorch-lightning/pull/3130))
   * training forward refactor ([#3134](https://github.com/PyTorchLightning/pytorch-lightning/pull/3134))
   * training AMP scaling refactor ([#3135](https://github.com/PyTorchLightning/pytorch-lightning/pull/3135))
   * eval step scaling factor ([#3136](https://github.com/PyTorchLightning/pytorch-lightning/pull/3136))
   * add eval loop object to streamline eval loop ([#3138](https://github.com/PyTorchLightning/pytorch-lightning/pull/3138))
   * refactored dataloader process hook ([#3139](https://github.com/PyTorchLightning/pytorch-lightning/pull/3139))
   * refactored inner eval loop ([#3141](https://github.com/PyTorchLightning/pytorch-lightning/pull/3141))
   * final inner eval loop hooks ([#3154](https://github.com/PyTorchLightning/pytorch-lightning/pull/3154))
   * clean up hooks in `run_evaluation` ([#3156](https://github.com/PyTorchLightning/pytorch-lightning/pull/3156))
   * clean up data reset ([#3161](https://github.com/PyTorchLightning/pytorch-lightning/pull/3161))
   * expand eval loop out ([#3165](https://github.com/PyTorchLightning/pytorch-lightning/pull/3165))
   * moved hooks around in eval loop ([#3195](https://github.com/PyTorchLightning/pytorch-lightning/pull/3195))
   * remove `_evaluate` fx ([#3197](https://github.com/PyTorchLightning/pytorch-lightning/pull/3197))
   * `Trainer.fit` hook clean up ([#3198](https://github.com/PyTorchLightning/pytorch-lightning/pull/3198))
   * DDPs train hooks ([#3203](https://github.com/PyTorchLightning/pytorch-lightning/pull/3203))
   * refactor DDP backend ([#3204](https://github.com/PyTorchLightning/pytorch-lightning/pull/3204),
        [#3207](https://github.com/PyTorchLightning/pytorch-lightning/pull/3207),
        [#3208](https://github.com/PyTorchLightning/pytorch-lightning/pull/3208),
        [#3209](https://github.com/PyTorchLightning/pytorch-lightning/pull/3209),
        [#3210](https://github.com/PyTorchLightning/pytorch-lightning/pull/3210))
   * reduced accelerator selection ([#3211](https://github.com/PyTorchLightning/pytorch-lightning/pull/3211))
   * group prepare data hook ([#3212](https://github.com/PyTorchLightning/pytorch-lightning/pull/3212))
   * added data connector ([#3285](https://github.com/PyTorchLightning/pytorch-lightning/pull/3285))
   * modular is_overridden ([#3290](https://github.com/PyTorchLightning/pytorch-lightning/pull/3290))
   * adding `Trainer.tune()` ([#3293](https://github.com/PyTorchLightning/pytorch-lightning/pull/3293))
   * move `run_pretrain_routine` -> `setup_training` ([#3294](https://github.com/PyTorchLightning/pytorch-lightning/pull/3294))
   * move train outside of setup training ([#3297](https://github.com/PyTorchLightning/pytorch-lightning/pull/3297))
   * move `prepare_data` to data connector ([#3307](https://github.com/PyTorchLightning/pytorch-lightning/pull/3307))
   * moved accelerator router ([#3309](https://github.com/PyTorchLightning/pytorch-lightning/pull/3309))
   * train loop refactor - moving train loop to own object ([#3310](https://github.com/PyTorchLightning/pytorch-lightning/pull/3310),
        [#3312](https://github.com/PyTorchLightning/pytorch-lightning/pull/3312),
        [#3313](https://github.com/PyTorchLightning/pytorch-lightning/pull/3313),
        [#3314](https://github.com/PyTorchLightning/pytorch-lightning/pull/3314))
   * duplicate data interface definition up into DataHooks class ([#3344](https://github.com/PyTorchLightning/pytorch-lightning/pull/3344))
   * inner train loop ([#3359](https://github.com/PyTorchLightning/pytorch-lightning/pull/3359),
        [#3361](https://github.com/PyTorchLightning/pytorch-lightning/pull/3361),
        [#3362](https://github.com/PyTorchLightning/pytorch-lightning/pull/3362),
        [#3363](https://github.com/PyTorchLightning/pytorch-lightning/pull/3363),
        [#3365](https://github.com/PyTorchLightning/pytorch-lightning/pull/3365),
        [#3366](https://github.com/PyTorchLightning/pytorch-lightning/pull/3366),
        [#3367](https://github.com/PyTorchLightning/pytorch-lightning/pull/3367),
        [#3368](https://github.com/PyTorchLightning/pytorch-lightning/pull/3368),
        [#3369](https://github.com/PyTorchLightning/pytorch-lightning/pull/3369),
        [#3370](https://github.com/PyTorchLightning/pytorch-lightning/pull/3370),
        [#3371](https://github.com/PyTorchLightning/pytorch-lightning/pull/3371),
        [#3372](https://github.com/PyTorchLightning/pytorch-lightning/pull/3372),
        [#3373](https://github.com/PyTorchLightning/pytorch-lightning/pull/3373),
        [#3374](https://github.com/PyTorchLightning/pytorch-lightning/pull/3374),
        [#3375](https://github.com/PyTorchLightning/pytorch-lightning/pull/3375),
        [#3376](https://github.com/PyTorchLightning/pytorch-lightning/pull/3376),
        [#3385](https://github.com/PyTorchLightning/pytorch-lightning/pull/3385),
        [#3388](https://github.com/PyTorchLightning/pytorch-lightning/pull/3388),
        [#3397](https://github.com/PyTorchLightning/pytorch-lightning/pull/3397))
   * all logging related calls in a connector ([#3395](https://github.com/PyTorchLightning/pytorch-lightning/pull/3395))
   * device parser ([#3400](https://github.com/PyTorchLightning/pytorch-lightning/pull/3400),
        [#3405](https://github.com/PyTorchLightning/pytorch-lightning/pull/3405))
   * added model connector ([#3407](https://github.com/PyTorchLightning/pytorch-lightning/pull/3407))
   * moved eval loop logging to loggers ([#3408](https://github.com/PyTorchLightning/pytorch-lightning/pull/3408))
   * moved eval loop (#3412[#3408](https://github.com/PyTorchLightning/pytorch-lightning/pull/3408))
   * trainer/separate argparse ([#3421](https://github.com/PyTorchLightning/pytorch-lightning/pull/3421),
        [#3428](https://github.com/PyTorchLightning/pytorch-lightning/pull/3428),
        [#3432](https://github.com/PyTorchLightning/pytorch-lightning/pull/3432))
   * move `lr_finder` ([#3434](https://github.com/PyTorchLightning/pytorch-lightning/pull/3434))
   * organize args (#[#3435](https://github.com/PyTorchLightning/pytorch-lightning/pull/3435),
        [#3442](https://github.com/PyTorchLightning/pytorch-lightning/pull/3442),
        [#3447](https://github.com/PyTorchLightning/pytorch-lightning/pull/3447),
        [#3448](https://github.com/PyTorchLightning/pytorch-lightning/pull/3448),
        [#3449](https://github.com/PyTorchLightning/pytorch-lightning/pull/3449),
        [#3456](https://github.com/PyTorchLightning/pytorch-lightning/pull/3456))
   * move specific accelerator code ([#3457](https://github.com/PyTorchLightning/pytorch-lightning/pull/3457))
   * group connectors ([#3472](https://github.com/PyTorchLightning/pytorch-lightning/pull/3472))
   * accelerator connector methods x/n ([#3469](https://github.com/PyTorchLightning/pytorch-lightning/pull/3469),
        [#3470](https://github.com/PyTorchLightning/pytorch-lightning/pull/3470),
        [#3474](https://github.com/PyTorchLightning/pytorch-lightning/pull/3474))
   * merge backends x/n ([#3476](https://github.com/PyTorchLightning/pytorch-lightning/pull/3476),
        [#3477](https://github.com/PyTorchLightning/pytorch-lightning/pull/3477),
        [#3478](https://github.com/PyTorchLightning/pytorch-lightning/pull/3478),
        [#3480](https://github.com/PyTorchLightning/pytorch-lightning/pull/3480),
        [#3482](https://github.com/PyTorchLightning/pytorch-lightning/pull/3482))
   * apex plugin ([#3502](https://github.com/PyTorchLightning/pytorch-lightning/pull/3502))
   * precision plugins ([#3504](https://github.com/PyTorchLightning/pytorch-lightning/pull/3504))
   * Result - make monitor default to `checkpoint_on` to simplify ([#3571](https://github.com/PyTorchLightning/pytorch-lightning/pull/3571))
   * reference to the Trainer on the `LightningDataModule` ([#3684](https://github.com/PyTorchLightning/pytorch-lightning/pull/3684))
   * add `.log` to lightning module ([#3686](https://github.com/PyTorchLightning/pytorch-lightning/pull/3686),
        [#3699](https://github.com/PyTorchLightning/pytorch-lightning/pull/3699),
        [#3701](https://github.com/PyTorchLightning/pytorch-lightning/pull/3701),
        [#3704](https://github.com/PyTorchLightning/pytorch-lightning/pull/3704),
        [#3715](https://github.com/PyTorchLightning/pytorch-lightning/pull/3715))
   * enable tracking original metric when step and epoch are both true ([#3685](https://github.com/PyTorchLightning/pytorch-lightning/pull/3685))
   * deprecated results obj, added support for simpler comms ([#3681](https://github.com/PyTorchLightning/pytorch-lightning/pull/3681))
   * move backends back to individual files ([#3712](https://github.com/PyTorchLightning/pytorch-lightning/pull/3712))
   * fixes logging for eval steps ([#3763](https://github.com/PyTorchLightning/pytorch-lightning/pull/3763))
   * decoupled DDP, DDP spawn ([#3733](https://github.com/PyTorchLightning/pytorch-lightning/pull/3733),
        [#3766](https://github.com/PyTorchLightning/pytorch-lightning/pull/3766),
        [#3767](https://github.com/PyTorchLightning/pytorch-lightning/pull/3767),
        [#3774](https://github.com/PyTorchLightning/pytorch-lightning/pull/3774),
        [#3802](https://github.com/PyTorchLightning/pytorch-lightning/pull/3802),
        [#3806](https://github.com/PyTorchLightning/pytorch-lightning/pull/3806),
        [#3817](https://github.com/PyTorchLightning/pytorch-lightning/pull/3817),
        [#3819](https://github.com/PyTorchLightning/pytorch-lightning/pull/3819),
        [#3927](https://github.com/PyTorchLightning/pytorch-lightning/pull/3927))
   * remove weight loading hack for ddp_cpu ([#3808](https://github.com/PyTorchLightning/pytorch-lightning/pull/3808))
   * separate `torchelastic` from DDP ([#3810](https://github.com/PyTorchLightning/pytorch-lightning/pull/3810))
   * separate SLURM from DDP ([#3809](https://github.com/PyTorchLightning/pytorch-lightning/pull/3809))
   * decoupled DDP2 ([#3816](https://github.com/PyTorchLightning/pytorch-lightning/pull/3816))
   * bug fix with logging val epoch end + monitor ([#3812](https://github.com/PyTorchLightning/pytorch-lightning/pull/3812))
   * callback system and init DDP ([#3836](https://github.com/PyTorchLightning/pytorch-lightning/pull/3836))
   * adding compute environments ([#3837](https://github.com/PyTorchLightning/pytorch-lightning/pull/3837), [#3842](https://github.com/PyTorchLightning/pytorch-lightning/pull/3842))
   * epoch can now log independently ([#3843](https://github.com/PyTorchLightning/pytorch-lightning/pull/3843))
   * test selecting the correct backend. temp backends while slurm and TorchElastic are decoupled ([#3848](https://github.com/PyTorchLightning/pytorch-lightning/pull/3848))
   * fixed `init_slurm_connection` causing hostname errors ([#3856](https://github.com/PyTorchLightning/pytorch-lightning/pull/3856))
   * moves init apex from LM to apex connector ([#3923](https://github.com/PyTorchLightning/pytorch-lightning/pull/3923))
   * moves sync bn to each backend ([#3925](https://github.com/PyTorchLightning/pytorch-lightning/pull/3925))
   * moves configure ddp to each backend ([#3924](https://github.com/PyTorchLightning/pytorch-lightning/pull/3924))
- Deprecation warning ([#3844](https://github.com/PyTorchLightning/pytorch-lightning/pull/3844))
- Changed `LearningRateLogger` to `LearningRateMonitor` ([#3251](https://github.com/PyTorchLightning/pytorch-lightning/pull/3251))
- Used `fsspec` instead of `gfile` for all IO ([#3320](https://github.com/PyTorchLightning/pytorch-lightning/pull/3320))
    * Swaped `torch.load` for `fsspec` load in DDP spawn backend ([#3787](https://github.com/PyTorchLightning/pytorch-lightning/pull/3787))
    * Swaped `torch.load` for `fsspec` load in cloud_io loading ([#3692](https://github.com/PyTorchLightning/pytorch-lightning/pull/3692))
    * Added support for `to_disk()` to use remote filepaths with `fsspec` ([#3930](https://github.com/PyTorchLightning/pytorch-lightning/pull/3930))
    * Updated model_checkpoint's to_yaml to use `fsspec` open ([#3801](https://github.com/PyTorchLightning/pytorch-lightning/pull/3801))
    * Fixed `fsspec` is inconsistent when doing `fs.ls` ([#3805](https://github.com/PyTorchLightning/pytorch-lightning/pull/3805))
- Refactor `GPUStatsMonitor` to improve training speed ([#3257](https://github.com/PyTorchLightning/pytorch-lightning/pull/3257))
- Changed IoU score behavior for classes absent in target and pred ([#3098](https://github.com/PyTorchLightning/pytorch-lightning/pull/3098))
- Changed IoU `remove_bg` bool to `ignore_index` optional int ([#3098](https://github.com/PyTorchLightning/pytorch-lightning/pull/3098))
- Changed defaults of `save_top_k` and `save_last` to `None` in ModelCheckpoint ([#3680](https://github.com/PyTorchLightning/pytorch-lightning/pull/3680))
- `row_log_interval` and `log_save_interval` are now based on training loop's `global_step` instead of epoch-internal batch index ([#3667](https://github.com/PyTorchLightning/pytorch-lightning/pull/3667))
- Silenced some warnings. verified ddp refactors ([#3483](https://github.com/PyTorchLightning/pytorch-lightning/pull/3483))
- Cleaning up stale logger tests ([#3490](https://github.com/PyTorchLightning/pytorch-lightning/pull/3490))
- Allow `ModelCheckpoint` monitor to be `None` ([#3633](https://github.com/PyTorchLightning/pytorch-lightning/pull/3633))
- Enable `None` model checkpoint default ([#3669](https://github.com/PyTorchLightning/pytorch-lightning/pull/3669))
- Skipped `best_model_path` if `checkpoint_callback` is `None` ([#2962](https://github.com/PyTorchLightning/pytorch-lightning/pull/2962))
- Used `raise .. from ..` to explicitly chain exceptions ([#3750](https://github.com/PyTorchLightning/pytorch-lightning/pull/3750))
-  Mocking loggers ([#3596](https://github.com/PyTorchLightning/pytorch-lightning/pull/3596),
    [#3617](https://github.com/PyTorchLightning/pytorch-lightning/pull/3617),
    [#3851](https://github.com/PyTorchLightning/pytorch-lightning/pull/3851),
    [#3859](https://github.com/PyTorchLightning/pytorch-lightning/pull/3859),
    [#3884](https://github.com/PyTorchLightning/pytorch-lightning/pull/3884),
    [#3853](https://github.com/PyTorchLightning/pytorch-lightning/pull/3853),
    [#3910](https://github.com/PyTorchLightning/pytorch-lightning/pull/3910),
    [#3889](https://github.com/PyTorchLightning/pytorch-lightning/pull/3889),
    [#3926](https://github.com/PyTorchLightning/pytorch-lightning/pull/3926))
- Write predictions in LightningModule instead of EvalResult [#3882](https://github.com/PyTorchLightning/pytorch-lightning/pull/3882)

### Deprecated

- Deprecated `TrainResult` and `EvalResult`, use `self.log` and `self.write` from the `LightningModule` to log metrics and write predictions. `training_step` can now only return a scalar (for the loss) or a dictionary with anything you want. ([#3681](https://github.com/PyTorchLightning/pytorch-lightning/pull/3681))
- Deprecate `early_stop_callback` Trainer argument ([#3845](https://github.com/PyTorchLightning/pytorch-lightning/pull/3845))
- Rename Trainer arguments `row_log_interval` >> `log_every_n_steps` and `log_save_interval` >> `flush_logs_every_n_steps` ([#3748](https://github.com/PyTorchLightning/pytorch-lightning/pull/3748))

### Removed

- Removed experimental Metric API ([#3943](https://github.com/PyTorchLightning/pytorch-lightning/pull/3943),
        [#3949](https://github.com/PyTorchLightning/pytorch-lightning/pull/3949),
        [#3946](https://github.com/PyTorchLightning/pytorch-lightning/pull/3946)), listed changes before final removal:
    * Added `EmbeddingSimilarity` metric ([#3349](https://github.com/PyTorchLightning/pytorch-lightning/pull/3349), [#3358](https://github.com/PyTorchLightning/pytorch-lightning/pull/3358))
    * Added hooks to metric module interface ([#2528](https://github.com/PyTorchLightning/pytorch-lightning/pull/2528))
    * Added error when AUROC metric is used for multiclass problems ([#3350](https://github.com/PyTorchLightning/pytorch-lightning/pull/3350))
    * Fixed `ModelCheckpoint` with `save_top_k=-1` option not tracking the best models when a monitor metric is available ([#3735](https://github.com/PyTorchLightning/pytorch-lightning/pull/3735))
    * Fixed counter-intuitive error being thrown in `Accuracy` metric for zero target tensor ([#3764](https://github.com/PyTorchLightning/pytorch-lightning/pull/3764))
    * Fixed aggregation of metrics ([#3517](https://github.com/PyTorchLightning/pytorch-lightning/pull/3517))
    * Fixed Metric aggregation ([#3321](https://github.com/PyTorchLightning/pytorch-lightning/pull/3321))
    * Fixed RMSLE metric ([#3188](https://github.com/PyTorchLightning/pytorch-lightning/pull/3188))
    * Renamed `reduction` to `class_reduction` in classification metrics ([#3322](https://github.com/PyTorchLightning/pytorch-lightning/pull/3322))
    * Changed `class_reduction` similar to sklearn for classification metrics ([#3322](https://github.com/PyTorchLightning/pytorch-lightning/pull/3322))
    * Renaming of precision recall metric ([#3308](https://github.com/PyTorchLightning/pytorch-lightning/pull/3308))

### Fixed

- Fixed `on_train_batch_start` hook to end epoch early ([#3700](https://github.com/PyTorchLightning/pytorch-lightning/pull/3700))
- Fixed `num_sanity_val_steps` is clipped to `limit_val_batches` ([#2917](https://github.com/PyTorchLightning/pytorch-lightning/pull/2917))
- Fixed ONNX model save on GPU ([#3145](https://github.com/PyTorchLightning/pytorch-lightning/pull/3145))
- Fixed `GpuUsageLogger` to work on different platforms ([#3008](https://github.com/PyTorchLightning/pytorch-lightning/pull/3008))
- Fixed auto-scale batch size not dumping `auto_lr_find` parameter ([#3151](https://github.com/PyTorchLightning/pytorch-lightning/pull/3151))
- Fixed `batch_outputs` with optimizer frequencies ([#3229](https://github.com/PyTorchLightning/pytorch-lightning/pull/3229))
- Fixed setting batch size in `LightningModule.datamodule` when using `auto_scale_batch_size` ([#3266](https://github.com/PyTorchLightning/pytorch-lightning/pull/3266))
- Fixed Horovod distributed backend compatibility with native AMP ([#3404](https://github.com/PyTorchLightning/pytorch-lightning/pull/3404))
- Fixed batch size auto scaling exceeding the size of the dataset ([#3271](https://github.com/PyTorchLightning/pytorch-lightning/pull/3271))
- Fixed getting `experiment_id` from MLFlow only once instead of each training loop ([#3394](https://github.com/PyTorchLightning/pytorch-lightning/pull/3394))
- Fixed `overfit_batches` which now correctly disables shuffling for the training loader. ([#3501](https://github.com/PyTorchLightning/pytorch-lightning/pull/3501))
- Fixed gradient norm tracking for `row_log_interval > 1` ([#3489](https://github.com/PyTorchLightning/pytorch-lightning/pull/3489))
- Fixed `ModelCheckpoint` name formatting ([#3164](https://github.com/PyTorchLightning/pytorch-lightning/pull/3163))
- Fixed example implementation of AutoEncoder ([#3190](https://github.com/PyTorchLightning/pytorch-lightning/pull/3190))
- Fixed invalid paths when remote logging with TensorBoard ([#3236](https://github.com/PyTorchLightning/pytorch-lightning/pull/3236))
- Fixed change `t()` to `transpose()` as XLA devices do not support `.t()` on 1-dim tensor ([#3252](https://github.com/PyTorchLightning/pytorch-lightning/pull/3252))
- Fixed (weights only) checkpoints loading without PL ([#3287](https://github.com/PyTorchLightning/pytorch-lightning/pull/3287))
- Fixed `gather_all_tensors` cross GPUs in DDP ([#3319](https://github.com/PyTorchLightning/pytorch-lightning/pull/3319))
- Fixed CometML save dir ([#3419](https://github.com/PyTorchLightning/pytorch-lightning/pull/3419))
- Fixed forward key metrics ([#3467](https://github.com/PyTorchLightning/pytorch-lightning/pull/3467))
- Fixed normalize mode at confusion matrix (replace NaNs with zeros) ([#3465](https://github.com/PyTorchLightning/pytorch-lightning/pull/3465))
- Fixed global step increment in training loop when `training_epoch_end` hook is used ([#3673](https://github.com/PyTorchLightning/pytorch-lightning/pull/3673))
- Fixed dataloader shuffling not getting turned off with `overfit_batches > 0` and `distributed_backend = "ddp"` ([#3534](https://github.com/PyTorchLightning/pytorch-lightning/pull/3534))
- Fixed determinism in `DDPSpawnBackend` when using `seed_everything` in main process ([#3335](https://github.com/PyTorchLightning/pytorch-lightning/pull/3335))
- Fixed `ModelCheckpoint` `period` to actually save every `period` epochs ([#3630](https://github.com/PyTorchLightning/pytorch-lightning/pull/3630))
- Fixed `val_progress_bar` total with `num_sanity_val_steps` ([#3751](https://github.com/PyTorchLightning/pytorch-lightning/pull/3751))
- Fixed Tuner dump: add `current_epoch` to dumped_params ([#3261](https://github.com/PyTorchLightning/pytorch-lightning/pull/3261))
- Fixed `current_epoch` and `global_step` properties mismatch between `Trainer` and `LightningModule` ([#3785](https://github.com/PyTorchLightning/pytorch-lightning/pull/3785))
- Fixed learning rate scheduler for optimizers with internal state ([#3897](https://github.com/PyTorchLightning/pytorch-lightning/pull/3897))
- Fixed `tbptt_reduce_fx` when non-floating tensors are logged ([#3796](https://github.com/PyTorchLightning/pytorch-lightning/pull/3796))
- Fixed model checkpoint frequency ([#3852](https://github.com/PyTorchLightning/pytorch-lightning/pull/3852))
- Fixed logging non-tensor scalar with result breaks subsequent epoch aggregation ([#3855](https://github.com/PyTorchLightning/pytorch-lightning/pull/3855))
- Fixed `TrainerEvaluationLoopMixin` activates `model.train()` at the end ([#3858](https://github.com/PyTorchLightning/pytorch-lightning/pull/3858))
- Fixed `overfit_batches` when using with multiple val/test_dataloaders ([#3857](https://github.com/PyTorchLightning/pytorch-lightning/pull/3857))
- Fixed enables `training_step` to return `None` ([#3862](https://github.com/PyTorchLightning/pytorch-lightning/pull/3862))
- Fixed init nan for checkpointing ([#3863](https://github.com/PyTorchLightning/pytorch-lightning/pull/3863))
- Fixed for `load_from_checkpoint` ([#2776](https://github.com/PyTorchLightning/pytorch-lightning/pull/2776))
- Fixes incorrect `batch_sizes` when Dataloader returns a dict with multiple tensors ([#3668](https://github.com/PyTorchLightning/pytorch-lightning/pull/3668))
- Fixed unexpected signature for `validation_step` ([#3947](https://github.com/PyTorchLightning/pytorch-lightning/pull/3947))

## [0.9.0] - 2020-08-20

### Added

- Added SyncBN for DDP ([#2801](https://github.com/PyTorchLightning/pytorch-lightning/pull/2801),
     [#2838](https://github.com/PyTorchLightning/pytorch-lightning/pull/2838))
- Added basic `CSVLogger` ([#2721](https://github.com/PyTorchLightning/pytorch-lightning/pull/2721))
- Added SSIM metrics ([#2671](https://github.com/PyTorchLightning/pytorch-lightning/pull/2671))
- Added BLEU metrics ([#2535](https://github.com/PyTorchLightning/pytorch-lightning/pull/2535))
- Added support to export a model to ONNX format ([#2596](https://github.com/PyTorchLightning/pytorch-lightning/pull/2596))
- Added support for `Trainer(num_sanity_val_steps=-1)` to check all validation data before training ([#2246](https://github.com/PyTorchLightning/pytorch-lightning/pull/2246))
- Added struct. output:
  * tests for val loop flow ([#2605](https://github.com/PyTorchLightning/pytorch-lightning/pull/2605))
  * `EvalResult` support for train and val. loop ([#2615](https://github.com/PyTorchLightning/pytorch-lightning/pull/2615),
       [#2651](https://github.com/PyTorchLightning/pytorch-lightning/pull/2651))
  * weighted average in results obj ([#2930](https://github.com/PyTorchLightning/pytorch-lightning/pull/2930))
  * fix result obj DP auto reduce ([#3013](https://github.com/PyTorchLightning/pytorch-lightning/pull/3013))
- Added class `LightningDataModule` ([#2668](https://github.com/PyTorchLightning/pytorch-lightning/pull/2668))
- Added support for PyTorch 1.6 ([#2745](https://github.com/PyTorchLightning/pytorch-lightning/pull/2745))
- Added call DataModule hooks implicitly in trainer ([#2755](https://github.com/PyTorchLightning/pytorch-lightning/pull/2755))
- Added support for Mean in DDP Sync ([#2568](https://github.com/PyTorchLightning/pytorch-lightning/pull/2568))
- Added remaining `sklearn` metrics: `AveragePrecision`, `BalancedAccuracy`, `CohenKappaScore`, `DCG`, `Hamming`, `Hinge`, `Jaccard`, `MeanAbsoluteError`, `MeanSquaredError`, `MeanSquaredLogError`, `MedianAbsoluteError`, `R2Score`, `MeanPoissonDeviance`, `MeanGammaDeviance`, `MeanTweedieDeviance`, `ExplainedVariance` ([#2562](https://github.com/PyTorchLightning/pytorch-lightning/pull/2562))
- Added support for `limit_{mode}_batches (int)` to work with infinite dataloader (IterableDataset) ([#2840](https://github.com/PyTorchLightning/pytorch-lightning/pull/2840))
- Added support returning python scalars in DP ([#1935](https://github.com/PyTorchLightning/pytorch-lightning/pull/1935))
- Added support to Tensorboard logger for OmegaConf `hparams` ([#2846](https://github.com/PyTorchLightning/pytorch-lightning/pull/2846))
- Added tracking of basic states in `Trainer` ([#2541](https://github.com/PyTorchLightning/pytorch-lightning/pull/2541))
- Tracks all outputs including TBPTT and multiple optimizers ([#2890](https://github.com/PyTorchLightning/pytorch-lightning/pull/2890))
- Added GPU Usage Logger ([#2932](https://github.com/PyTorchLightning/pytorch-lightning/pull/2932))
- Added `strict=False` for `load_from_checkpoint` ([#2819](https://github.com/PyTorchLightning/pytorch-lightning/pull/2819))
- Added saving test predictions on multiple GPUs ([#2926](https://github.com/PyTorchLightning/pytorch-lightning/pull/2926))
- Auto log the computational graph for loggers that support this ([#3003](https://github.com/PyTorchLightning/pytorch-lightning/pull/3003))
- Added warning when changing monitor and using results obj ([#3014](https://github.com/PyTorchLightning/pytorch-lightning/pull/3014))
- Added a hook `transfer_batch_to_device` to the `LightningDataModule` ([#3038](https://github.com/PyTorchLightning/pytorch-lightning/pull/3038))

### Changed

- Truncated long version numbers in progress bar ([#2594](https://github.com/PyTorchLightning/pytorch-lightning/pull/2594))
- Enabling val/test loop disabling ([#2692](https://github.com/PyTorchLightning/pytorch-lightning/pull/2692))
- Refactored into `accelerator` module:
    * GPU training ([#2704](https://github.com/PyTorchLightning/pytorch-lightning/pull/2704))
    * TPU training ([#2708](https://github.com/PyTorchLightning/pytorch-lightning/pull/2708))
    * DDP(2) backend ([#2796](https://github.com/PyTorchLightning/pytorch-lightning/pull/2796))
    * Retrieve last logged val from result by key ([#3049](https://github.com/PyTorchLightning/pytorch-lightning/pull/3049))
- Using `.comet.config` file for `CometLogger` ([#1913](https://github.com/PyTorchLightning/pytorch-lightning/pull/1913))
- Updated hooks arguments - breaking for `setup` and `teardown` ([#2850](https://github.com/PyTorchLightning/pytorch-lightning/pull/2850))
- Using `gfile` to support remote directories ([#2164](https://github.com/PyTorchLightning/pytorch-lightning/pull/2164))
- Moved optimizer creation after device placement for DDP backends ([#2904](https://github.com/PyTorchLightning/pytorch-lighting/pull/2904))
- Support `**DictConfig` for `hparam` serialization ([#2519](https://github.com/PyTorchLightning/pytorch-lightning/pull/2519))
- Removed callback metrics from test results obj ([#2994](https://github.com/PyTorchLightning/pytorch-lightning/pull/2994))
- Re-enabled naming metrics in ckpt name ([#3060](https://github.com/PyTorchLightning/pytorch-lightning/pull/3060))
- Changed progress bar epoch counting to start from 0 ([#3061](https://github.com/PyTorchLightning/pytorch-lightning/pull/3061))

### Deprecated

- Deprecated Trainer attribute `ckpt_path`, which will now be set by `weights_save_path` ([#2681](https://github.com/PyTorchLightning/pytorch-lightning/pull/2681))

### Removed

- Removed deprecated: ([#2760](https://github.com/PyTorchLightning/pytorch-lightning/pull/2760))
    * core decorator `data_loader`
    * Module hook `on_sanity_check_start` and loading `load_from_metrics`
    * package `pytorch_lightning.logging`
    * Trainer arguments: `show_progress_bar`, `num_tpu_cores`, `use_amp`, `print_nan_grads`
    * LR Finder argument `num_accumulation_steps`

### Fixed

- Fixed `accumulate_grad_batches` for last batch ([#2853](https://github.com/PyTorchLightning/pytorch-lightning/pull/2853))
- Fixed setup call while testing ([#2624](https://github.com/PyTorchLightning/pytorch-lightning/pull/2624))
- Fixed local rank zero casting ([#2640](https://github.com/PyTorchLightning/pytorch-lightning/pull/2640))
- Fixed single scalar return from training ([#2587](https://github.com/PyTorchLightning/pytorch-lightning/pull/2587))
- Fixed Horovod backend to scale LR schedlers with the optimizer ([#2626](https://github.com/PyTorchLightning/pytorch-lightning/pull/2626))
- Fixed `dtype` and `device` properties not getting updated in submodules ([#2657](https://github.com/PyTorchLightning/pytorch-lightning/pull/2657))
- Fixed `fast_dev_run` to run for all dataloaders ([#2581](https://github.com/PyTorchLightning/pytorch-lightning/pull/2581))
- Fixed `save_dir` in loggers getting ignored by default value of `weights_save_path` when user did not specify `weights_save_path` ([#2681](https://github.com/PyTorchLightning/pytorch-lightning/pull/2681))
- Fixed `weights_save_path` getting ignored when `logger=False` is passed to Trainer ([#2681](https://github.com/PyTorchLightning/pytorch-lightning/pull/2681))
- Fixed TPU multi-core and Float16 ([#2632](https://github.com/PyTorchLightning/pytorch-lightning/pull/2632))
- Fixed test metrics not being logged with `LoggerCollection` ([#2723](https://github.com/PyTorchLightning/pytorch-lightning/pull/2723))
- Fixed data transfer to device when using `torchtext.data.Field` and `include_lengths is True` ([#2689](https://github.com/PyTorchLightning/pytorch-lightning/pull/2689))
- Fixed shuffle argument for distributed sampler ([#2789](https://github.com/PyTorchLightning/pytorch-lightning/pull/2789))
- Fixed logging interval ([#2694](https://github.com/PyTorchLightning/pytorch-lightning/pull/2694))
- Fixed loss value in the progress bar is wrong when `accumulate_grad_batches > 1` ([#2738](https://github.com/PyTorchLightning/pytorch-lightning/pull/2738))
- Fixed correct CWD for ddp sub-processes when using Hydra ([#2719](https://github.com/PyTorchLightning/pytorch-lightning/pull/2719))
- Fixed selecting GPUs using `CUDA_VISIBLE_DEVICES` ([#2739](https://github.com/PyTorchLightning/pytorch-lightning/pull/2739))
- Fixed false `num_classes` warning in metrics ([#2781](https://github.com/PyTorchLightning/pytorch-lightning/pull/2781))
- Fixed shell injection vulnerability in subprocess call ([#2786](https://github.com/PyTorchLightning/pytorch-lightning/pull/2786))
- Fixed LR finder and `hparams` compatibility ([#2821](https://github.com/PyTorchLightning/pytorch-lightning/pull/2821))
- Fixed `ModelCheckpoint` not saving the latest information when `save_last=True` ([#2881](https://github.com/PyTorchLightning/pytorch-lightning/pull/2881))
- Fixed ImageNet example: learning rate scheduler, number of workers and batch size when using DDP ([#2889](https://github.com/PyTorchLightning/pytorch-lightning/pull/2889))
- Fixed apex gradient clipping ([#2829](https://github.com/PyTorchLightning/pytorch-lightning/pull/2829))
- Fixed save apex scaler states ([#2828](https://github.com/PyTorchLightning/pytorch-lightning/pull/2828))
- Fixed a model loading issue with inheritance and variable positional arguments ([#2911](https://github.com/PyTorchLightning/pytorch-lightning/pull/2911))
- Fixed passing `non_blocking=True` when transferring a batch object that does not support it ([#2910](https://github.com/PyTorchLightning/pytorch-lightning/pull/2910))
- Fixed checkpointing to remote file paths ([#2925](https://github.com/PyTorchLightning/pytorch-lightning/pull/2925))
- Fixed adding val step argument to metrics ([#2986](https://github.com/PyTorchLightning/pytorch-lightning/pull/2986))
- Fixed an issue that caused `Trainer.test()` to stall in ddp mode ([#2997](https://github.com/PyTorchLightning/pytorch-lightning/pull/2997))
- Fixed gathering of results with tensors of varying shape ([#3020](https://github.com/PyTorchLightning/pytorch-lightning/pull/3020))
- Fixed batch size auto-scaling feature to set the new value on the correct model attribute ([#3043](https://github.com/PyTorchLightning/pytorch-lightning/pull/3043))
- Fixed automatic batch scaling not working with half precision ([#3045](https://github.com/PyTorchLightning/pytorch-lightning/pull/3045))
- Fixed setting device to root gpu ([#3042](https://github.com/PyTorchLightning/pytorch-lightning/pull/3042))

## [0.8.5] - 2020-07-09

### Added

- Added a PSNR metric: peak signal-to-noise ratio ([#2483](https://github.com/PyTorchLightning/pytorch-lightning/pull/2483))
- Added functional regression metrics ([#2492](https://github.com/PyTorchLightning/pytorch-lightning/pull/2492))

### Removed

- Removed auto val reduce ([#2462](https://github.com/PyTorchLightning/pytorch-lightning/pull/2462))

### Fixed

- Flattening Wandb Hyperparameters ([#2459](https://github.com/PyTorchLightning/pytorch-lightning/pull/2459))
- Fixed using the same DDP python interpreter and actually running ([#2482](https://github.com/PyTorchLightning/pytorch-lightning/pull/2482))
- Fixed model summary input type conversion for models that have input dtype different from model parameters ([#2510](https://github.com/PyTorchLightning/pytorch-lightning/pull/2510))
- Made `TensorBoardLogger` and `CometLogger` pickleable ([#2518](https://github.com/PyTorchLightning/pytorch-lightning/pull/2518))
- Fixed a problem with `MLflowLogger` creating multiple run folders ([#2502](https://github.com/PyTorchLightning/pytorch-lightning/pull/2502))
- Fixed global_step increment ([#2455](https://github.com/PyTorchLightning/pytorch-lightning/pull/2455))
- Fixed TPU hanging example ([#2488](https://github.com/PyTorchLightning/pytorch-lightning/pull/2488))
- Fixed `argparse` default value bug ([#2526](https://github.com/PyTorchLightning/pytorch-lightning/pull/2526))
- Fixed Dice and IoU to avoid NaN by adding small eps ([#2545](https://github.com/PyTorchLightning/pytorch-lightning/pull/2545))
- Fixed accumulate gradients schedule at epoch 0 (continued) ([#2513](https://github.com/PyTorchLightning/pytorch-lightning/pull/2513))
- Fixed Trainer `.fit()` returning last not best weights in "ddp_spawn" ([#2565](https://github.com/PyTorchLightning/pytorch-lightning/pull/2565))
- Fixed passing (do not pass) TPU weights back on test ([#2566](https://github.com/PyTorchLightning/pytorch-lightning/pull/2566))
- Fixed DDP tests and `.test()` ([#2512](https://github.com/PyTorchLightning/pytorch-lightning/pull/2512),
     [#2570](https://github.com/PyTorchLightning/pytorch-lightning/pull/2570))

## [0.8.4] - 2020-07-01

### Added

- Added reduce ddp results on eval ([#2434](https://github.com/PyTorchLightning/pytorch-lightning/pull/2434))
- Added a warning when an `IterableDataset` has `__len__` defined ([#2437](https://github.com/PyTorchLightning/pytorch-lightning/pull/2437))

### Changed

- Enabled no returns from eval ([#2446](https://github.com/PyTorchLightning/pytorch-lightning/pull/2446))

### Fixed

- Fixes train outputs ([#2428](https://github.com/PyTorchLightning/pytorch-lightning/pull/2428))
- Fixes Conda dependencies ([#2412](https://github.com/PyTorchLightning/pytorch-lightning/pull/2412))
- Fixed Apex scaling with decoupled backward ([#2433](https://github.com/PyTorchLightning/pytorch-lightning/pull/2433))
- Fixed crashing or wrong displaying progressbar because of missing ipywidgets ([#2417](https://github.com/PyTorchLightning/pytorch-lightning/pull/2417))
- Fixed TPU saving dir ([fc26078e](https://github.com/PyTorchLightning/pytorch-lightning/commit/fc26078e395f8a001f4c6dd7b3fe7ca202f914a3), [04e68f02](https://github.com/PyTorchLightning/pytorch-lightning/commit/04e68f022fc03dd5f1555ee86dea997d42a448ad))
- Fixed logging on rank 0 only ([#2425](https://github.com/PyTorchLightning/pytorch-lightning/pull/2425))


## [0.8.3] - 2020-06-29

### Fixed

- Fixed AMP wrong call ([593837e](https://github.com/PyTorchLightning/pytorch-lightning/commit/593837e1da24ff6c942b24ed803fc1496a304609))
- Fixed batch typo ([92d1e75](https://github.com/PyTorchLightning/pytorch-lightning/commit/92d1e75b2638a493d9d21ed5fe00a22093888285))

## [0.8.2] - 2020-06-28

### Added

- Added TorchText support for moving data to GPU ([#2379](https://github.com/PyTorchLightning/pytorch-lightning/pull/2379))

### Changed

- Changed epoch indexing from 0 instead of 1 ([#2289](https://github.com/PyTorchLightning/pytorch-lightning/pull/2289))
- Refactor Model `backward` ([#2276](https://github.com/PyTorchLightning/pytorch-lightning/pull/2276))
- Refactored `training_batch` + tests to verify correctness ([#2327](https://github.com/PyTorchLightning/pytorch-lightning/pull/2327),
     [#2328](https://github.com/PyTorchLightning/pytorch-lightning/pull/2328))
- Refactored training loop ([#2336](https://github.com/PyTorchLightning/pytorch-lightning/pull/2336))
- Made optimization steps for hooks ([#2363](https://github.com/PyTorchLightning/pytorch-lightning/pull/2363))
- Changed default apex level to 'O2' ([#2362](https://github.com/PyTorchLightning/pytorch-lightning/pull/2362))

### Removed

- Moved `TrainsLogger` to Bolts ([#2384](https://github.com/PyTorchLightning/pytorch-lightning/pull/2384))

### Fixed

- Fixed parsing TPU arguments and TPU tests ([#2094](https://github.com/PyTorchLightning/pytorch-lightning/pull/2094))
- Fixed number batches in case of multiple dataloaders and `limit_{*}_batches` ([#1920](https://github.com/PyTorchLightning/pytorch-lightning/pull/1920),
     [#2226](https://github.com/PyTorchLightning/pytorch-lightning/pull/2226))
- Fixed an issue with forward hooks not being removed after model summary ([#2298](https://github.com/PyTorchLightning/pytorch-lightning/pull/2298))
- Fix for `load_from_checkpoint()` not working with absolute path on Windows ([#2294](https://github.com/PyTorchLightning/pytorch-lightning/pull/2294))
- Fixed an issue how _has_len handles `NotImplementedError` e.g. raised by `torchtext.data.Iterator` ([#2293](https://github.com/PyTorchLightning/pytorch-lightning/pull/2293)), ([#2307](https://github.com/PyTorchLightning/pytorch-lightning/pull/2307))
- Fixed `average_precision` metric ([#2319](https://github.com/PyTorchLightning/pytorch-lightning/pull/2319))
- Fixed ROC metric for CUDA tensors ([#2304](https://github.com/PyTorchLightning/pytorch-lightning/pull/2304))
- Fixed lost compatibility with custom datatypes implementing `.to` ([#2335](https://github.com/PyTorchLightning/pytorch-lightning/pull/2335))
- Fixed loading model with kwargs ([#2387](https://github.com/PyTorchLightning/pytorch-lightning/pull/2387))
- Fixed sum(0) for `trainer.num_val_batches` ([#2268](https://github.com/PyTorchLightning/pytorch-lightning/pull/2268))
- Fixed checking if the parameters are a `DictConfig` Object ([#2216](https://github.com/PyTorchLightning/pytorch-lightning/pull/2216))
- Fixed SLURM weights saving ([#2341](https://github.com/PyTorchLightning/pytorch-lightning/pull/2341))
- Fixed swaps LR scheduler order ([#2356](https://github.com/PyTorchLightning/pytorch-lightning/pull/2356))
- Fixed adding tensorboard `hparams` logging test ([#2342](https://github.com/PyTorchLightning/pytorch-lightning/pull/2342))
- Fixed use model ref for tear down ([#2360](https://github.com/PyTorchLightning/pytorch-lightning/pull/2360))
- Fixed logger crash on DDP ([#2388](https://github.com/PyTorchLightning/pytorch-lightning/pull/2388))
- Fixed several issues with early stopping and checkpoint callbacks ([#1504](https://github.com/PyTorchLightning/pytorch-lightning/pull/1504),
     [#2391](https://github.com/PyTorchLightning/pytorch-lightning/pull/2391))
- Fixed loading past checkpoints from v0.7.x ([#2405](https://github.com/PyTorchLightning/pytorch-lightning/pull/2405))
- Fixed loading model without arguments ([#2403](https://github.com/PyTorchLightning/pytorch-lightning/pull/2403))
- Fixed Windows compatibility issue ([#2358](https://github.com/PyTorchLightning/pytorch-lightning/pull/2358))

## [0.8.1] - 2020-06-19

### Fixed

- Fixed the `load_from_checkpoint` path detected as URL bug ([#2244](https://github.com/PyTorchLightning/pytorch-lightning/pull/2244))
- Fixed hooks - added barrier ([#2245](https://github.com/PyTorchLightning/pytorch-lightning/pull/2245),
     [#2257](https://github.com/PyTorchLightning/pytorch-lightning/pull/2257),
     [#2260](https://github.com/PyTorchLightning/pytorch-lightning/pull/220))
- Fixed `hparams` - remove frame inspection on `self.hparams` ([#2253](https://github.com/PyTorchLightning/pytorch-lightning/pull/2253))
- Fixed setup and on fit calls ([#2252](https://github.com/PyTorchLightning/pytorch-lightning/pull/2252))
- Fixed GPU template ([#2255](https://github.com/PyTorchLightning/pytorch-lightning/pull/2255))

## [0.8.0] - 2020-06-18

### Added

- Added `overfit_batches`, `limit_{val|test}_batches` flags (overfit now uses training set for all three) ([#2213](https://github.com/PyTorchLightning/pytorch-lightning/pull/2213))
- Added metrics
  * Base classes ([#1326](https://github.com/PyTorchLightning/pytorch-lightning/pull/1326),
       [#1877](https://github.com/PyTorchLightning/pytorch-lightning/pull/1877))
  * Sklearn metrics classes ([#1327](https://github.com/PyTorchLightning/pytorch-lightning/pull/1327))
  * Native torch metrics ([#1488](https://github.com/PyTorchLightning/pytorch-lightning/pull/1488),
       [#2062](https://github.com/PyTorchLightning/pytorch-lightning/pull/2062))
  * docs for all Metrics ([#2184](https://github.com/PyTorchLightning/pytorch-lightning/pull/2184),
       [#2209](https://github.com/PyTorchLightning/pytorch-lightning/pull/2209))
  * Regression metrics ([#2221](https://github.com/PyTorchLightning/pytorch-lightning/pull/2221))
- Allow dataloaders without sampler field present ([#1907](https://github.com/PyTorchLightning/pytorch-lightning/pull/1907))
- Added option `save_last` to save the model at the end of every epoch in `ModelCheckpoint` ([#1908](https://github.com/PyTorchLightning/pytorch-lightning/pull/1908))
- Early stopping checks `on_validation_end` ([#1458](https://github.com/PyTorchLightning/pytorch-lightning/pull/1458))
- Speed up single-core TPU training by loading data using `ParallelLoader` ([#2033](https://github.com/PyTorchLightning/pytorch-lightning/pull/2033))
- Added a model hook `transfer_batch_to_device` that enables moving custom data structures to the target device ([#1756](https://github.com/PyTorchLightning/pytorch-lightning/pull/1756))
- Added [black](https://black.readthedocs.io/en/stable/) formatter for the code with code-checker on pull ([#1610](https://github.com/PyTorchLightning/pytorch-lightning/pull/1610))
- Added back the slow spawn ddp implementation as `ddp_spawn` ([#2115](https://github.com/PyTorchLightning/pytorch-lightning/pull/2115))
- Added loading checkpoints from URLs ([#1667](https://github.com/PyTorchLightning/pytorch-lightning/pull/1667))
- Added a callback method `on_keyboard_interrupt` for handling KeyboardInterrupt events during training ([#2134](https://github.com/PyTorchLightning/pytorch-lightning/pull/2134))
- Added a decorator `auto_move_data` that moves data to the correct device when using the LightningModule for inference ([#1905](https://github.com/PyTorchLightning/pytorch-lightning/pull/1905))
- Added `ckpt_path` option to `LightningModule.test(...)` to load particular checkpoint ([#2190](https://github.com/PyTorchLightning/pytorch-lightning/pull/2190))
- Added `setup` and `teardown` hooks for model ([#2229](https://github.com/PyTorchLightning/pytorch-lightning/pull/2229))

### Changed

- Allow user to select individual TPU core to train on ([#1729](https://github.com/PyTorchLightning/pytorch-lightning/pull/1729))
- Removed non-finite values from loss in `LRFinder` ([#1862](https://github.com/PyTorchLightning/pytorch-lightning/pull/1862))
- Allow passing model hyperparameters as complete kwarg list ([#1896](https://github.com/PyTorchLightning/pytorch-lightning/pull/1896))
- Renamed `ModelCheckpoint`'s attributes `best` to `best_model_score` and `kth_best_model` to `kth_best_model_path` ([#1799](https://github.com/PyTorchLightning/pytorch-lightning/pull/1799))
- Re-Enable Logger's `ImportError`s ([#1938](https://github.com/PyTorchLightning/pytorch-lightning/pull/1938))
- Changed the default value of the Trainer argument `weights_summary` from `full` to `top` ([#2029](https://github.com/PyTorchLightning/pytorch-lightning/pull/2029))
- Raise an error when lightning replaces an existing sampler ([#2020](https://github.com/PyTorchLightning/pytorch-lightning/pull/2020))
- Enabled `prepare_data` from correct processes - clarify local vs global rank ([#2166](https://github.com/PyTorchLightning/pytorch-lightning/pull/2166))
- Remove explicit flush from tensorboard logger ([#2126](https://github.com/PyTorchLightning/pytorch-lightning/pull/2126))
- Changed epoch indexing from 1 instead of 0 ([#2206](https://github.com/PyTorchLightning/pytorch-lightning/pull/2206))

### Deprecated

- Deprecated flags: ([#2213](https://github.com/PyTorchLightning/pytorch-lightning/pull/2213))
  * `overfit_pct` in favour of `overfit_batches`
  * `val_percent_check` in favour of `limit_val_batches`
  * `test_percent_check` in favour of `limit_test_batches`
- Deprecated `ModelCheckpoint`'s attributes `best` and `kth_best_model` ([#1799](https://github.com/PyTorchLightning/pytorch-lightning/pull/1799))
- Dropped official support/testing for older PyTorch versions <1.3 ([#1917](https://github.com/PyTorchLightning/pytorch-lightning/pull/1917))
- Deprecated Trainer `proc_rank` in favour of `global_rank` ([#2166](https://github.com/PyTorchLightning/pytorch-lightning/pull/2166),
     [#2269](https://github.com/PyTorchLightning/pytorch-lightning/pull/2269))

### Removed

- Removed unintended Trainer argument `progress_bar_callback`, the callback should be passed in by `Trainer(callbacks=[...])` instead ([#1855](https://github.com/PyTorchLightning/pytorch-lightning/pull/1855))
- Removed obsolete `self._device` in Trainer ([#1849](https://github.com/PyTorchLightning/pytorch-lightning/pull/1849))
- Removed deprecated API ([#2073](https://github.com/PyTorchLightning/pytorch-lightning/pull/2073))
   * Packages: `pytorch_lightning.pt_overrides`, `pytorch_lightning.root_module`
   * Modules: `pytorch_lightning.logging.comet_logger`, `pytorch_lightning.logging.mlflow_logger`, `pytorch_lightning.logging.test_tube_logger`, `pytorch_lightning.overrides.override_data_parallel`, `pytorch_lightning.core.model_saving`, `pytorch_lightning.core.root_module`
   * Trainer arguments: `add_row_log_interval`, `default_save_path`, `gradient_clip`, `nb_gpu_nodes`, `max_nb_epochs`, `min_nb_epochs`, `nb_sanity_val_steps`
   * Trainer attributes: `nb_gpu_nodes`, `num_gpu_nodes`, `gradient_clip`, `max_nb_epochs`, `min_nb_epochs`, `nb_sanity_val_steps`, `default_save_path`, `tng_tqdm_dic`

### Fixed

- Run graceful training teardown on interpreter exit ([#1631](https://github.com/PyTorchLightning/pytorch-lightning/pull/1631))
- Fixed user warning when apex was used together with learning rate schedulers ([#1873](https://github.com/PyTorchLightning/pytorch-lightning/pull/1873))
- Fixed multiple calls of `EarlyStopping` callback ([#1863](https://github.com/PyTorchLightning/pytorch-lightning/pull/1863))
- Fixed an issue with `Trainer.from_argparse_args` when passing in unknown Trainer args ([#1932](https://github.com/PyTorchLightning/pytorch-lightning/pull/1932))
- Fixed bug related to logger not being reset correctly for model after tuner algorithms ([#1933](https://github.com/PyTorchLightning/pytorch-lightning/pull/1933))
- Fixed root node resolution for SLURM cluster with dash in host name ([#1954](https://github.com/PyTorchLightning/pytorch-lightning/pull/1954))
- Fixed `LearningRateLogger` in multi-scheduler setting ([#1944](https://github.com/PyTorchLightning/pytorch-lightning/pull/1944))
- Fixed test configuration check and testing ([#1804](https://github.com/PyTorchLightning/pytorch-lightning/pull/1804))
- Fixed an issue with Trainer constructor silently ignoring unknown/misspelled arguments ([#1820](https://github.com/PyTorchLightning/pytorch-lightning/pull/1820))
- Fixed `save_weights_only` in ModelCheckpoint ([#1780](https://github.com/PyTorchLightning/pytorch-lightning/pull/1780))
- Allow use of same `WandbLogger` instance for multiple training loops ([#2055](https://github.com/PyTorchLightning/pytorch-lightning/pull/2055))
- Fixed an issue with `_auto_collect_arguments` collecting local variables that are not constructor arguments and not working for signatures that have the instance not named `self` ([#2048](https://github.com/PyTorchLightning/pytorch-lightning/pull/2048))
- Fixed mistake in parameters' grad norm tracking ([#2012](https://github.com/PyTorchLightning/pytorch-lightning/pull/2012))
- Fixed CPU and hanging GPU crash ([#2118](https://github.com/PyTorchLightning/pytorch-lightning/pull/2118))
- Fixed an issue with the model summary and `example_input_array` depending on a specific ordering of the submodules in a LightningModule ([#1773](https://github.com/PyTorchLightning/pytorch-lightning/pull/1773))
- Fixed Tpu logging ([#2230](https://github.com/PyTorchLightning/pytorch-lightning/pull/2230))
- Fixed Pid port + duplicate `rank_zero` logging ([#2140](https://github.com/PyTorchLightning/pytorch-lightning/pull/2140),
     [#2231](https://github.com/PyTorchLightning/pytorch-lightning/pull/2231))

## [0.7.6] - 2020-05-16

### Added

- Added callback for logging learning rates ([#1498](https://github.com/PyTorchLightning/pytorch-lightning/pull/1498))
- Added transfer learning example (for a binary classification task in computer vision) ([#1564](https://github.com/PyTorchLightning/pytorch-lightning/pull/1564))
- Added type hints in `Trainer.fit()` and `Trainer.test()` to reflect that also a list of dataloaders can be passed in ([#1723](https://github.com/PyTorchLightning/pytorch-lightning/pull/1723)).
- Added auto scaling of batch size ([#1638](https://github.com/PyTorchLightning/pytorch-lightning/pull/1638))
- The progress bar metrics now also get updated in `training_epoch_end` ([#1724](https://github.com/PyTorchLightning/pytorch-lightning/pull/1724))
- Enable `NeptuneLogger` to work with `distributed_backend=ddp` ([#1753](https://github.com/PyTorchLightning/pytorch-lightning/pull/1753))
- Added option to provide seed to random generators to ensure reproducibility ([#1572](https://github.com/PyTorchLightning/pytorch-lightning/pull/1572))
- Added override for hparams in `load_from_ckpt` ([#1797](https://github.com/PyTorchLightning/pytorch-lightning/pull/1797))
- Added support multi-node distributed execution under `torchelastic` ([#1811](https://github.com/PyTorchLightning/pytorch-lightning/pull/1811),
     [#1818](https://github.com/PyTorchLightning/pytorch-lightning/pull/1818))
- Added using `store_true` for bool args ([#1822](https://github.com/PyTorchLightning/pytorch-lightning/pull/1822),
     [#1842](https://github.com/PyTorchLightning/pytorch-lightning/pull/1842))
- Added dummy logger for internally disabling logging for some features ([#1836](https://github.com/PyTorchLightning/pytorch-lightning/pull/1836))

### Changed

- Enable `non-blocking` for device transfers to GPU ([#1843](https://github.com/PyTorchLightning/pytorch-lightning/pull/1843))
- Replace mata_tags.csv with hparams.yaml ([#1271](https://github.com/PyTorchLightning/pytorch-lightning/pull/1271))
- Reduction when `batch_size < num_gpus` ([#1609](https://github.com/PyTorchLightning/pytorch-lightning/pull/1609))
- Updated LightningTemplateModel to look more like Colab example ([#1577](https://github.com/PyTorchLightning/pytorch-lightning/pull/1577))
- Don't convert `namedtuple` to `tuple` when transferring the batch to target device ([#1589](https://github.com/PyTorchLightning/pytorch-lightning/pull/1589))
- Allow passing hparams as keyword argument to LightningModule when loading from checkpoint ([#1639](https://github.com/PyTorchLightning/pytorch-lightning/pull/1639))
- Args should come after the last positional argument ([#1807](https://github.com/PyTorchLightning/pytorch-lightning/pull/1807))
- Made ddp the default if no backend specified with multiple GPUs ([#1789](https://github.com/PyTorchLightning/pytorch-lightning/pull/1789))

### Deprecated

- Deprecated `tags_csv` in favor of `hparams_file` ([#1271](https://github.com/PyTorchLightning/pytorch-lightning/pull/1271))

### Fixed

- Fixed broken link in PR template ([#1675](https://github.com/PyTorchLightning/pytorch-lightning/pull/1675))
- Fixed ModelCheckpoint not None checking filepath ([#1654](https://github.com/PyTorchLightning/pytorch-lightning/pull/1654))
- Trainer now calls `on_load_checkpoint()` when resuming from a checkpoint ([#1666](https://github.com/PyTorchLightning/pytorch-lightning/pull/1666))
- Fixed sampler logic for ddp with iterable dataset ([#1734](https://github.com/PyTorchLightning/pytorch-lightning/pull/1734))
- Fixed `_reset_eval_dataloader()` for IterableDataset ([#1560](https://github.com/PyTorchLightning/pytorch-lightning/pull/1560))
- Fixed Horovod distributed backend to set the `root_gpu` property ([#1669](https://github.com/PyTorchLightning/pytorch-lightning/pull/1669))
- Fixed wandb logger `global_step` affects other loggers ([#1492](https://github.com/PyTorchLightning/pytorch-lightning/pull/1492))
- Fixed disabling progress bar on non-zero ranks using Horovod backend ([#1709](https://github.com/PyTorchLightning/pytorch-lightning/pull/1709))
- Fixed bugs that prevent lr finder to be used together with early stopping and validation dataloaders ([#1676](https://github.com/PyTorchLightning/pytorch-lightning/pull/1676))
- Fixed a bug in Trainer that prepended the checkpoint path with `version_` when it shouldn't ([#1748](https://github.com/PyTorchLightning/pytorch-lightning/pull/1748))
- Fixed lr key name in case of param groups in LearningRateLogger ([#1719](https://github.com/PyTorchLightning/pytorch-lightning/pull/1719))
- Fixed accumulation parameter and suggestion method for learning rate finder ([#1801](https://github.com/PyTorchLightning/pytorch-lightning/pull/1801))
- Fixed num processes wasn't being set properly and auto sampler was ddp failing ([#1819](https://github.com/PyTorchLightning/pytorch-lightning/pull/1819))
- Fixed bugs in semantic segmentation example ([#1824](https://github.com/PyTorchLightning/pytorch-lightning/pull/1824))
- Fixed saving native AMP scaler state ([#1777](https://github.com/PyTorchLightning/pytorch-lightning/pull/1777))
- Fixed native amp + ddp ([#1788](https://github.com/PyTorchLightning/pytorch-lightning/pull/1788))
- Fixed `hparam` logging with metrics ([#1647](https://github.com/PyTorchLightning/pytorch-lightning/pull/1647))

## [0.7.5] - 2020-04-27

### Changed

- Allow logging of metrics together with `hparams` ([#1630](https://github.com/PyTorchLightning/pytorch-lightning/pull/1630))

### Removed

- Removed Warning from trainer loop ([#1634](https://github.com/PyTorchLightning/pytorch-lightning/pull/1634))

### Fixed

- Fixed ModelCheckpoint not being fixable ([#1632](https://github.com/PyTorchLightning/pytorch-lightning/pull/1632))
- Fixed CPU DDP breaking change and DDP change ([#1635](https://github.com/PyTorchLightning/pytorch-lightning/pull/1635))
- Tested pickling ([#1636](https://github.com/PyTorchLightning/pytorch-lightning/pull/1636))


## [0.7.4] - 2020-04-26

### Added

- Added flag `replace_sampler_ddp` to manually disable sampler replacement in DDP  ([#1513](https://github.com/PyTorchLightning/pytorch-lightning/pull/1513))
- Added `auto_select_gpus` flag to trainer that enables automatic selection of available GPUs on exclusive mode systems.
- Added learning rate finder ([#1347](https://github.com/PyTorchLightning/pytorch-lightning/pull/1347))
- Added support for DDP mode in clusters without SLURM ([#1387](https://github.com/PyTorchLightning/pytorch-lightning/pull/1387))
- Added `test_dataloaders` parameter to `Trainer.test()` ([#1434](https://github.com/PyTorchLightning/pytorch-lightning/pull/1434))
- Added `terminate_on_nan` flag to trainer that performs a NaN check with each training iteration when set to `True` ([#1475](https://github.com/PyTorchLightning/pytorch-lightning/pull/1475))
- Added speed parity tests (max 1 sec difference per epoch)([#1482](https://github.com/PyTorchLightning/pytorch-lightning/pull/1482))
- Added `ddp_cpu` backend for testing ddp without GPUs ([#1158](https://github.com/PyTorchLightning/pytorch-lightning/pull/1158))
- Added [Horovod](http://horovod.ai) support as a distributed backend `Trainer(distributed_backend='horovod')` ([#1529](https://github.com/PyTorchLightning/pytorch-lightning/pull/1529))
- Added support for 8 core distributed training on Kaggle TPU's ([#1568](https://github.com/PyTorchLightning/pytorch-lightning/pull/1568))
- Added support for native AMP ([#1561](https://github.com/PyTorchLightning/pytorch-lightning/pull/1561),
    [#1580](https://github.com/PyTorchLightning/pytorch-lightning/pull/1580))

### Changed

- Changed the default behaviour to no longer include a NaN check with each training iteration ([#1475](https://github.com/PyTorchLightning/pytorch-lightning/pull/1475))
- Decoupled the progress bar from trainer` it is a callback now and can be customized or even be replaced entirely ([#1450](https://github.com/PyTorchLightning/pytorch-lightning/pull/1450)).
- Changed lr schedule step interval behavior to update every backwards pass instead of every forwards pass ([#1477](https://github.com/PyTorchLightning/pytorch-lightning/pull/1477))
- Defines shared proc. rank, remove rank from instances (e.g. loggers) ([#1408](https://github.com/PyTorchLightning/pytorch-lightning/pull/1408))
- Updated semantic segmentation example with custom U-Net and logging ([#1371](https://github.com/PyTorchLightning/pytorch-lightning/pull/1371))
- Disabled val and test shuffling ([#1600](https://github.com/PyTorchLightning/pytorch-lightning/pull/1600))

### Deprecated

- Deprecated `training_tqdm_dict` in favor of `progress_bar_dict` ([#1450](https://github.com/PyTorchLightning/pytorch-lightning/pull/1450)).

### Removed

- Removed `test_dataloaders` parameter from `Trainer.fit()` ([#1434](https://github.com/PyTorchLightning/pytorch-lightning/pull/1434))

### Fixed

- Added the possibility to pass nested metrics dictionaries to loggers ([#1582](https://github.com/PyTorchLightning/pytorch-lightning/pull/1582))
- Fixed memory leak from opt return ([#1528](https://github.com/PyTorchLightning/pytorch-lightning/pull/1528))
- Fixed saving checkpoint before deleting old ones ([#1453](https://github.com/PyTorchLightning/pytorch-lightning/pull/1453))
- Fixed loggers - flushing last logged metrics even before continue, e.g. `trainer.test()` results ([#1459](https://github.com/PyTorchLightning/pytorch-lightning/pull/1459))
- Fixed optimizer configuration when `configure_optimizers` returns dict without `lr_scheduler` ([#1443](https://github.com/PyTorchLightning/pytorch-lightning/pull/1443))
- Fixed `LightningModule` - mixing hparams and arguments in `LightningModule.__init__()` crashes load_from_checkpoint() ([#1505](https://github.com/PyTorchLightning/pytorch-lightning/pull/1505))
- Added a missing call to the `on_before_zero_grad` model hook ([#1493](https://github.com/PyTorchLightning/pytorch-lightning/pull/1493)).
- Allow use of sweeps with `WandbLogger` ([#1512](https://github.com/PyTorchLightning/pytorch-lightning/pull/1512))
- Fixed a bug that caused the `callbacks` Trainer argument to reference a global variable ([#1534](https://github.com/PyTorchLightning/pytorch-lightning/pull/1534)).
- Fixed a bug that set all boolean CLI arguments from `Trainer.add_argparse_args` always to True ([#1571](https://github.com/PyTorchLightning/pytorch-lightning/pull/1571))
- Fixed do not copy the batch when training on a single GPU ([#1576](https://github.com/PyTorchLightning/pytorch-lightning/pull/1576),
    [#1579](https://github.com/PyTorchLightning/pytorch-lightning/pull/1579))
- Fixed soft checkpoint removing on DDP ([#1408](https://github.com/PyTorchLightning/pytorch-lightning/pull/1408))
- Fixed automatic parser bug ([#1585](https://github.com/PyTorchLightning/pytorch-lightning/pull/1585))
- Fixed bool conversion from string ([#1606](https://github.com/PyTorchLightning/pytorch-lightning/pull/1606))

## [0.7.3] - 2020-04-09

### Added

- Added `rank_zero_warn` for warning only in rank 0 ([#1428](https://github.com/PyTorchLightning/pytorch-lightning/pull/1428))

### Fixed

- Fixed default `DistributedSampler` for DDP training ([#1425](https://github.com/PyTorchLightning/pytorch-lightning/pull/1425))
- Fixed workers warning not on windows ([#1430](https://github.com/PyTorchLightning/pytorch-lightning/pull/1430))
- Fixed returning tuple from `run_training_batch` ([#1431](https://github.com/PyTorchLightning/pytorch-lightning/pull/1431))
- Fixed gradient clipping ([#1438](https://github.com/PyTorchLightning/pytorch-lightning/pull/1438))
- Fixed pretty print ([#1441](https://github.com/PyTorchLightning/pytorch-lightning/pull/1441))


## [0.7.2] - 2020-04-07

### Added

- Added same step loggers' metrics aggregation ([#1278](https://github.com/PyTorchLightning/pytorch-lightning/pull/1278))
- Added parity test between a vanilla MNIST model and lightning model ([#1284](https://github.com/PyTorchLightning/pytorch-lightning/pull/1284))
- Added parity test between a vanilla RNN model and lightning model ([#1351](https://github.com/PyTorchLightning/pytorch-lightning/pull/1351))
- Added Reinforcement Learning - Deep Q-network (DQN) lightning example ([#1232](https://github.com/PyTorchLightning/pytorch-lightning/pull/1232))
- Added support for hierarchical `dict` ([#1152](https://github.com/PyTorchLightning/pytorch-lightning/pull/1152))
- Added `TrainsLogger` class ([#1122](https://github.com/PyTorchLightning/pytorch-lightning/pull/1122))
- Added type hints to `pytorch_lightning.core` ([#946](https://github.com/PyTorchLightning/pytorch-lightning/pull/946))
- Added support for `IterableDataset` in validation and testing ([#1104](https://github.com/PyTorchLightning/pytorch-lightning/pull/1104))
- Added support for non-primitive types in `hparams` for `TensorboardLogger` ([#1130](https://github.com/PyTorchLightning/pytorch-lightning/pull/1130))
- Added a check that stops the training when loss or weights contain `NaN` or `inf` values. ([#1097](https://github.com/PyTorchLightning/pytorch-lightning/pull/1097))
- Added support for `IterableDataset` when `val_check_interval=1.0` (default), this will trigger validation at the end of each epoch. ([#1283](https://github.com/PyTorchLightning/pytorch-lightning/pull/1283))
- Added `summary` method to Profilers. ([#1259](https://github.com/PyTorchLightning/pytorch-lightning/pull/1259))
- Added informative errors if user defined dataloader has zero length ([#1280](https://github.com/PyTorchLightning/pytorch-lightning/pull/1280))
- Added testing for python 3.8 ([#915](https://github.com/PyTorchLightning/pytorch-lightning/pull/915))
- Added model configuration checking ([#1199](https://github.com/PyTorchLightning/pytorch-lightning/pull/1199))
- Added support for optimizer frequencies through `LightningModule.configure_optimizers()` ([#1269](https://github.com/PyTorchLightning/pytorch-lightning/pull/1269))
- Added option to run without an optimizer by returning `None` from `configure_optimizers`. ([#1279](https://github.com/PyTorchLightning/pytorch-lightning/pull/1279))
- Added a warning when the number of data loader workers is small. ([#1378](https://github.com/PyTorchLightning/pytorch-lightning/pull/1378))

### Changed

- Changed (renamed and refatored) `TensorRunningMean` -> `TensorRunningAccum`: running accumulations were generalized. ([#1278](https://github.com/PyTorchLightning/pytorch-lightning/pull/1278))
- Changed `progress_bar_refresh_rate` trainer flag to disable progress bar when set to 0. ([#1108](https://github.com/PyTorchLightning/pytorch-lightning/pull/1108))
- Enhanced `load_from_checkpoint` to also forward params to the model ([#1307](https://github.com/PyTorchLightning/pytorch-lightning/pull/1307))
- Updated references to `self.forward()` to instead use the `__call__` interface. ([#1211](https://github.com/PyTorchLightning/pytorch-lightning/pull/1211))
- Changed default behaviour of `configure_optimizers` to use no optimizer rather than Adam. ([#1279](https://github.com/PyTorchLightning/pytorch-lightning/pull/1279))
- Allow to upload models on W&B ([#1339](https://github.com/PyTorchLightning/pytorch-lightning/pull/1339))
- On DP and DDP2 unsqueeze is automated now ([#1319](https://github.com/PyTorchLightning/pytorch-lightning/pull/1319))
- Did not always create a DataLoader during reinstantiation, but the same type as before (if subclass of DataLoader) ([#1346](https://github.com/PyTorchLightning/pytorch-lightning/pull/1346))
- Did not interfere with a default sampler ([#1318](https://github.com/PyTorchLightning/pytorch-lightning/pull/1318))
- Remove default Adam optimizer ([#1317](https://github.com/PyTorchLightning/pytorch-lightning/pull/1317))
- Give warnings for unimplemented required lightning methods ([#1317](https://github.com/PyTorchLightning/pytorch-lightning/pull/1317))
- Made `evaluate` method private >> `Trainer._evaluate(...)`. ([#1260](https://github.com/PyTorchLightning/pytorch-lightning/pull/1260))
- Simplify the PL examples structure (shallower and more readable) ([#1247](https://github.com/PyTorchLightning/pytorch-lightning/pull/1247))
- Changed min max gpu memory to be on their own plots ([#1358](https://github.com/PyTorchLightning/pytorch-lightning/pull/1358))
- Remove `.item` which causes sync issues ([#1254](https://github.com/PyTorchLightning/pytorch-lightning/pull/1254))
- Changed smoothing in TQDM to decrease variability of time remaining between training / eval ([#1194](https://github.com/PyTorchLightning/pytorch-lightning/pull/1194))
- Change default logger to dedicated one ([#1064](https://github.com/PyTorchLightning/pytorch-lightning/pull/1064))

### Deprecated

- Deprecated Trainer argument `print_nan_grads` ([#1097](https://github.com/PyTorchLightning/pytorch-lightning/pull/1097))
- Deprecated Trainer argument `show_progress_bar` ([#1108](https://github.com/PyTorchLightning/pytorch-lightning/pull/1108))

### Removed

- Removed test for no test dataloader in .fit ([#1495](https://github.com/PyTorchLightning/pytorch-lightning/pull/1495))
- Removed duplicated module `pytorch_lightning.utilities.arg_parse` for loading CLI arguments ([#1167](https://github.com/PyTorchLightning/pytorch-lightning/pull/1167))
- Removed wandb logger's `finalize` method ([#1193](https://github.com/PyTorchLightning/pytorch-lightning/pull/1193))
- Dropped `torchvision` dependency in tests and added own MNIST dataset class instead ([#986](https://github.com/PyTorchLightning/pytorch-lightning/pull/986))

### Fixed

- Fixed `model_checkpoint` when saving all models ([#1359](https://github.com/PyTorchLightning/pytorch-lightning/pull/1359))
- `Trainer.add_argparse_args` classmethod fixed. Now it adds a type for the arguments ([#1147](https://github.com/PyTorchLightning/pytorch-lightning/pull/1147))
- Fixed bug related to type checking of `ReduceLROnPlateau` lr schedulers([#1126](https://github.com/PyTorchLightning/pytorch-lightning/pull/1126))
- Fixed a bug to ensure lightning checkpoints to be backward compatible ([#1132](https://github.com/PyTorchLightning/pytorch-lightning/pull/1132))
- Fixed a bug that created an extra dataloader with active `reload_dataloaders_every_epoch` ([#1196](https://github.com/PyTorchLightning/pytorch-lightning/pull/1196))
- Fixed all warnings and errors in the docs build process ([#1191](https://github.com/PyTorchLightning/pytorch-lightning/pull/1191))
- Fixed an issue where `val_percent_check=0` would not disable validation ([#1251](https://github.com/PyTorchLightning/pytorch-lightning/pull/1251))
- Fixed average of incomplete `TensorRunningMean` ([#1309](https://github.com/PyTorchLightning/pytorch-lightning/pull/1309))
- Fixed `WandbLogger.watch` with `wandb.init()` ([#1311](https://github.com/PyTorchLightning/pytorch-lightning/pull/1311))
- Fixed an issue with early stopping that would prevent it from monitoring training metrics when validation is disabled / not implemented ([#1235](https://github.com/PyTorchLightning/pytorch-lightning/pull/1235)).
- Fixed a bug that would cause `trainer.test()` to run on the validation set when overloading `validation_epoch_end` and `test_end` ([#1353](https://github.com/PyTorchLightning/pytorch-lightning/pull/1353))
- Fixed `WandbLogger.watch` - use of the watch method without importing `wandb` ([#1311](https://github.com/PyTorchLightning/pytorch-lightning/pull/1311))
- Fixed `WandbLogger` to be used with 'ddp' - allow reinits in sub-processes ([#1149](https://github.com/PyTorchLightning/pytorch-lightning/pull/1149),
     [#1360](https://github.com/PyTorchLightning/pytorch-lightning/pull/1360))
- Made `training_epoch_end` behave like `validation_epoch_end` ([#1357](https://github.com/PyTorchLightning/pytorch-lightning/pull/1357))
- Fixed `fast_dev_run` running validation twice ([#1365](https://github.com/PyTorchLightning/pytorch-lightning/pull/1365))
- Fixed pickle error from quick patch `__code__` ([#1352](https://github.com/PyTorchLightning/pytorch-lightning/pull/1352))
- Fixed memory leak on GPU0 ([#1094](https://github.com/PyTorchLightning/pytorch-lightning/pull/1094),
     [#1349](https://github.com/PyTorchLightning/pytorch-lightning/pull/1349))
- Fixed checkpointing interval ([#1272](https://github.com/PyTorchLightning/pytorch-lightning/pull/1272))
- Fixed validation and training loops run the partial dataset ([#1192](https://github.com/PyTorchLightning/pytorch-lightning/pull/1192))
- Fixed running `on_validation_end` only on main process in DDP ([#1125](https://github.com/PyTorchLightning/pytorch-lightning/pull/1125))
- Fixed `load_spawn_weights` only in proc rank 0 ([#1385](https://github.com/PyTorchLightning/pytorch-lightning/pull/1385))
- Fixes using deprecated `use_amp` attribute ([#1145](https://github.com/PyTorchLightning/pytorch-lightning/pull/1145))
- Fixed Tensorboard logger error: lightning_logs directory not exists in multi-node DDP on nodes with rank != 0 ([#1377](https://github.com/PyTorchLightning/pytorch-lightning/pull/1377))
- Fixed `Unimplemented backend XLA` error on TPU ([#1387](https://github.com/PyTorchLightning/pytorch-lightning/pull/1387))

## [0.7.1] - 2020-03-07

### Fixed

- Fixes `print` issues and `data_loader` ([#1080](https://github.com/PyTorchLightning/pytorch-lightning/pull/1080))

## [0.7.0] - 2020-03-06

### Added

- Added automatic sampler setup. Depending on DDP or TPU, lightning configures the sampler correctly (user needs to do nothing) ([#926](https://github.com/PyTorchLightning/pytorch-lightning/pull/926))
- Added `reload_dataloaders_every_epoch=False` flag for trainer. Some users require reloading data every epoch ([#926](https://github.com/PyTorchLightning/pytorch-lightning/pull/926))
- Added `progress_bar_refresh_rate=50` flag for trainer. Throttle refresh rate on notebooks ([#926](https://github.com/PyTorchLightning/pytorch-lightning/pull/926))
- Updated governance docs
- Added a check to ensure that the metric used for early stopping exists before training commences ([#542](https://github.com/PyTorchLightning/pytorch-lightning/pull/542))
- Added `optimizer_idx` argument to `backward` hook ([#733](https://github.com/PyTorchLightning/pytorch-lightning/pull/733))
- Added `entity` argument to `WandbLogger` to be passed to `wandb.init` ([#783](https://github.com/PyTorchLightning/pytorch-lightning/pull/783))
- Added a tool for profiling training runs ([#782](https://github.com/PyTorchLightning/pytorch-lightning/pull/782))
- Improved flexibility for naming of TensorBoard logs, can now set `version` to a `str` to just save to that directory, and use `name=''` to prevent experiment-name directory ([#804](https://github.com/PyTorchLightning/pytorch-lightning/pull/804))
- Added option to specify `step` key when logging metrics ([#808](https://github.com/PyTorchLightning/pytorch-lightning/pull/808))
- Added `train_dataloader`, `val_dataloader` and `test_dataloader` arguments to `Trainer.fit()`, for alternative data parsing ([#759](https://github.com/PyTorchLightning/pytorch-lightning/pull/759))
- Added Tensor Processing Unit (TPU) support ([#868](https://github.com/PyTorchLightning/pytorch-lightning/pull/868))
- Added semantic segmentation example ([#751](https://github.com/PyTorchLightning/pytorch-lightning/pull/751),[#876](https://github.com/PyTorchLightning/pytorch-lightning/pull/876),
     [#881](https://github.com/PyTorchLightning/pytorch-lightning/pull/881))
- Split callbacks in multiple files ([#849](https://github.com/PyTorchLightning/pytorch-lightning/pull/849))
- Support for user defined callbacks ([#889](https://github.com/PyTorchLightning/pytorch-lightning/pull/889) and [#950](https://github.com/PyTorchLightning/pytorch-lightning/pull/950))
- Added support for multiple loggers to be passed to `Trainer` as an iterable (e.g. list, tuple, etc.) ([#903](https://github.com/PyTorchLightning/pytorch-lightning/pull/903))
- Added support for step-based learning rate scheduling ([#941](https://github.com/PyTorchLightning/pytorch-lightning/pull/941))
- Added support for logging `hparams` as dict ([#1029](https://github.com/PyTorchLightning/pytorch-lightning/pull/1029))
- Checkpoint and early stopping now work without val. step ([#1041](https://github.com/PyTorchLightning/pytorch-lightning/pull/1041))
- Support graceful training cleanup after Keyboard Interrupt ([#856](https://github.com/PyTorchLightning/pytorch-lightning/pull/856),
     [#1019](https://github.com/PyTorchLightning/pytorch-lightning/pull/1019))
- Added type hints for function arguments ([#912](https://github.com/PyTorchLightning/pytorch-lightning/pull/912), )
- Added default `argparser` for `Trainer` ([#952](https://github.com/PyTorchLightning/pytorch-lightning/pull/1023),
     [#1023](https://github.com/PyTorchLightning/pytorch-lightning/pull/1023))
- Added TPU gradient clipping ([#963](https://github.com/PyTorchLightning/pytorch-lightning/pull/963))
- Added max/min number of steps in `Trainer` ([#728](https://github.com/PyTorchLightning/pytorch-lightning/pull/728))

### Changed

- Improved `NeptuneLogger` by adding `close_after_fit` argument to allow logging after training([#908](https://github.com/PyTorchLightning/pytorch-lightning/pull/1084))
- Changed default TQDM to use `tqdm.auto` for prettier outputs in IPython notebooks ([#752](https://github.com/PyTorchLightning/pytorch-lightning/pull/752))
- Changed `pytorch_lightning.logging` to `pytorch_lightning.loggers` ([#767](https://github.com/PyTorchLightning/pytorch-lightning/pull/767))
- Moved the default `tqdm_dict` definition from Trainer to `LightningModule`, so it can be overridden by the user ([#749](https://github.com/PyTorchLightning/pytorch-lightning/pull/749))
- Moved functionality of `LightningModule.load_from_metrics` into `LightningModule.load_from_checkpoint` ([#995](https://github.com/PyTorchLightning/pytorch-lightning/pull/995))
- Changed Checkpoint path parameter from `filepath` to `dirpath` ([#1016](https://github.com/PyTorchLightning/pytorch-lightning/pull/1016))
- Freezed models `hparams` as `Namespace` property ([#1029](https://github.com/PyTorchLightning/pytorch-lightning/pull/1029))
- Dropped `logging` config in package init ([#1015](https://github.com/PyTorchLightning/pytorch-lightning/pull/1015))
- Renames model steps ([#1051](https://github.com/PyTorchLightning/pytorch-lightning/pull/1051))
  - `training_end` >> `training_epoch_end`
  - `validation_end` >> `validation_epoch_end`
  - `test_end` >> `test_epoch_end`
- Refactor dataloading, supports infinite dataloader ([#955](https://github.com/PyTorchLightning/pytorch-lightning/pull/955))
- Create single file in `TensorBoardLogger` ([#777](https://github.com/PyTorchLightning/pytorch-lightning/pull/777))

### Deprecated

- Deprecated `pytorch_lightning.logging` ([#767](https://github.com/PyTorchLightning/pytorch-lightning/pull/767))
- Deprecated `LightningModule.load_from_metrics` in favour of `LightningModule.load_from_checkpoint` ([#995](https://github.com/PyTorchLightning/pytorch-lightning/pull/995),
     [#1079](https://github.com/PyTorchLightning/pytorch-lightning/pull/1079))
- Deprecated `@data_loader` decorator ([#926](https://github.com/PyTorchLightning/pytorch-lightning/pull/926))
- Deprecated model steps `training_end`, `validation_end` and `test_end` ([#1051](https://github.com/PyTorchLightning/pytorch-lightning/pull/1051),
     [#1056](https://github.com/PyTorchLightning/pytorch-lightning/pull/1056))

### Removed

- Removed dependency on `pandas` ([#736](https://github.com/PyTorchLightning/pytorch-lightning/pull/736))
- Removed dependency on `torchvision` ([#797](https://github.com/PyTorchLightning/pytorch-lightning/pull/797))
- Removed dependency on `scikit-learn` ([#801](https://github.com/PyTorchLightning/pytorch-lightning/pull/801))

### Fixed

- Fixed a bug where early stopping `on_end_epoch` would be called inconsistently when `check_val_every_n_epoch == 0` ([#743](https://github.com/PyTorchLightning/pytorch-lightning/pull/743))
- Fixed a bug where the model checkpointer didn't write to the same directory as the logger ([#771](https://github.com/PyTorchLightning/pytorch-lightning/pull/771))
- Fixed a bug where the `TensorBoardLogger` class would create an additional empty log file during fitting ([#777](https://github.com/PyTorchLightning/pytorch-lightning/pull/777))
- Fixed a bug where `global_step` was advanced incorrectly when using `accumulate_grad_batches > 1` ([#832](https://github.com/PyTorchLightning/pytorch-lightning/pull/832))
- Fixed a bug when calling `self.logger.experiment` with multiple loggers ([#1009](https://github.com/PyTorchLightning/pytorch-lightning/pull/1009))
- Fixed a bug when calling `logger.append_tags` on a `NeptuneLogger` with a single tag ([#1009](https://github.com/PyTorchLightning/pytorch-lightning/pull/1009))
- Fixed sending back data from `.spawn` by saving and loading the trained model in/out of the process ([#1017](https://github.com/PyTorchLightning/pytorch-lightning/pull/1017)
- Fixed port collision on DDP ([#1010](https://github.com/PyTorchLightning/pytorch-lightning/pull/1010))
- Fixed/tested pass overrides ([#918](https://github.com/PyTorchLightning/pytorch-lightning/pull/918))
- Fixed comet logger to log after train ([#892](https://github.com/PyTorchLightning/pytorch-lightning/pull/892))
- Remove deprecated args to learning rate step function ([#890](https://github.com/PyTorchLightning/pytorch-lightning/pull/890))

## [0.6.0] - 2020-01-21

### Added

- Added support for resuming from a specific checkpoint via `resume_from_checkpoint` argument ([#516](https://github.com/PyTorchLightning/pytorch-lightning/pull/516))
- Added support for `ReduceLROnPlateau` scheduler ([#320](https://github.com/PyTorchLightning/pytorch-lightning/pull/320))
- Added support for Apex mode `O2` in conjunction with Data Parallel ([#493](https://github.com/PyTorchLightning/pytorch-lightning/pull/493))
- Added option (`save_top_k`) to save the top k models in the `ModelCheckpoint` class ([#128](https://github.com/PyTorchLightning/pytorch-lightning/pull/128))
- Added `on_train_start` and `on_train_end` hooks to `ModelHooks` ([#598](https://github.com/PyTorchLightning/pytorch-lightning/pull/598))
- Added `TensorBoardLogger` ([#607](https://github.com/PyTorchLightning/pytorch-lightning/pull/607))
- Added support for weight summary of model with multiple inputs ([#543](https://github.com/PyTorchLightning/pytorch-lightning/pull/543))
- Added `map_location` argument to `load_from_metrics` and `load_from_checkpoint` ([#625](https://github.com/PyTorchLightning/pytorch-lightning/pull/625))
- Added option to disable validation by setting `val_percent_check=0` ([#649](https://github.com/PyTorchLightning/pytorch-lightning/pull/649))
- Added `NeptuneLogger` class ([#648](https://github.com/PyTorchLightning/pytorch-lightning/pull/648))
- Added `WandbLogger` class ([#627](https://github.com/PyTorchLightning/pytorch-lightning/pull/627))

### Changed

- Changed the default progress bar to print to stdout instead of stderr ([#531](https://github.com/PyTorchLightning/pytorch-lightning/pull/531))
- Renamed `step_idx` to `step`, `epoch_idx` to `epoch`, `max_num_epochs` to `max_epochs` and `min_num_epochs` to `min_epochs` ([#589](https://github.com/PyTorchLightning/pytorch-lightning/pull/589))
- Renamed `total_batch_nb` to `total_batches`, `nb_val_batches` to `num_val_batches`, `nb_training_batches` to `num_training_batches`, `max_nb_epochs` to `max_epochs`, `min_nb_epochs` to `min_epochs`, `nb_test_batches` to `num_test_batches`, and `nb_val_batches` to `num_val_batches` ([#567](https://github.com/PyTorchLightning/pytorch-lightning/pull/567))
- Changed gradient logging to use parameter names instead of indexes ([#660](https://github.com/PyTorchLightning/pytorch-lightning/pull/660))
- Changed the default logger to `TensorBoardLogger` ([#609](https://github.com/PyTorchLightning/pytorch-lightning/pull/609))
- Changed the directory for tensorboard logging to be the same as model checkpointing ([#706](https://github.com/PyTorchLightning/pytorch-lightning/pull/706))

### Deprecated

- Deprecated `max_nb_epochs` and `min_nb_epochs` ([#567](https://github.com/PyTorchLightning/pytorch-lightning/pull/567))
- Deprecated the `on_sanity_check_start` hook in `ModelHooks` ([#598](https://github.com/PyTorchLightning/pytorch-lightning/pull/598))

### Removed

- Removed the `save_best_only` argument from `ModelCheckpoint`, use `save_top_k=1` instead ([#128](https://github.com/PyTorchLightning/pytorch-lightning/pull/128))

### Fixed

- Fixed a bug which ocurred when using Adagrad with cuda ([#554](https://github.com/PyTorchLightning/pytorch-lightning/pull/554))
- Fixed a bug where training would be on the GPU despite setting `gpus=0` or `gpus=[]` ([#561](https://github.com/PyTorchLightning/pytorch-lightning/pull/561))
- Fixed an error with `print_nan_gradients` when some parameters do not require gradient ([#579](https://github.com/PyTorchLightning/pytorch-lightning/pull/579))
- Fixed a bug where the progress bar would show an incorrect number of total steps during the validation sanity check when using multiple validation data loaders ([#597](https://github.com/PyTorchLightning/pytorch-lightning/pull/597))
- Fixed support for PyTorch 1.1.0 ([#552](https://github.com/PyTorchLightning/pytorch-lightning/pull/552))
- Fixed an issue with early stopping when using a `val_check_interval < 1.0` in `Trainer` ([#492](https://github.com/PyTorchLightning/pytorch-lightning/pull/492))
- Fixed bugs relating to the `CometLogger` object that would cause it to not work properly ([#481](https://github.com/PyTorchLightning/pytorch-lightning/pull/481))
- Fixed a bug that would occur when returning `-1` from `on_batch_start` following an early exit or when the batch was `None` ([#509](https://github.com/PyTorchLightning/pytorch-lightning/pull/509))
- Fixed a potential race condition with several processes trying to create checkpoint directories ([#530](https://github.com/PyTorchLightning/pytorch-lightning/pull/530))
- Fixed a bug where batch 'segments' would remain on the GPU when using `truncated_bptt > 1` ([#532](https://github.com/PyTorchLightning/pytorch-lightning/pull/532))
- Fixed a bug when using `IterableDataset` ([#547](https://github.com/PyTorchLightning/pytorch-lightning/pull/547))
- Fixed a bug where `.item` was called on non-tensor objects ([#602](https://github.com/PyTorchLightning/pytorch-lightning/pull/602))
- Fixed a bug where `Trainer.train` would crash on an uninitialized variable if the trainer was run after resuming from a checkpoint that was already at `max_epochs` ([#608](https://github.com/PyTorchLightning/pytorch-lightning/pull/608))
- Fixed a bug where early stopping would begin two epochs early ([#617](https://github.com/PyTorchLightning/pytorch-lightning/pull/617))
- Fixed a bug where `num_training_batches` and `num_test_batches` would sometimes be rounded down to zero ([#649](https://github.com/PyTorchLightning/pytorch-lightning/pull/649))
- Fixed a bug where an additional batch would be processed when manually setting `num_training_batches` ([#653](https://github.com/PyTorchLightning/pytorch-lightning/pull/653))
- Fixed a bug when batches did not have a `.copy` method ([#701](https://github.com/PyTorchLightning/pytorch-lightning/pull/701))
- Fixed a bug when using `log_gpu_memory=True` in Python 3.6 ([#715](https://github.com/PyTorchLightning/pytorch-lightning/pull/715))
- Fixed a bug where checkpoint writing could exit before completion, giving incomplete checkpoints ([#689](https://github.com/PyTorchLightning/pytorch-lightning/pull/689))
- Fixed a bug where `on_train_end` was not called when ealy stopping ([#723](https://github.com/PyTorchLightning/pytorch-lightning/pull/723))

## [0.5.3] - 2019-11-06

### Added

- Added option to disable default logger, checkpointer, and early stopping by passing `logger=False`, `checkpoint_callback=False` and `early_stop_callback=False` respectively
- Added `CometLogger` for use with Comet.ml
- Added `val_check_interval` argument to `Trainer` allowing validition to be performed at every given number of batches
- Added functionality to save and load hyperparameters using the standard checkpoint mechanism
- Added call to `torch.cuda.empty_cache` before training starts
- Added option for user to override the call t `backward`
- Added support for truncated backprop through time via the `truncated_bptt_steps` argument in `Trainer`
- Added option to operate on all outputs from `training_step` in DDP2
- Added a hook for modifying DDP init
- Added a hook for modifying Apex

### Changed

- Changed experiment version to be padded with zeros (e.g. `/dir/version_9` becomes `/dir/version_0009`)
- Changed callback metrics to include any metrics given in logs or progress bar
- Changed the default for `save_best_only` in `ModelCheckpoint` to `True`
- Added `tng_data_loader` for backwards compatibility
- Renamed `MLFlowLogger.client` to `MLFlowLogger.experiment` for consistency
- Moved `global_step` increment to happen after the batch has been processed
- Changed weights restore to first attempt HPC weights before restoring normally, preventing both weights being restored and running out of memory
- Changed progress bar functionality to add multiple progress bars for train/val/test
- Changed calls to `print` to use `logging` instead

### Deprecated

- Deprecated `tng_dataloader`

### Fixed

- Fixed an issue where the number of batches was off by one during training
- Fixed a bug that occured when setting a ckeckpoint callback and `early_stop_callback=False`
- Fixed an error when importing CometLogger
- Fixed a bug where the `gpus` argument had some unexpected behaviour
- Fixed a bug where the computed total number of batches was sometimes incorrect
- Fixed a bug where the progress bar would sometimes not show the total number of batches in test mode
- Fixed a bug when using the `log_gpu_memory='min_max'` option in `Trainer`
- Fixed a bug where checkpointing would sometimes erase the current directory

## [0.5.2] - 2019-10-10

### Added

- Added `weights_summary` argument to `Trainer` to be set to `full` (full summary), `top` (just top level modules) or other
- Added `tags` argument to `MLFlowLogger`

### Changed

- Changed default for `amp_level` to `O1`

### Removed

- Removed the `print_weights_summary` argument from `Trainer`

### Fixed

- Fixed a bug where logs were not written properly
- Fixed a bug where `logger.finalize` wasn't called after training is complete
- Fixed callback metric errors in DDP
- Fixed a bug where `TestTubeLogger` didn't log to the correct directory

## [0.5.1] - 2019-10-05

### Added

- Added the `LightningLoggerBase` class for experiment loggers
- Added `MLFlowLogger` for logging with `mlflow`
- Added `TestTubeLogger` for logging with `test_tube`
- Added a different implementation of DDP (`distributed_backed='ddp2'`) where every node has one model using all GPUs
- Added support for optimisers which require a closure (e.g. LBFGS)
- Added automatic `MASTER_PORT` defualt for DDP when not set manually
- Added new GPU memory logging options `'min_max'` (log only the min/max utilization) and `'all'` (log all the GPU memory)

### Changed

- Changed schedulers to always be called with the current epoch
- Changed `test_tube` to an optional dependency
- Changed data loaders to internally use a getter instead of a python property
- Disabled auto GPU loading when restoring weights to prevent out of memory errors
- Changed logging, early stopping and checkpointing to occur by default

### Fixed

- Fixed a bug with samplers that do not specify `set_epoch`
- Fixed a bug when using the `MLFlowLogger` with unsupported data types, this will now raise a warning
- Fixed a bug where gradient norms were alwasy zero using `track_grad_norm`
- Fixed a bug which causes a crash when logging memory

## [0.5.0] - 2019-09-26

### Changed

- Changed `data_batch` argument to `batch` throughout
- Changed `batch_i` argument to `batch_idx` throughout
- Changed `tng_dataloader` method to `train_dataloader`
- Changed `on_tng_metrics` method to `on_training_metrics`
- Changed `gradient_clip` argument to `gradient_clip_val`
- Changed `add_log_row_interval` to `row_log_interval`

### Fixed

- Fixed a bug with tensorboard logging in multi-gpu setup

## [0.4.9] - 2019-09-16

### Added

- Added the flag `log_gpu_memory` to `Trainer` to deactivate logging of GPU memory utilization
- Added SLURM resubmit functionality (port from test-tube)
- Added optional weight_save_path to trainer to remove the need for a checkpoint_callback when using cluster training
- Added option to use single gpu per node with `DistributedDataParallel`

### Changed

- Changed functionality of `validation_end` and `test_end` with multiple dataloaders to be given all of the dataloaders at once rather than in seperate calls
- Changed print_nan_grads to only print the parameter value and gradients when they contain NaN
- Changed gpu API to take integers as well (e.g. `gpus=2` instead of `gpus=[0, 1]`)
- All models now loaded on to CPU to avoid device and out of memory issues in PyTorch

### Fixed

- Fixed a bug where data types that implement `.to` but not `.cuda` would not be properly moved onto the GPU
- Fixed a bug where data would not be re-shuffled every epoch when using a `DistributedSampler`

## [0.4.8] - 2019-08-31

### Added

- Added `test_step` and `test_end` methods, used when `Trainer.test` is called
- Added `GradientAccumulationScheduler` callback which can be used to schedule changes to the number of accumulation batches
- Added option to skip the validation sanity check by setting `nb_sanity_val_steps = 0`

### Fixed

- Fixed a bug when setting `nb_sanity_val_steps = 0`

## [0.4.7] - 2019-08-24

### Changed

- Changed the default `val_check_interval` to `1.0`
- Changed defaults for `nb_val_batches`, `nb_tng_batches` and `nb_test_batches` to 0

### Fixed

- Fixed a bug where the full validation set as used despite setting `val_percent_check`
- Fixed a bug where an `Exception` was thrown when using a data set containing a single batch
- Fixed a bug where an `Exception` was thrown if no `val_dataloader` was given
- Fixed a bug where tuples were not properly transfered to the GPU
- Fixed a bug where data of a non standard type was not properly handled by the trainer
- Fixed a bug when loading data as a tuple
- Fixed a bug where `AttributeError` could be suppressed by the `Trainer`

## [0.4.6] - 2019-08-15

### Added

- Added support for data to be given as a `dict` or `list` with a single gpu
- Added support for `configure_optimizers` to return a single optimizer, two list (optimizers and schedulers), or a single list

### Fixed

- Fixed a bug where returning just an optimizer list (i.e. without schedulers) from `configure_optimizers` would throw an `Exception`

## [0.4.5] - 2019-08-13

### Added

- Added `optimizer_step` method that can be overridden to change the standard optimizer behaviour

## [0.4.4] - 2019-08-12

### Added

- Added supoort for multiple validation dataloaders
- Added support for latest test-tube logger (optimised for `torch==1.2.0`)

### Changed

- `validation_step` and `val_dataloader` are now optional
- `lr_scheduler` is now activated after epoch

### Fixed

- Fixed a bug where a warning would show when using `lr_scheduler` in `torch>1.1.0`
- Fixed a bug where an `Exception` would be thrown if using `torch.DistributedDataParallel` without using a `DistributedSampler`, this now throws a `Warning` instead

## [0.4.3] - 2019-08-10

### Fixed

- Fixed a bug where accumulate gradients would scale the loss incorrectly

## [0.4.2] - 2019-08-08

### Changed

- Changed install requirement to `torch==1.2.0`

## [0.4.1] - 2019-08-08

### Changed

- Changed install requirement to `torch==1.1.0`

## [0.4.0] - 2019-08-08

### Added

- Added 16-bit support for a single GPU
- Added support for training continuation (preserves epoch, global step etc.)

### Changed

- Changed `training_step` and `validation_step`, outputs will no longer be automatically reduced

### Removed

- Removed need for `Experiment` object in `Trainer`

### Fixed

- Fixed issues with reducing outputs from generative models (such as images and text)

## [0.3.6] - 2019-07-25

### Added

- Added a decorator to do lazy data loading internally

### Fixed

- Fixed a bug where `Experiment` object was not process safe, potentially causing logs to be overwritten

## [0.3.5] - 2019-07-25

## [0.3.4] - 2019-07-22

## [0.3.3] - 2019-07-22

## [0.3.2] - 2019-07-21

## [0.3.1] - 2019-07-21

## [0.2.x] - 2019-07-09

## [0.1.x] - 2019-06-DD
