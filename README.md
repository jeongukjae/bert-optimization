# bert-optimization

![Test Python Status](https://github.com/jeongukjae/bert-optimization/workflows/Test%20Python/badge.svg)

Test repository to optimize BERT

## Fine Tuning

### CoLA

```
2020-04-27 19:22:17,641: [INFO] Training Parameters
2020-04-27 19:22:17,642: [INFO]  - model: tmp/bert_model.ckpt
2020-04-27 19:22:17,642: [INFO]  - config: tmp/bert_config.json
2020-04-27 19:22:17,642: [INFO]  - output: tmp
2020-04-27 19:22:17,642: [INFO]  - dataset: tmp/cola/
2020-04-27 19:22:17,642: [INFO]  - vocab: tmp/vocab.txt
2020-04-27 19:22:17,642: [INFO]  - task: cola
2020-04-27 19:22:17,642: [INFO]  - use_gpu: False
2020-04-27 19:22:17,642: [INFO]  - epoch: 3
2020-04-27 19:22:17,642: [INFO]  - learning_rate: 4e-05
2020-04-27 19:22:17,642: [INFO]  - weight_decay: 0.01
2020-04-27 19:22:17,642: [INFO]  - warmup_ratio: 0.1
2020-04-27 19:22:17,642: [INFO]  - do_lower_case: False
2020-04-27 19:22:17,642: [INFO]  - max_sequence_length: 128
2020-04-27 19:22:17,642: [INFO]  - eval_batch_size: 128
2020-04-27 19:22:17,642: [INFO]  - train_batch_size: 32
2020-04-27 19:22:17,642: [INFO]  - warmup_rate: 0.1
2020-04-27 19:22:17,642: [INFO]  - log_interval: 50
2020-04-27 19:22:17,642: [INFO]  - val_interval: 50
```

```
2020-04-27 19:22:19,311: [INFO] Train Dataset Size: 8551
2020-04-27 19:22:19,312: [INFO] Dev Dataset Size: 1043
2020-04-27 19:22:19,312: [INFO] Train Batches: 268
2020-04-27 19:22:19,312: [INFO] Dev Batches: 9
2020-04-27 19:22:24,742: [INFO] Initialize model
2020-04-27 19:22:24,742: [INFO] Model Config
2020-04-27 19:22:24,742: [INFO]  - attention_probs_dropout_prob: 0.1
2020-04-27 19:22:24,742: [INFO]  - hidden_act: gelu
2020-04-27 19:22:24,742: [INFO]  - hidden_dropout_prob: 0.1
2020-04-27 19:22:24,742: [INFO]  - hidden_size: 768
2020-04-27 19:22:24,742: [INFO]  - intermediate_size: 3072
2020-04-27 19:22:24,742: [INFO]  - max_position_embeddings: 512
2020-04-27 19:22:24,742: [INFO]  - num_attention_heads: 12
2020-04-27 19:22:24,742: [INFO]  - num_hidden_layers: 12
2020-04-27 19:22:24,742: [INFO]  - type_vocab_size: 2
2020-04-27 19:22:24,742: [INFO]  - vocab_size: 30522
2020-04-27 19:22:24,743: [INFO]  - output_hidden_states: True
2020-04-27 19:22:24,743: [INFO]  - output_embedding: True
2020-04-27 19:22:24,743: [INFO]  - use_splitted: False
````

```
2020-04-27 19:28:01,138: [INFO] Epoch 2 step: 50, loss: 0.20514549314975739, Acc: 0.9256250262260437, MCC: 0.8203318119049072
2020-04-27 19:28:05,089: [INFO] [Eval] Epoch 2 step: 50 loss: 0.5271073579788208, Acc: 0.8005752563476562, MCC: 0.5131129026412964
2020-04-27 19:28:05,091: [INFO] Reached Best Score.
2020-04-27 19:28:05,587: [INFO] Saved model in tmp/checkpoints/model-cola-0.5131-0.8006-epoch2-step586
2020-04-27 19:28:28,212: [INFO] Epoch 2 step: 100, loss: 0.19740892946720123, Acc: 0.9306250214576721, MCC: 0.8248364925384521
2020-04-27 19:28:32,130: [INFO] [Eval] Epoch 2 step: 100 loss: 0.6247080564498901, Acc: 0.794822633266449, MCC: 0.48564839363098145
2020-04-27 19:28:54,963: [INFO] Epoch 2 step: 150, loss: 0.18384340405464172, Acc: 0.9293749928474426, MCC: 0.832387387752533
2020-04-27 19:28:58,892: [INFO] [Eval] Epoch 2 step: 150 loss: 0.547747790813446, Acc: 0.8101630210876465, MCC: 0.5324115753173828
2020-04-27 19:28:58,893: [INFO] Reached Best Score.
2020-04-27 19:28:59,356: [INFO] Saved model in tmp/checkpoints/model-cola-0.5324-0.8102-epoch2-step686
2020-04-27 19:29:21,945: [INFO] Epoch 2 step: 200, loss: 0.20011669397354126, Acc: 0.9287499785423279, MCC: 0.8275661468505859
2020-04-27 19:29:25,896: [INFO] [Eval] Epoch 2 step: 200 loss: 0.54879230260849, Acc: 0.8072866797447205, MCC: 0.5226700305938721
2020-04-27 19:29:48,895: [INFO] Epoch 2 step: 250, loss: 0.19881491363048553, Acc: 0.9268749952316284, MCC: 0.8256911039352417
2020-04-27 19:29:52,894: [INFO] [Eval] Epoch 2 step: 250 loss: 0.5452569127082825, Acc: 0.8063278794288635, MCC: 0.5193025469779968
2020-04-27 19:30:01,117: [INFO] Epoch 2 loss: 0.1735537201166153, Acc: 0.9183303117752075, MCC: 0.799972414970398
2020-04-27 19:30:05,061: [INFO] [Eval] Epoch 2 step: 268 loss: 0.5461156368255615, Acc: 0.8063278794288635, MCC: 0.5190938115119934
```
