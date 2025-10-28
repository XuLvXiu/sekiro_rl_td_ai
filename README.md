
## 动机与成效



## 演示视频

三周目白金的时序差分 Sarsa 狼 再战 稀世强者弦一郎

https://www.bilibili.com/video//

## 游戏设置

- steam 开始游戏
- 游戏设置：图像设定 -- 屏幕模式设置为窗口，游戏分辨率调整为1280*720
- 游戏设置：按键设置 -- ‘使用道具‘ 设置为按键 p, 动作（长按）吸引设置为按键 e. 重置视角/固定目标设置为 q，跳跃键为 f(未用到)，垫步/识破键为空格，鼠标左键攻击，右键防御
- 把 '伤药葫芦' 设置为第一个快捷使用的道具, 最好只设置这一个道具
- 游戏设置：网络设定 -- 标题画面启动设定为 离线游玩.  目的是为了关闭残影，满地的残影烦死了。
- 保持游戏窗口在最上层，不要最小化或者被其它的全屏窗口覆盖。

## 思路与问题

image - class(不准确) - state(相对准确) - action(需要探索)

像是在做特征工程 + grid search

由于一个招式有若干帧，同一招式中连续多个帧的分类应该是相同的，在预测时，连续帧相同分类出现的次数可以称为“信号强度”，信号越强，表示对该招式的预测越准确。

危机危机，危里面含着机.

最后，设计了两个新的 state：8 与 9。  state-8 指的是 player 受到了伤害； state-9 指的是 boss 受到了伤害且 player hp < 60，此时应该去喝血瓶。

从最终结果来看，这种状态定义方式也并不见得有多合理，state 未能完整反映出游戏中的真实情况。


## 如何训练

- 确认环境

`python debug_display_game_info.py`

会在 assets 目录中生成 debug_ui_elements.jpg，该图片中会绘制游戏屏幕截图中的各个 window 区域

同时还会弹出一个小的 tk 窗口实时显示 player 与 boss 的 hp. 这个功能得感谢原作者。


- 收集数据

`python data_collector.py`

按 ] 键开始收集; 

所谓的收集，就是在使用分类模型与 MC policy 自动打游戏的过程中，定期对游戏屏幕的 boss 区域进行截屏(301x301)，以 list 的形式保存在内存中。

一个 episode 结束的时候，内存中的数据集会被保存到硬盘文件中。

按 Backspace 键，退出程序

如果在命令行中使用了 `--new` 参数，会首先清除 images 目录

也可以不必收集，直接把图片分类项目中收集的图片文件 copy 到 images 目录中也可以，图片分类项目中的图片是人工打出来的，可能效果会更好一些吧。


- 训练聚类模型

`python cluster_model.py` 

会结合图片分类项目中训练出来的 resnet 模型与 kmeans，训练出一个 8 分类的聚类模型 model.cluster.kmeans

resnet 负责提取 images 目录中各个图片的特征，kmeans 对这些特征进行聚类，最后会把各个分类的图片 copy 一份到 images/1 image/2 等子目录中。

在我们这次训练的过程中，images/7 里面是大量的各种危的图片，images/2 里面也有少量的危。


- 训练 MC policy

`python train.py`

默认会加载 checkpoint 文件中的训练相关信息以及Q和N，然后在此基础上进行训练。

进入游戏后，按 q 键锁定 boss 之后，按 ] 键开始正式的训练。

如果在命令行中使用了 `--new` 参数，会从第 0 个 episode 开始重新训练。

每一个 episode 结束之后，更新 Q 与 N 并保存到 checkpoint 文件中。


- 查看 Q与 N

`python checkpoint.py`


- 测试：执行某个或者某几个动作

`python test.py`


## 预测

进入游戏，

在 cmd 窗口中运行：
```
python main.py 
```

等待模型加载完，

按 q 键锁定敌方

按 ] 键, 就会针对敌方的出招自动做出预测动作了。

再次按下 ] 键，会停止预测。

按 Backspace 键，退出程序。


## 人工备份

模型的训练结果主要涉及到如下的几个文件：
- images	            收集到的截屏图像文件
- checkpoint.json		记录了当前是哪个episode，以及完成训练时的时间。
- checkpoint.pkl		存储了 Q 与 N
- model.cluster.kmeans  聚类模型
- model.resnet.v1  图片分类项目的训练成果


如果要训练新模型的话，可能需要对老模型的这些数据进行备份。


## 大部分代码和思路来自以下网址，感谢他们

- https://github.com/XR-stb/DQN_WUKONG
- https://github.com/analoganddigital/DQN_play_sekiro
- https://github.com/Sentdex/pygta5
- https://github.com/RongKaiWeskerMA/sekiro_play

- https://www.lapis.cafe/posts/ai-and-deep-learning/%E4%BD%BF%E7%94%A8resnet%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E5%9B%BE%E7%89%87%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B
- https://blog.csdn.net/qq_36795658/article/details/100533639
- https://blog.csdn.net/Guo_Python/article/details/134922730

- 图片分类项目 https://github.com/XuLvXiu/sekiro_classifier_ai
- 蒙特卡洛项目 https://github.com/XuLvXiu/sekiro_rl_mc_ai
