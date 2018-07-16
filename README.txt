实验计划：

state: 10: 3 position, 3 linear velocity, 4 angular velocity
action: 3 position control
goal: 3 position


实验1：    训练：
          输入 state + next_state
          输出 action
          测试：
          输入 state + goal
          输出 action
结果1：    对于 fetch_reacher 任务来说效果不错

实验2：    训练：
          输入 state + goal
          输出 action
          测试：
          输入 state + goal
          输出 action
结果2：    对于 fetch_pick_and_place 任务来说效果不错

实验3：    在实验2上减少训练的样本数量


问题1：怎么让 random 采集到的数据 make sense ?!!
答案1：手动编写控制代码

问题2：训练得到的 policy 的成功率为0
答案2：让原来的回归任务改成分类任务
      训练4个分类器，其中3个是3分类任务，一个是2分类任务

      上面的答案不对！！
      正确的原因是提供的特征的维度太多，造成了维度爆炸。
      解决办法是把原来的长度为25的observation缩短到长度为6

      结果：
      无论是分类任务还是回归任务，准确率都可以接近100%

问题3：如何从gym中得到图片
答案3：
        1. 设置render的mode参数
            env.render(mode='rgb_array')
        2. 去除多余的显示信息
            ~/Documents/mujoco-py/mujoco_py/mjviewer.py中
           class MjViewer 中
           self._hide_overlay = True
        3. 更改图片的大小
            ~/anaconda3/envs/tf-cpu/lib/python3.6/site-packages/gym/envs/mujoco/mujoco_env.py中
           class:MujocoEnv function:render中
           width, height = 1744, 992



注意注意！！！！
    之前的代码 “forward-consistent-feature-reduced-GSP.py” 有错误
    在 train 函数的 next_state_feed 的获取中，我应该把 j+36 而不是加 32 ！！！！


问题4：如果模型A包含模型B，如何在模型A中加载模型B预训练好的权重？

    我的一个担忧是：如果在模型A中直接调用模型B，（即把模型B当做一个层来调用）那么在模型A中加载模型B的权重会成功吗？
    更保险地做法是：在模型A中加入和模型B一模一样的层，而不是直接调用B。



问题5：  用两个LSTM来实现 auto-encoder 存在的问题

    decoder可能什么也没学到

    有两个解决办法：
    1,decoder不用LSTM
    2,对状态序列进行降采样


gym 里面的一个bug是：
    when using render(), the firs time the claw is very low
    but if not using render(), the problem is gone


之前犯了一个非常愚蠢地错误是在我的第一个版本的top-sub policy 中，我加载权重的时候设置了 by_name = True, 导致权重加载不正确

random policy:
    没有解决：
        物体从垫片上掉下来的情况
        垫片立起来的情况
        物体或垫片从桌上掉下里的情况

    抓物体
        物体没有在目标点 物体没被抓住 物体不在爪子xy范围 爪子不在目标高度1
        raise_claw_up

        物体没有在目标点 物体没被抓住 物体不在爪子xy范围 爪子在目标高度1
        reach_object_above

        物体没有在目标点 物体没被抓住 物体在爪子xy范围 物体不在爪子z范围
        reach_object

        物体没有在目标点 物体没被抓住 物体在爪子xy范围 物体在爪子z范围
        grasp_object

    放物体
        物体没有在目标点 物体被抓住 物体不在目标xy范围 物体不在目标高度1
        raise_object_up

        物体没有在目标点 物体被抓住 物体不在目标xy范围 物体到达目标高度1
        reach_target_above

        物体没有在目标点 物体被抓住 物体在目标xy范围 物体未到达目标高度2
        lower_object

        物体没有在目标点 物体被抓住 物体在目标xy范围 物体到达目标高度2
        release_object

    升爪子
        物体在目标点 爪子不在目标高度1
        raise_claw_up

    结束
        物体在目标点 爪子在目标高度1