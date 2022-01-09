# reinforcement-learning-project
Reinforcement Learning - AI Projects #7

This repository includes a drone environment that does not include any obstacles, and the aim of the drone is to fly to an arbitrary target and hover. It contains the Soft Actor Critic and Hindsight Experience Replay algorithms to train the drone.

To run the code, first download the base drone environment, which can be [found here.](https://github.com/utiasDSL/gym-pybullet-drones.git) Then, to run the training code modify the ```main.py``` file that includes all hyperparameters and the main training loop. 

To start training run
```bash
$ python main.py
```
For GPU add ```--cuda``` attribute.

This repository supports tensorboard for logging the values.

The eval folder contains code for evaluating the trained agent visually and extract a dataset for offline learning/imitation learning for RL. The previously created dataset can also downloaded [from here](https://drive.google.com/file/d/1UjY-G1cZcpEP6ibJ-282XjJzvC9NO5N2/view?usp=sharing). It includes a checkpoint from the previous trainings to run the model.

To start the PyBullet simulation window
```bash
$ python eval.py
```

## References
- Haarnoja, T., Zhou, A., Abbeel, P. &amp; Levine, S.. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. <i>Proceedings of the 35th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 80:1861-1870 Available from https://proceedings.mlr.press/v80/haarnoja18b.html.
- Andrychowicz, M., Crow, D., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017). Hindsight Experience Replay. NIPS.
- Panerati, J., Zheng, H., Zhou, S., Xu, J., Prorok, A., Studies, A.P., Intelligence, V.I., & Cambridge, U.O. (2021). Learning to Fly—a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control. 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 7512-7519.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. ArXiv, abs/2106.01345.