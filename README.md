# Example of Domain Randomization in Mujoco Environment + Gymnasium.
A simple domain randomization using gymnasium + mujoco. The environment is a simple ball agent but random weights are attache on the body.

The domain randomization includes
```
initial position
friction coefficient
lighting condition
floor texture
agent color
agent size
sensor position (imu)
number of attached weights
positions of attached weights
```


```bash
pip install -e .
```

run and see the result

```bash
python sample.py
```
