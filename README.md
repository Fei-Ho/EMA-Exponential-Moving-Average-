# EMA(Exponential Moving Average)
When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.
### Create an EMA instance
ema = EMA(model,0.999)
### Create shadow parameters of trainable parameters of model
ema.register()
### Update the value of shadow parameters after the change of trainable parameters (This is usually used in the training loop of the model).
ema.update()
### Set the parameters of the model as shadow parameters
ema.apply_shadow()
### Set the parameters of the model to actual values
ema.restore()
