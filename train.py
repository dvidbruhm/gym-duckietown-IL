import numpy as np
from tqdm import tqdm

from _loggers import Reader
from model import TensorflowModel

from pytorch_model import PytorchTrainer

# configuration zone
BATCH_SIZE = 64
EPOCHS = 50
# here we assume the observations have been resized to 60x80
OBSERVATIONS_SHAPE = (None, 60, 80, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 2345
STORAGE_LOCATION = "trained_models/behavioral_cloning"

reader = Reader('train.log')

observations, actions = reader.read()
actions = np.array(actions)
observations = np.array(observations)

print("Nb of data : ", len(observations))

model = TensorflowModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    graph_location=STORAGE_LOCATION,  # where do we want to store our trained models
    seed=SEED  # to seed all random operations in the model (e.g., dropout)
)


trainer = PytorchTrainer()

min_loss = 10000

# we trained for EPOCHS epochs
epochs_bar = tqdm(range(EPOCHS))
for i in epochs_bar:
    # we defined the batch size, this can be adjusted according to your computing resources...
    loss = None
    loss2 = None
    for batch in range(0, len(observations), BATCH_SIZE):
        #loss2 = model.train(
        #    observations=observations[batch:batch + BATCH_SIZE],
        #    actions=actions[batch:batch + BATCH_SIZE]
        #)

        observation_batch = observations[batch:batch + BATCH_SIZE]
        action_batch = actions[batch:batch + BATCH_SIZE]
        loss = trainer.train(observation_batch, action_batch)

    epochs_bar.set_postfix({
        'loss': loss,
        "loss_tf": loss2
        })

    # every 10 epochs, we store the model we have
    # but I'm sure that you're smarter than that, what if this model is worse than the one we had before
    if loss < min_loss:
        min_loss = loss
        model.commit()
        trainer.save()
        epochs_bar.set_description('Model saved...')
    else:
        epochs_bar.set_description('')

# the loss at this point should be on the order of 2e-2, which is far for great, right?

# we release the resources...
#model.close()
reader.close()

