import sys
import torch
import ignite
from collections import namedtuple
from ignite.engine import Events
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

# Load experiment ingredients and their respective configs.
from dataset.sacred import get_dataset_ingredient
from model.sacred import model, init_model
from training.handlers import Tracer
from training.sacred import training, set_seed_and_device, \
                            setup_training, run_training


Config = namedtuple('Config', ['name', 'config'], defaults=[None, {}])

def update_configs_(ingredient, configs):
    for name, config_info in configs:
        if name is None:
            ingredient.add_config(config_info)
        else:
            ingredient.add_named_config(name, config_info)


def create_experiment(task, name, dataset_configs, training_configs, 
                    model_configs, observers, experiment_configs=None):

    dataset, load_dataset = get_dataset_ingredient(task)
    
    # Create experiment
    ex = Experiment(
        name=name,
        ingredients=[dataset, model, training]
    )

    update_configs_(dataset, dataset_configs)
    update_configs_(training, training_configs)
    update_configs_(model, model_configs)

    if experiment_configs is not None:
        update_configs_(ex, experiment_configs)

    # Runtime options
    save_folder = '../../data/sims/deladd/temp/'
    ex.add_config({
        'no_cuda': False,
    })

    # Add dependencies
    ex.add_source_file('../../src/model/subLSTM/nn.py')
    ex.add_source_file('../../src/model/subLSTM/functional.py')
    ex.add_package_dependency('torch', torch.__version__)
    ex.observers.extend(observers)

    def _log_training(tracer):
        ex.log_scalar('training_loss', tracer.trace[-1])
        tracer.trace.clear()


    def _log_validation(engine):
        for metric, value in engine.state.metrics.items():
            ex.log_scalar('val_{}'.format(metric), value)

    def _run_experiment(_config, seed):
        no_cuda = _config['no_cuda']
        batch_size = _config['training']['batch_size']

        device = set_seed_and_device(seed, no_cuda)
        training_set, test_set, validation_set = load_dataset(
                                                    batch_size=batch_size)
        model = init_model(device=device)

        trainer, validator, checkpoint, metrics = setup_training(
            model, validation_set,
            save=save_folder, device=device,
            trace=False, time=False)[:4]

        tracer = Tracer().attach(trainer)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, lambda e: _log_training(tracer))
        validator.add_event_handler(Events.EPOCH_COMPLETED, _log_validation)

        test_metrics = run_training(
            model=model,
            train_data=training_set,
            trainer=trainer,
            test_data=test_set,
            metrics=metrics,
            model_checkpoint=checkpoint,
            device=device
        )

        # save best model performance and state
        for metric, value in test_metrics.items():
            ex.log_scalar('test_{}'.format(metric), value)

        ex.add_artifact(str(checkpoint._saved[-1][1][0]), 'trained-model')

    return ex, _run_experiment
