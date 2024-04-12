from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import EarlyStopping, LinearCyclicalScheduler, ModelCheckpoint, global_step_from_engine

class TrainingManager:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, epochs, lr):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr

        self.setup_metrics()
        self.setup_trainer()
        self.setup_evaluator()
        self.setup_handlers()

    def setup_metrics(self):
        self.metrics = {"loss": Loss(self.criterion)}

    def setup_trainer(self):
        self.trainer = create_supervised_trainer(self.model, self.optimizer, self.criterion, device=self.device)

    def setup_evaluator(self):
        self.evaluator = create_supervised_evaluator(self.model, metrics=self.metrics, device=self.device)

    def setup_handlers(self):
        @self.trainer.on(Events.ITERATION_STARTED)
        def move_model_to_gpu(engine):
            self.model.to(self.device)

        @self.trainer.on(Events.EPOCH_STARTED(every=10))
        def print_epoch(engine):
            epoch = engine.state.epoch
            print(f"Epoch: {epoch}")

        @self.trainer.on(Events.EPOCH_COMPLETED(every=10))
        def log_training_results(engine):
            epoch = engine.state.epoch
            avg_loss = engine.state.output
            print(f"Training Results - Avg loss: {avg_loss:.2f}")

        @self.evaluator.on(Events.COMPLETED)
        def log_validation_results(engine):
            epoch = self.trainer.state.epoch
            if epoch % 10 == 0:
                metrics = self.evaluator.state.metrics
                print(f"Validation Results - Avg loss: {metrics['loss']:.2f}")

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def run_evaluator(engine):
            self.evaluator.run(self.val_loader)  # Run evaluator on every epoch

        # self.trainer.add_handler(ProgressBar(output_transform=lambda x: {'batch loss': x}))

        def score_function(engine):
            val_loss = engine.state.metrics['loss']
            return -val_loss  # Minimize MSE for EarlyStopping

        self.es = EarlyStopping(patience=100, score_function=score_function, trainer=self.trainer)
        self.evaluator.add_event_handler(Events.COMPLETED, self.es)

        self.best_model_saver = ModelCheckpoint(
            dirname='saved_models',
            filename_prefix='cnn',
            n_saved=1,
            create_dir=True,
            require_empty=False,
            score_function=score_function,
            global_step_transform=global_step_from_engine(self.trainer),
        )
        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.best_model_saver, {'model': self.model})

        self.scheduler = LinearCyclicalScheduler(self.optimizer, 'lr', start_value=self.lr, end_value=0, cycle_size=20)
        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.scheduler)

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=self.epochs)



