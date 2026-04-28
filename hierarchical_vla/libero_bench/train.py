import logging
import os
import sys
import warnings
from omegaconf import DictConfig, OmegaConf
import hydra

# Suppress all warnings BEFORE any other imports
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Suppress PyTorch pin_memory deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*pin_memory.*device.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*is_pinned.*device.*')

# Redirect stderr to suppress robosuite/gym warnings that print directly
# Save original stderr
_original_stderr = sys.stderr

class FilteredStderr:
    """Filter stderr to suppress unwanted warnings."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, text):
        # Filter out robosuite and gym warnings
        if any(keyword in text for keyword in [
            '[robosuite WARNING]',
            'No private macro file found',
            'Gym has been unmaintained',
            'upgrade to Gymnasium',
            'The parameter.*pretrained.*is deprecated',
            'Scope.user.*setter is deprecated',
            'robosuite WARNING',
            'Gymnasium',
            'pin_memory() is deprecated',
            'is_pinned() is deprecated',
            'The argument \'device\' of Tensor.pin_memory()',
            'The argument \'device\' of Tensor.is_pinned()'
        ]):
            return  # Suppress these messages
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Apply stderr filter BEFORE any imports that might trigger warnings
sys.stderr = FilteredStderr(_original_stderr)

# Now import other modules
import pickle
import torch
from torch.utils.data import ConcatDataset, Dataset

from .dataloader import LiberoDataset, sim_framework_path
from MambaVLA import train_policy
from .libero_sim import MultiTaskSim

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CombinedLiberoDataset(Dataset):
    """
    Wrapper around ConcatDataset that preserves camera_names attribute
    needed for model creation.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.concat_dataset = ConcatDataset(datasets)
        # Get camera_names from first dataset (all should have the same)
        if len(datasets) > 0 and hasattr(datasets[0], 'camera_names'):
            self.camera_names = datasets[0].camera_names
        else:
            self.camera_names = ['agentview', 'eye_in_hand']  # Default
    
    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, idx):
        return self.concat_dataset[idx]
    
    def get_all_actions(self):
        """Get all actions from all datasets."""
        all_actions = []
        for dataset in self.datasets:
            if hasattr(dataset, 'get_all_actions'):
                all_actions.append(dataset.get_all_actions())
        if all_actions:
            return torch.cat(all_actions, dim=0)
        return None


def create_eval_callback(
    data_directory: str,
    benchmark_type: str = 'libero_object',
    all_suites: bool = False,
    num_rollouts: int = 5,
    max_steps: int = 400,
    device: str = None,
    use_multiprocessing: bool = False,
    n_cores: int = 4,
    render_image: bool = False,
    save_video: bool = False,
    save_video_dir: str = None,
    seed: int = 42,
    demos_per_task: int = 1,
):
    """
    Create an evaluation callback function for use during training.
    
    Args:
        all_suites: If True, evaluate on all suites one by one
    
    Returns:
        Callback function(model, epoch) that runs evaluation
    """
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Determine which suites to evaluate on
    if all_suites:
        suite_names = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10', 'libero_90']
        # Determine base directory
        dir_basename = os.path.basename(data_directory)
        if dir_basename in suite_names:
            base_dir = os.path.dirname(data_directory)
        else:
            base_dir = data_directory
        benchmark_types = [os.path.join(base_dir, suite) for suite in suite_names]
        # Filter to only existing directories
        benchmark_types = [bt for bt in benchmark_types if os.path.exists(bt)]
        if not benchmark_types:
            logger.warning("No valid suite directories found for evaluation. Skipping evaluation.")
            return None
        # Extract just the suite names for benchmark_type parameter
        benchmark_type_names = [os.path.basename(bt) for bt in benchmark_types]
    else:
        # For single suite evaluation, construct the proper path
        dir_basename = os.path.basename(data_directory)
        suite_names = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10', 'libero_90']
        if dir_basename in suite_names:
            # data_directory already points to a specific suite
            benchmark_types = [data_directory]
            benchmark_type_names = [benchmark_type]
        else:
            # data_directory is a parent directory, construct path from benchmark_type
            suite_path = os.path.join(data_directory, benchmark_type)
            if os.path.exists(suite_path):
                benchmark_types = [suite_path]
                benchmark_type_names = [benchmark_type]
            else:
                # Fallback: try using data_directory directly
                benchmark_types = [data_directory]
                benchmark_type_names = [benchmark_type]
    
    # Load task embeddings for all suites
    task_emb_dir = sim_framework_path("language_embeddings")
    if not os.path.exists(task_emb_dir):
        mambavla_emb_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../MambaVLA/language_embeddings')
        )
        if os.path.exists(mambavla_emb_dir):
            task_emb_dir = mambavla_emb_dir
    
    task_embs_dicts = {}
    for suite_dir, suite_name in zip(benchmark_types, benchmark_type_names):
        task_emb_path = os.path.join(task_emb_dir, suite_name + ".pkl")
        if os.path.exists(task_emb_path):
            with open(task_emb_path, 'rb') as f:
                task_embs_dict = pickle.load(f)
            # Convert to tensors
            task_embs_tensor = {}
            for key, emb in task_embs_dict.items():
                if isinstance(emb, torch.Tensor):
                    task_embs_tensor[key] = emb
                else:
                    task_embs_tensor[key] = torch.tensor(emb, dtype=torch.float32)
            task_embs_dicts[suite_name] = task_embs_tensor
        else:
            logger.warning(f"Task embeddings not found: {task_emb_path}. Will skip evaluation on {suite_name}.")
    
    if not task_embs_dicts:
        logger.warning("No task embeddings found for any suite. Evaluation will be skipped.")
        return None
    
    # Determine if using eye_in_hand camera (check first available suite)
    use_eye_in_hand = False
    for suite_dir in benchmark_types:
        try:
            dataset = LiberoDataset(
                data_directory=suite_dir,
                device="cpu",
                obs_dim=32,
                action_dim=7,
                state_dim=45,
                max_len_data=260,
                chunck_size=1,
                start_idx=0,
                demos_per_task=demos_per_task,
            )
            camera_names = dataset.camera_names
            use_eye_in_hand = 'eye_in_hand' in camera_names
            break
        except Exception as e:
            logger.warning(f"Failed to determine camera setup from {suite_dir}: {e}")
            continue
    
    def eval_callback(model, epoch):
        """Evaluation callback function."""
        logger.info(f"Starting evaluation at epoch {epoch}")
        if all_suites:
            logger.info(f"Evaluating on {len(benchmark_type_names)} suites: {', '.join(benchmark_type_names)}")
        else:
            logger.info(f"Evaluating on: {benchmark_type_names[0]}")
        
        # Evaluate on each suite sequentially
        for suite_dir, suite_name in zip(benchmark_types, benchmark_type_names):
            if suite_name not in task_embs_dicts:
                logger.warning(f"Skipping evaluation on {suite_name} (no task embeddings)")
                continue
            
            logger.info(f"Evaluating on suite: {suite_name}")
            
            # Create simulator for this suite
            simulator = MultiTaskSim(
                rollouts=num_rollouts,
                max_step_per_episode=max_steps,
                benchmark_type=suite_name,
                use_eye_in_hand=use_eye_in_hand,
                seed=seed,
                device=device,
                render_image=render_image,
                n_cores=n_cores,
                use_multiprocessing=use_multiprocessing,
                save_video=save_video,
                save_video_dir=save_video_dir
            )
            
            # Set task embeddings for this suite
            simulator.get_task_embs(task_embs_dicts[suite_name])
            
            # Run evaluation
            simulator.test_model(model=model, model_config=None, cpu_set=None, epoch=epoch)
            
            # Log evaluation results to wandb
            try:
                import wandb
                if wandb.run is not None:
                    # Get evaluation results from simulator
                    success_rate = simulator.success_rate
                    if hasattr(simulator, 'success') and simulator.success is not None:
                        # Calculate per-task success rates
                        num_tasks = simulator.success.shape[0] if len(simulator.success.shape) > 0 else 1
                        if len(simulator.success.shape) == 2:
                            per_task_success = simulator.success.mean(dim=1)  # Average across rollouts
                            overall_success = simulator.success.mean().item()
                            
                            # Log overall success rate for this suite
                            wandb.log({
                                f"eval/{suite_name}/overall_success_rate": overall_success,
                                "epoch": epoch
                            }, step=epoch)
                            
                            # Log per-task success rates for this suite
                            for task_id in range(num_tasks):
                                wandb.log({
                                    f"eval/{suite_name}/task_{task_id}_success_rate": per_task_success[task_id].item(),
                                    "epoch": epoch
                                }, step=epoch)
                            
                            # Log episode lengths if available
                            if hasattr(simulator, 'episode_lengths') and simulator.episode_lengths is not None:
                                episode_lengths = simulator.episode_lengths
                                valid_lengths = episode_lengths[episode_lengths > 0]
                                if len(valid_lengths) > 0:
                                    avg_episode_length = valid_lengths.float().mean().item()
                                    wandb.log({
                                        f"eval/{suite_name}/avg_episode_length": avg_episode_length,
                                        "epoch": epoch
                                    }, step=epoch)
                                    
                                    # Log per-task episode lengths
                                    for task_id in range(num_tasks):
                                        task_lengths = episode_lengths[task_id][episode_lengths[task_id] > 0]
                                        if len(task_lengths) > 0:
                                            wandb.log({
                                                f"eval/{suite_name}/task_{task_id}_avg_episode_length": task_lengths.float().mean().item(),
                                                "epoch": epoch
                                            }, step=epoch)
            except Exception as e:
                logger.warning(f"Failed to log evaluation results to wandb for {suite_name}: {e}")
            
            logger.info(f"Completed evaluation on {suite_name}")
            logger.info(f"Success rate: {simulator.success_rate:.2%}")
        
        logger.info(f"Evaluation completed at epoch {epoch}")
    
    return eval_callback


def train(
    data_directory: str,
    use_all_suites: bool = False,  # New parameter to load all LIBERO suites
    batch_size: int = 128,  # Match working version: DataLoadingConfig.train_batch_size = 256
    num_epochs: int = 500,  # Match working version: TrainingConfig.epoch = 500
    learning_rate: float = 1e-4,  # Match working version: OptimizerConfig.learning_rate = 1e-4
    device: str = None,
    image_encoder_type: str = "resnet",  # Image encoder type: "resnet" or "eagle"
    latent_dim: int = 256,  # Match working version: LATENT_DIM = 256
    embed_dim: int = 256,  # Match working version: len_embd = 256
    n_layer: int = 5,  # Match working version: MambaEncoderConfig.n_layer = 5
    d_intermediate: int = 256,  # Match working version: MambaEncoderConfig.d_intermediate = 256
    obs_tok_len: int = 2,  # Match working version: obs_tokens = 2
    action_seq_len: int = 10,  # Match working version: action_seq_len = 10
    save_dir: str = './checkpoints',
    save_freq: int = 10,  # Match working version: TrainingConfig.save_every_n_epochs = 10
    max_len_data: int = 530,  # Increased to accommodate all LIBERO suites (libero_10 needs 517, libero_90 needs 373, libero_goal needs 347)
    enable_ema: bool = True,  # Match working version: EMAConfig.if_use_ema = True
    ema_decay_rate: float = 0.995,  # Match working version: EMAConfig.decay_ema = 0.995
    enable_data_scaling: bool = True,  # Match working version: DataScalingConfig.scale_data = True
    data_scaler_type: str = "minmax",  # Match working version: DataScalingConfig.scaling_type = "minmax"
    num_workers: int = 0,  # Match working version: DataLoadingConfig.num_workers = 4
    transformer_weight_decay: float = 0.05,  # Match working version: OptimizerConfig.transformer_weight_decay = 0.05
    obs_encoder_weight_decay: float = 0.05,  # Match working version: OptimizerConfig.obs_encoder_weight_decay = 0.05
    betas: list = None,  # Match working version: OptimizerConfig.betas = [0.9, 0.9]
    sampling_steps: int = 4,  # Match working version: ModelConfig.sampling_steps = 4
    eval_during_training: int = None,
    eval_benchmark_type: str = 'libero_object',
    eval_all_suites: bool = False,
    eval_num_rollouts: int = 5,
    eval_max_steps: int = 400,
    eval_use_multiprocessing: bool = False,
    eval_n_cores: int = 4,
    wandb_project: str = "MambaVLA",  # Match working version: WandbConfig.project = "MambaVLA"
    wandb_entity: str = "sainavaneet",  # Match working version: WandbConfig.entity = "sainavaneet"
    wandb_name: str = "new_model",
    demos_per_task: int = 50,
    # Eagle encoder configuration (only used when image_encoder_type=eagle)
    eagle_tune_llm: bool = False,  # Whether to tune Eagle's language model
    eagle_tune_visual: bool = True,  # Whether to tune Eagle's vision model
    resume_checkpoint: str = None,  # Path to checkpoint to resume from (e.g. ./checkpoints/libero_object/epoch_00110.pth)
):
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device not specified, auto-detected: {device}")
    else:
        logger.info(f"Using specified device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    
    # Normalize data_directory path (remove trailing slashes)
    data_directory = os.path.normpath(data_directory)
    
    if use_all_suites:
        # Determine base directory
        dir_basename = os.path.basename(data_directory)
        if dir_basename in ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10', 'libero_90']:
            base_dir = os.path.dirname(data_directory)
        else:
            base_dir = data_directory
        
        # Define all suite directories
        suite_names = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10', 'libero_90']
        suite_dirs = [os.path.join(base_dir, suite) for suite in suite_names]
        
        datasets = []
        # Use max_len_data=530 to accommodate all suites (libero_10 needs 517, libero_90 needs 373, libero_goal needs 347)
        # Add some buffer to be safe
        suite_max_len = max(max_len_data, 530)  # Use at least 530 for all suites
        
        for suite_dir in suite_dirs:
            if os.path.exists(suite_dir):
                logger.info(f"Loading suite: {os.path.basename(suite_dir)}")
                try:
                    suite_dataset = LiberoDataset(
                        data_directory=suite_dir,
                        device="cpu",
                        obs_dim=32,
                        action_dim=7,
                        state_dim=45,
                        max_len_data=suite_max_len,
                        chunck_size=10,
                        start_idx=0,
                        demos_per_task=demos_per_task,
                    )
                    datasets.append(suite_dataset)
                    logger.info(f"  Loaded {len(suite_dataset)} samples from {os.path.basename(suite_dir)}")
                except Exception as e:
                    logger.warning(f"  Failed to load {suite_dir}: {e}, skipping...")
            else:
                logger.warning(f"Suite directory not found: {suite_dir}, skipping...")
        
        if len(datasets) == 0:
            raise ValueError(f"No valid suite directories found! Checked: {suite_dirs}")
        
        # Combine all datasets with wrapper that preserves camera_names
        dataset = CombinedLiberoDataset(datasets)
        total_samples = sum(len(d) for d in datasets)
        logger.info(f"Combined dataset: {total_samples} total samples from {len(datasets)} suites")
        
    else:
        # Original single dataset loading
        # Check if data_directory is a parent directory (contains suite subdirectories)
        dir_basename = os.path.basename(data_directory)
        if dir_basename not in ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10', 'libero_90']:
            # Check if it's a parent directory containing suite subdirectories
            suite_names = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10', 'libero_90']
            found_suites = [s for s in suite_names if os.path.exists(os.path.join(data_directory, s))]
            if found_suites:
                raise ValueError(
                    f"Data directory '{data_directory}' appears to be a parent directory containing suites: {found_suites}.\n"
                    f"Please either:\n"
                    f"  1. Use --use_all_suites to load all suites, or\n"
                    f"  2. Specify a specific suite directory (e.g., {os.path.join(data_directory, 'libero_object')})"
                )
        
        dataset = LiberoDataset(
            data_directory=data_directory,
            device="cpu",
            obs_dim=32,
            action_dim=7,
            state_dim=45,
            max_len_data=max_len_data,
            chunck_size=10,
            start_idx=0,
            demos_per_task=demos_per_task,
        )
    
    # Create evaluation callback if eval_during_training is specified
    eval_callback = None
    if eval_during_training is not None:
        # Auto-enable all_suites evaluation if training on all suites
        actual_eval_all_suites = eval_all_suites
        if use_all_suites and not eval_all_suites:
            actual_eval_all_suites = True
        eval_callback = create_eval_callback(
            data_directory=data_directory,
            benchmark_type=eval_benchmark_type,
            all_suites=actual_eval_all_suites,
            num_rollouts=eval_num_rollouts,
            max_steps=eval_max_steps,
            device=device,
            use_multiprocessing=eval_use_multiprocessing,
            n_cores=eval_n_cores,
            render_image=False,  # Disable rendering during training
            save_video=False,
            save_video_dir=None,
            seed=42,
            demos_per_task=1,  # Use 1 for evaluation callback (just to determine camera setup)
        )
        if eval_callback is None:
            logger.warning("Failed to create evaluation callback. Continuing without evaluation.")
            eval_during_training = None
    
    # Set default betas if not provided
    if betas is None:
        betas = [0.9, 0.9]  # Match working version: OptimizerConfig.betas = [0.9, 0.9]
    
    # Train (model creation and policy setup handled internally)
    train_policy(
        dataloader=dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        image_encoder_type=image_encoder_type,  # Pass image encoder type
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        obs_tok_len=obs_tok_len,
        action_seq_len=action_seq_len,
        save_dir=save_dir,
        save_freq=save_freq,
        enable_ema=enable_ema,
        enable_data_scaling=enable_data_scaling,
        data_scaler_type=data_scaler_type,
        dataloader_workers=num_workers,  # Pass num_workers to trainer
        eval_during_training=eval_during_training,
        eval_callback=eval_callback,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        # Pass optimizer config parameters to match working version
        transformer_weight_decay=transformer_weight_decay,
        obs_encoder_weight_decay=obs_encoder_weight_decay,
        betas=betas,
        sampling_steps=sampling_steps,
        ema_decay_rate=ema_decay_rate,  # Pass EMA decay rate to trainer
        # Pass Eagle encoder configuration
        tune_llm=eagle_tune_llm,
        tune_visual=eagle_tune_visual,
        resume_checkpoint=resume_checkpoint,
    )


# Compute config path relative to this file
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf")

@hydra.main(version_base=None, config_path=_config_path, config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration.
    
    You can override any config value from command line, e.g.:
        python -m libero_bench.train batch_size=512 latent_dim=512
        python -m libero_bench.train --config-name=config_libero_object
        python -m libero_bench.train --config-name=config_all_suites wandb.name=my_experiment
    """
    # Validate configuration
    if not cfg.use_all_suites and not cfg.data_directory:
        raise ValueError("data_directory is required unless use_all_suites is True")
    
    # Handle wandb enabled/disabled
    wandb_project = cfg.wandb.project if cfg.wandb.enabled else None
    wandb_entity = cfg.wandb.entity if cfg.wandb.enabled else None
    wandb_name = cfg.wandb.name if cfg.wandb.enabled else None
    
    # Convert betas list if it's a list, otherwise keep as is
    betas = cfg.betas
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        betas = list(betas)
    else:
        betas = [0.9, 0.9]
    
    train(
        data_directory=cfg.data_directory,
        use_all_suites=cfg.use_all_suites,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        device=cfg.device,
        image_encoder_type=cfg.image_encoder_type,  # Pass image encoder type from config
        latent_dim=cfg.latent_dim,
        embed_dim=cfg.embed_dim,
        n_layer=cfg.n_layer,
        d_intermediate=cfg.d_intermediate,
        save_dir=cfg.save_dir,
        save_freq=cfg.save_freq,
        max_len_data=cfg.max_len_data,
        enable_ema=cfg.enable_ema,
        ema_decay_rate=cfg.ema_decay_rate,
        enable_data_scaling=cfg.enable_data_scaling,
        data_scaler_type=cfg.data_scaler_type,
        num_workers=cfg.num_workers,
        transformer_weight_decay=cfg.transformer_weight_decay,
        obs_encoder_weight_decay=cfg.obs_encoder_weight_decay,
        betas=betas,
        sampling_steps=cfg.sampling_steps,
        eval_during_training=cfg.eval_during_training,
        eval_benchmark_type=cfg.eval_benchmark_type,
        eval_all_suites=cfg.eval_all_suites,
        eval_num_rollouts=cfg.eval_num_rollouts,
        eval_max_steps=cfg.eval_max_steps,
        eval_use_multiprocessing=cfg.eval_use_multiprocessing,
        eval_n_cores=cfg.eval_n_cores,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        demos_per_task=cfg.demos_per_task,
        # Pass Eagle encoder configuration from config
        eagle_tune_llm=cfg.get('eagle_tune_llm', False),
        eagle_tune_visual=cfg.get('eagle_tune_visual', True),
        resume_checkpoint=cfg.get('resume_checkpoint', None),
    )


if __name__ == "__main__":
    main()

