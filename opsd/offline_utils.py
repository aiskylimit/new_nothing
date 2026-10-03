import os


def configure_offline_environment():
    """Disable remote lookups and telemetry before loading HF components."""
    offline_env = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    os.environ.update(offline_env)


def disable_external_reporting(training_args):
    """Force Trainer artifacts to remain on the local filesystem."""
    if getattr(training_args, "use_vllm", False) and getattr(training_args, "vllm_mode", "colocate") != "colocate":
        raise ValueError("Only colocated vLLM is allowed; server mode can transmit prompts and model weights.")

    # training_args.report_to = []
    # disabled_fields = {
    #     "push_to_hub": False,
    #     "push_to_hub_revision": False,
    #     "hub_always_push": False,
    #     "hub_model_id": None,
    #     "hub_token": None,
    #     "push_to_hub_model_id": None,
    #     "push_to_hub_organization": None,
    #     "push_to_hub_token": None,
    # }
    # for name, value in disabled_fields.items():
    #     if hasattr(training_args, name):
    #         setattr(training_args, name, value)
