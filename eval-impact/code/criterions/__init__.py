import importlib
import warnings

criterion_list = {}


def _register(name, module, attr):
    """Import criterion lazily; skip (with a warning) if optional deps are missing.

    Some criterions pull heavy optional deps (e.g. MCW_KD needs cvxpy). A missing
    dep for a criterion we are not using must not break the whole registry.
    """
    try:
        mod = importlib.import_module(f".{module}", package=__name__)
        criterion_list[name] = getattr(mod, attr)
    except Exception as e:  # noqa: BLE001 - optional criterion, report and continue
        warnings.warn(f"criterion '{name}' unavailable ({module}.{attr}): {e}")


_register("cross_entropy", "cross_entropy_loss", "CrossEntropyLoss")
_register("various_divergence", "various_divergence", "VariousDivergence")
_register("dual_space_kd_with_cma", "DSKD", "DualSpaceKDWithCMA")
_register("DSKD_IMPACT", "DSKD_IMPACT", "DualSpaceKDWithIMPACT")
_register("dual_space_kd_with_cma_impact", "DSKD_IMPACT", "DualSpaceKDWithIMPACT")
_register("universal_logit_distillation", "ULD", "UniversalLogitDistillation")
_register("min_edit_dis_kld", "MinED", "MinEditDisForwardKLD")
_register("MCW_KD", "MCW_KD", "MCW_KD")
_register("MCW_KD_Dual", "MCW_KD_Dual", "MCW_KD_Dual")
_register("MultiLevelOT", "MultiLevelOT", "MultiLevelOT")
_register("SRA", "SRA", "SRA")
_register("ALM", "ALM", "ALM")
_register("ResidualKD", "ResidualKD", "ResidualKD")
_register("IMPACT", "IMPACT", "IMPACT")
_register("SRA_IMPACT", "SRA_IMPACT", "SRA_IMPACT")
_register("ALM_IMPACT", "ALM_IMPACT", "ALM_IMPACT")
_register("ResidualKD_IMPACT", "ResidualKD_IMPACT", "ResidualKD_IMPACT")
_register("E_SCOT", "E_SCOT", "E_SCOT")
_register("E_SCOT_IMPACT", "E_SCOT_IMPACT", "E_SCOT_IMPACT")


def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")
